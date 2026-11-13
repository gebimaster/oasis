"""历史压缩（M3 第 3 步）：摘要 + 水位，原文一行不改。

流程：选切点 k → LLM 把「旧摘要 + history[:k]」写成新摘要 → CAS 提交 → 循环改用 (新摘要, history[k:])。
切点只能落在「真正的用户提问」上：它左边的 tool_use/tool_result 全部成对，切不开。
O4：一个提问连调很多次工具时没有这样的切点，改成「Q 原样 + 中间摘要 + 最近一组原样」；
水位指向最近一组，Q 被它跳过，另存副本 replay_prefix，下个 run 拼回摘要之后。
这里是 LLM 摘要这一级；更便宜的清旧工具结果在 microcompact，排在它前面（O9）。
"""

import functools
import json
import logging
import re
from typing import Awaitable, Callable
from uuid import UUID

from app import budget, context, db, llm, tools

log = logging.getLogger(__name__)

SUMMARY_MAX_TOKENS = 4000  # 摘要调用的输出上限（要给 DeepSeek 的 thinking 留余量）
MAX_SUMMARY_RETRIES = 3  # 摘要请求自己超窗时，缩短摘要源重试的次数
MIN_SUMMARY_CHARS = 8  # 比这还短不像摘要

# P1-C2：固定段落（参照 opencode core/src/session/compaction.ts 的 SUMMARY_TEMPLATE）。
# 「关键标识与 ref」不让模型写，由代码从工具调用生成（with_refs），模型抄 16 位哈希会抄错
SUMMARY_TEMPLATE = """按下面的格式写，某段没有内容就写「（无）」，不要输出 <template> 标签：
<template>
## 目标
- 用户要完成什么，一两句
## 约束与偏好
- 用户明确提出的要求、偏好、范围边界
## 已定决定
- 已经确定的结论和理由
## 进度
### 已完成
### 进行中
### 受阻
## 下一步
1. 按顺序列出
## 未回答的用户请求
- 还没答复、或还要继续做的用户请求，保留原话；已经答复的不写
</template>
规则：每段用简短的单行要点；名字、数字、ID、文件名、URL 原样保留；不要编造；不要写「关键标识与 ref」这一段，系统会自动补上；\
不要提到压缩或摘要过程。"""
# 增量合并（opencode 的 update prompt）：旧摘要 + 新历史 → 一份，新的优先，没被推翻的照抄，不整篇重写
MERGE_RULES = ("开头 <summary> 里是之前的摘要。把它和之后的对话合并成一份新摘要：新信息优先；旧摘要里没有被推翻、解决或过时的内容原样保留；"
               "做完的事从「进行中」挪到「已完成」，已经答复的请求从「未回答的用户请求」里删掉。")
SUMMARIZER_SYSTEM = "你是对话压缩器。把给出的对话整理成结构化的中文摘要，供助手继续对话时参考，不要回答对话里的问题。\n" + SUMMARY_TEMPLATE


# P1-C1：摘要请求复用正常请求的前缀，压缩指令代替运行时尾部。只能写在指令里，不能设 tool_choice none：
# 11-03 实测 none 会让 DeepSeek 把工具定义从 prompt 里拿掉，后面整段历史 0 命中
def compact_note(merge: bool, retry: bool = False) -> str:
    """merge：有旧摘要时用合并规则；retry：上一次调了工具，补一句更强的提醒（C1 选 A）。"""
    return ("<compact_request>" + ("上一次你调用了工具，压缩阶段不执行工具，这一次只输出摘要正文。" if retry else "")
            + "请暂停当前任务，不要调用任何工具，也不要回答对话里的问题。"
            + (MERGE_RULES if merge else "把以上对话整理成结构化摘要，供你之后继续工作时参考。")
            + "\n" + SUMMARY_TEMPLATE + "\n只输出摘要正文。</compact_request>")


REFS_HEADING = "## 关键标识与 ref"
MAX_REFS = 30  # 只留最近的 30 条调用，旧的丢掉（摘要不能无限长）
_REF = re.compile(r"ref=(out_[0-9a-f]{16})")


def tool_log(messages: list[dict]) -> list[str]:
    """从摘要源的原文里逐个读工具调用：真名 + 入参，结果被截断过带 ref，失败标「失败」。同样的消息永远得到同样的行。"""
    results = {b.get("tool_use_id"): b for m in messages if m["role"] == "user" and isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"}
    lines = []
    for m in messages:
        for b in m["content"] if m["role"] == "assistant" and isinstance(m["content"], list) else []:
            if b.get("type") != "tool_use":
                continue
            name, args = b.get("name"), b.get("input")
            if name == tools.CALL_TOOL and isinstance(args, dict) and isinstance(args.get("tool_name"), str):
                name, args = args["tool_name"], args.get("arguments")  # 延迟工具：记真名
            if name == tools.READ_TOOL:  # 回取本身不是新信息，ref 已经在发起它的那一行里
                continue
            arg_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
            line = f"- {name} {arg_text[:120] + '…' if len(arg_text) > 120 else arg_text}"
            r = results.get(b.get("id"))
            text = str(r.get("content")) if r else ""
            if r and (r.get("is_error") or text.startswith(tools.ERROR_PREFIX)):
                line += " → 失败"
            elif ref := _REF.search(text):
                line += f" → ref={ref.group(1)}"
            lines.append(line)
    return lines


def split_refs(summary: str | None) -> tuple[str, list[str]]:
    """拆成（模型写的部分, ref 段的行）。ref 段总在最后。"""
    head, _, tail = (summary or "").partition(REFS_HEADING)
    return head.strip(), [ln for ln in tail.splitlines() if ln.startswith("- ")]


def with_refs(model_text: str, old_summary: str | None, source: list[dict]) -> str:
    """模型写的 ref 段整段丢掉；旧摘要的 ref 段 + 这次摘要源的调用 → 去重（留最后一次）、保序、只留最近 MAX_REFS 条。"""
    body, _ = split_refs(model_text)
    lines = split_refs(old_summary)[1] + tool_log(source)
    merged = list(reversed(dict.fromkeys(reversed(lines))))[-MAX_REFS:]
    return f"{body}\n\n{REFS_HEADING}\n" + "\n".join(merged) if merged else body
# (旧摘要, 摘要源, 压缩指令) → {"system", "messages", "tools"}：由循环提供，和它的正常请求同一套拼法
Prompt = Callable[[str | None, list[dict], str], dict]


class SummaryIncomplete(Exception):
    """摘要没写完（被截断 / 空 / 太短）：和摘要调用出错一样处理，这次不压。"""


def is_real_user(m: dict) -> bool:
    """真正的用户提问：role=user 且不含 tool_result（tool_result 也放在 user 消息里）。
    O7：循环插的续写 / 空回复提示也不算，否则切点可能落在它前面，被截断的前半截进了摘要，续写就接不上了。"""
    if m["role"] != "user" or context.note_kind(m):
        return False
    return isinstance(m["content"], str) or not any(b.get("type") == "tool_result" for b in m["content"])


def choose_cut(history: list[dict], limit: int) -> int | None:
    """返回最小的 k，使 history[k] 是真提问且 history[k:] 估算 ≤ limit；k=0 不算（没东西可压）。"""
    sizes = [budget.estimate_json(m["content"]) for m in history]
    suffix = sum(sizes)
    for k in range(len(history)):
        if k > 0 and is_real_user(history[k]) and suffix <= limit:
            return k
        suffix -= sizes[k]
    return None  # 没有真提问切点：比如本轮之内的工具结果本身就超了，交给 O4 的单轮切法


def tail_start(history: list[dict]) -> int:
    """保护区起点：最近一条 assistant 连同它的工具结果；
    最近的是真提问（还没有 assistant 回应它）就从这条提问开始。保护区原样保留，不进摘要。"""
    for i in range(len(history) - 1, -1, -1):
        if history[i]["role"] == "assistant" or is_real_user(history[i]):
            return i
    return 0


def current_question(history: list[dict]) -> tuple[int, int | None]:
    """O4：返回 (保护区起点 start, 当前提问 q)，compact 和 temporary_window 共用，规则只写这一处。
    q 是 start 及之前最近的真提问；q == start 表示保护区本身就从当前提问开始，不用另留。"""
    start = tail_start(history)
    return start, next((i for i in range(start, -1, -1) if is_real_user(history[i])), None)


def _ids(m: dict | None, kind: str, key: str) -> set:
    blocks = m["content"] if m and isinstance(m["content"], list) else []
    return {b.get(key) for b in blocks if b.get("type") == kind}


def pairs_ok(history: list[dict]) -> bool:
    """配对不变量：每条 tool_result 都在发起它的 assistant 的下一条里；每个 tool_use 下一条都有结果（最后一条除外）。"""
    for i, m in enumerate(history):
        prev = history[i - 1] if i else None
        results = _ids(m, "tool_result", "tool_use_id")
        if results and not (prev and prev["role"] == "assistant" and results <= _ids(prev, "tool_use", "id")):
            return False
        uses = _ids(m, "tool_use", "id") if m["role"] == "assistant" else set()
        if uses and i + 1 < len(history) and not uses <= _ids(history[i + 1], "tool_result", "tool_use_id"):
            return False
    return True


def accepts(kept: list[dict], protected: list[dict]) -> bool:
    """采用规则里的两条结构检查：配对完整；受保护的消息按原顺序、逐字（role + content）都还在。"""
    rest = iter((m["role"], m["content"]) for m in kept)  # 子序列判断：in 会一路消耗迭代器
    return pairs_ok(kept) and all((p["role"], p["content"]) in rest for p in protected)


def temporary_window(history: list[dict]) -> list[dict] | None:
    """O4 超窗兜底：只留当前真提问 + 保护区，更早的原文这次不发。
    不摘要、不落库；形状和单轮投影一样，同样过配对和逐字检查。没有可丢的就返回 None。"""
    start, q = current_question(history)
    protected = ([history[q]] if q is not None and q < start else []) + history[start:]
    if len(protected) == len(history) or not accepts(protected, protected):
        return None
    return protected


def render(messages: list[dict]) -> str:
    """转成纯文本给摘要模型看；thinking 不进摘要。"""
    lines = []
    for m in messages:
        blocks = [{"type": "text", "text": m["content"]}] if isinstance(m["content"], str) else m["content"]
        for b in blocks:
            if b.get("type") == "text":
                lines.append(f"[{m['role']}] {b['text']}")
            elif b.get("type") == "tool_use":
                lines.append(f"[assistant 调用工具] {b['name']}({json.dumps(b['input'], ensure_ascii=False)})")
            elif b.get("type") == "tool_result":
                lines.append(f"[工具结果] {b.get('content')}")
    return "\n".join(lines)


def _checked(text: str, stop: str | None) -> str:
    if stop != "end_turn" or len(text) < MIN_SUMMARY_CHARS:
        raise SummaryIncomplete(f"stop_reason={stop}, {len(text)} chars")
    return text


async def summarize_alone(old_summary: str | None, messages: list[dict]) -> str:
    """单独摘要器：自己的 system + 渲染成纯文本，和正常请求没有公共前缀（缓存 0 命中）。C1 起只做退路。"""
    prefix = f"<summary>\n{old_summary}\n</summary>\n{MERGE_RULES}\n\n之后的对话：\n" if old_summary else "对话：\n"
    text, stop = await llm.complete_with_stop(system=SUMMARIZER_SYSTEM, prompt=prefix + render(messages),
                                              max_tokens=SUMMARY_MAX_TOKENS)
    return _checked(text, stop)


async def summarize(old_summary: str | None, messages: list[dict], prompt: Prompt | None = None,
                    stats: dict | None = None) -> str:
    """C1：同 system、同 tools、tool_choice auto，只把运行时尾部换成压缩指令。
    模型去调工具 → 丢掉这次回复，换更强的指令重发一次（前缀不变，仍命中）→ 还调就退回单独摘要器。
    没有 prompt（单元测试直接调 compactor）就只走单独摘要器。
    C7：stats 记走了哪条路（prefix / retry / alone）和摘要请求自己的 usage（单独摘要器拿不到 usage）。"""
    stats = {} if stats is None else stats
    stats["mode"] = "alone"
    stats.setdefault("usages", [])  # 摘要请求自己超窗、缩短重试时会调几次，usage 累加
    if prompt is None:
        return await summarize_alone(old_summary, messages)
    for retry in (False, True):
        note = compact_note(merge=bool(old_summary), retry=retry)
        req = prompt(old_summary, messages, note)
        final = None
        async for ev in llm.stream_message(system=req["system"], messages=req["messages"], tools=req["tools"],
                                           tool_choice={"type": "auto"}):
            if ev["type"] == "final":
                final = ev
        stats["usages"].append(final.get("usage") or {})
        if final["stop_reason"] != "tool_use":  # 调了工具的回复一个字都不用：不执行、不进请求、不进摘要
            stats["mode"] = "retry" if retry else "prefix"
            return _checked(context.text_of(final["content"]).strip(), final["stop_reason"])
        log.warning("summary request answered with tool_use (%s)", "fallback" if retry else "retrying")
    return await summarize_alone(old_summary, messages)


def summary_budget() -> int:
    """O4：摘要源（旧摘要 + 渲染后的对话）的上限。窗口算的是输入 + max_tokens（实测），
    扣掉摘要输出和摘要器 system，再按目标线比例留余量：本地估算有误差。"""
    window = budget.context_window()  # 摘要用的也是 MODEL，窗口同一个来源
    return int((window - SUMMARY_MAX_TOKENS - budget.estimate_text(SUMMARIZER_SYSTEM)) * budget.TARGET_RATIO)


def snap_cut(history: list[dict], j: int) -> int:
    """推迟切点往前退到一组的开头（assistant 或真提问），不拆配对；回答留在原文时，紧挨着的提问一起留下
    返回 0 表示一条都放不下。"""
    while j > 0 and not (history[j]["role"] == "assistant" or is_real_user(history[j])):
        j -= 1
    if j > 0 and history[j]["role"] == "assistant" and is_real_user(history[j - 1]):
        j -= 1
    return j


def fit_source(old_summary: str | None, history: list[dict], end: int) -> int:
    """O4 推迟尾部：history[:end] 是摘要源，返回放得下的最长前缀长度（放得下全部就是 end）。"""
    used, limit = budget.estimate_text(old_summary or ""), summary_budget()
    for i in range(end):
        used += budget.estimate_text(render([history[i]]))
        if used > limit:
            return snap_cut(history, i)
    return end


def strip_usage(history: list[dict]) -> list[dict]:
    """压缩后这些消息的 usage 是压缩前那份大请求的，拿来打底会立刻再触发压缩。"""
    return [{**m, "usage": None} for m in history]


MIN_PROGRESS = 0.1  # C4：压完至少省 10%
FALLBACK_HEADING = "## 降级记录（模型摘要失败，由代码按原话生成，未经整理）"
USER_CHARS, ANSWER_CHARS = 2000, 800  # 降级记录里每条用户原话 / 每轮回答最多留多少字（粘贴整篇文档时不能让它自己超窗）


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else f"{text[:n]}……（后面还有 {len(text) - n} 字，原文可用 recall 搜）"


def fallback_summary(old_summary: str | None, messages: list[dict]) -> str:
    """C4（Cline 的 basic 策略）：不调模型。旧摘要正文原样留着；之后每个回合留用户原话 + 这一回合最后一条有字的回答，
    工具调用和结果全删（ref 段由 with_refs 另补）。下次模型摘要成功时，它是「旧摘要」，按 C2 的规则合并进去。"""
    lines, answer = [], None

    def flush():
        if answer:
            lines.append(f"- 助手：{_clip(answer, ANSWER_CHARS)}")
    for m in messages:
        if is_real_user(m):
            flush()
            lines.append(f"- 用户：{_clip(context.text_of(m['content']), USER_CHARS)}")
            answer = None
        elif m["role"] == "assistant" and (t := context.text_of(m["content"]).strip()):
            answer = t  # 同一回合里后面的覆盖前面的：只留最后一条
    flush()
    body = split_refs(old_summary)[0]
    return (f"{body}\n\n" if body else "") + FALLBACK_HEADING + "\n" + "\n".join(lines)


Measure = Callable[[str | None, list[dict]], int]  # (摘要, 历史) → 整份请求的估算 token，由循环提供
Compactor = Callable[..., Awaitable[tuple[str, list[dict]] | None]]  # (历史, 摘要, 额度, measure, prompt=)


def make_compactor(conversation_id: UUID, revision: int,
                   summarize_fn: Callable | None = None) -> Compactor:
    """runner 注入给循环用：循环本身不碰数据库。revision 是加载历史时读到的值。
    O4：压缩是投机修改，采用规则（整份请求变小、配对完整、保护区逐字不变）全部满足才提交或返回。"""
    state = {"revision": revision}

    async def compact(history: list[dict], summary: str | None, limit: int, measure: Measure,
                      prompt: Prompt | None = None, raw: list[dict] | None = None, fallback: bool = False):
        stats = compact.stats = {}  # C7：这次压缩走了哪条路、摘要请求的 usage；循环把它并进 compacted 事件
        start, q = current_question(history)
        k = choose_cut(history, limit)
        if k is None and 0 < start == q:
            k = start  # 额度放不下整轮，也至少切到当前提问：能落库的切点优先
        if k is not None:
            end, q = k, None  # 摘要源是 history[:end]；当前提问（若有）在 history[k:] 里原样保留
            protected = history[start:]
        else:
            # O4 单轮内：Q 原样 + 中间的工具轮次摘要掉 + 保护区原样。Q 也进摘要输入，摘要才知道这些调用是为什么。
            # 落库时 keep_from_id 指向保护区起点，Q 被它跳过，所以另存一份副本（replay_prefix）
            if all(i == q for i in range(start)):  # 走到这里 q < start（或 start == 0，什么都没得压）
                log.warning("conversation %s over budget but nothing to summarize", conversation_id)
                return None
            end = start
            protected = ([history[q]] if q is not None else []) + history[start:]
        # O4 第 3 步：摘要源放不下就推迟尾部（只摘要放得下的前缀，推迟的原文原样留着，下次再压；不分段合并）
        j, retries = fit_source(summary, history, end), 0
        while j > 0:  # 耗时几秒，不在事务里；运行时才查 summarize，测试可以替换
            try:
                fn = summarize_fn or functools.partial(summarize, prompt=prompt, stats=stats)
                new_summary = await fn(summary, history[:j])
                # C2：ref 段由代码生成。history 可能是 O9 投影（占位符里没有 ref），从原文 raw 读，两者一一对应
                new_summary = with_refs(new_summary, summary, (raw or history)[:j])
                break
            except (llm.LLMError, SummaryIncomplete) as e:  # 摘要失败不等于 run 失败：这次不压、照发
                retries += 1  # 只有摘要请求自己超窗值得再试：缩短摘要源
                overflow = llm.classify_error(e) == "context_overflow" and retries <= MAX_SUMMARY_RETRIES
                j = snap_cut(history, j - max(1, j // 5)) if overflow else 0
                log.warning("conversation %s summary failed (%s), retry=%s", conversation_id, e, j > 0)
        if j == 0 and fallback and end > 0:  # C4：模型摘要失败、请求已经超窗：不调模型，确定性降级，同样验收和落库
            j = end
            new_summary = with_refs(fallback_summary(summary, history[:end]), summary, (raw or history)[:end])
            stats.update(mode="fallback")
            log.warning("conversation %s summary failed, using deterministic fallback", conversation_id)
        if j == 0:
            return None
        prefix = [history[q]] if q is not None and q < j else []  # Q 被推迟时它本来就在原文里
        kept, cursor = prefix + history[j:], history[j]
        before, after = measure(summary, history), measure(new_summary, kept)
        # C4 最小进展守卫（OpenHands minimum_progress）：省不到 10% 不压，白丢缓存、下一轮还要再压
        if not (after <= before * (1 - MIN_PROGRESS) and accepts(kept, protected)):  # 先验收再提交
            log.warning("conversation %s compaction rejected: %s -> %s tokens", conversation_id, before, after)
            return None
        if cursor.get("id") is None:  # 游标还没落库（测试里绕过 runner 直接跑循环）：只在本 run 内生效
            return new_summary, strip_usage(kept)
        copies = [{"role": m["role"], "content": m["content"]} for m in prefix]
        if await db.commit_compaction(conversation_id, new_summary, cursor["id"], state["revision"], copies):
            state["revision"] += 1
            return new_summary, strip_usage(kept)
        # 抢输了：丢掉自己的摘要，改用赢家的水位和副本（自己历史里的旧副本不再要）
        won = await db.get_compaction(conversation_id)
        state["revision"] = won["revision"]
        kept = [{**m, "replayed": True} for m in won["replay_prefix"]] + [
            m for m in history if not m.get("replayed") and (m.get("id") is None or m["id"] >= won["keep_from_id"])]
        if not pairs_ok(kept):  # 没 id 的消息无条件留下，它的调用可能被赢家切掉：这次不压，照发（下个 run 用赢家的）
            log.warning("conversation %s lost compaction race, rebuilt history breaks pairs", conversation_id)
            return None
        return won["summary"], strip_usage(kept)

    compact.stats = {}
    return compact
