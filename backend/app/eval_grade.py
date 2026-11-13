"""M8 第 1 步：端到端评测的评分器。纯函数 + 一个可注入的评委，跑评测的脚本在 scripts/agent_eval.py。

用户的选择：① 写操作看库里的执行记录，不看模型怎么说 ② 确定性断言当硬门槛，忠实度交给 LLM 评委
（评委用 deepseek-v4-pro 评 deepseek-flash，给它本次 run 的检索原文）③ 每题跑 3 次。
评委全挂了不记 0 分，而是记 judge_error、不进分母，
「回答差」和「评委坏了」分开算。
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app import citations

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-flash")  # 用户：pro 太贵，评委也用 flash（自评偏差靠引文复核 + 人工抽样兜）


@dataclass
class Trial:
    """一次 trial 收集到的事实，全部从库里读（runner 不信模型的自述）。"""
    answer: str = ""                                    # 最后一段回答（O7：续写过的拼回来，context.final_answer）
    calls: list[dict] = field(default_factory=list)     # [{name, input, is_error, result}]，call_tool 已拆包
    stop_reasons: list[str] = field(default_factory=list)  # 本题各个父 run 的 stop_reason，按时间顺序
    ledger: list[dict] = field(default_factory=list)    # [{to_name, amount}]
    approvals: list[dict] = field(default_factory=list)  # tool_approval 行 [{tool_name, args, status}]
    children: list[dict] = field(default_factory=list)  # 子 run [{status, start, end}]，start/end 是秒
    compacted: bool = False                             # 本题的 run 里发生过压缩
    continuations: int = 0                              # O7：本题因 max_tokens 续写的次数（库里 continue 提示的条数）


def _subset(want: dict, got: dict) -> bool:
    """断言里写了的参数必须相等（数字按值比），没写的不管：模型多填个可选参数不算错。"""
    return all(k in got and (got[k] == v or (isinstance(v, (int, float)) and got[k] == float(v)))
               for k, v in want.items())


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _has(text: str, item) -> bool:  # item 是字符串，或同义词列表（命中任一即可）
    return any(_norm(x) in _norm(text) for x in ([item] if isinstance(item, str) else item))


# ---------- 确定性断言：每个返回 (通过?, 说明) ----------

def tool_called(t: Trial, name: str, args: dict | None = None) -> tuple[bool, str]:
    ok = [c for c in t.calls if c["name"] == name and not c["is_error"] and _subset(args or {}, c["input"])]
    return bool(ok), f"{name} 成功调用 {len(ok)} 次"


def tool_not_succeeded(t: Trial, name: str) -> tuple[bool, str]:
    n = sum(1 for c in t.calls if c["name"] == name and not c["is_error"])
    return n == 0, f"{name} 成功 {n} 次"


def tool_attempted(t: Trial, name: str) -> tuple[bool, str]:
    """O6 复核丙：试过这个工具（失败也算）。tool_called 只认成功的；工具挂掉的题要确认模型真去调了，没有直接猜。"""
    n = sum(1 for c in t.calls if c["name"] == name)
    return n > 0, f"{name} 调用 {n} 次（含失败）"


def result_has(t: Trial, name: str, text: str) -> tuple[bool, str]:
    """工具结果里有某句话：如 kb_search 返回了干净的 not_found。"""
    ok = any(c["name"] == name and text in str(c["result"]) for c in t.calls)
    return ok, f"{name} 结果{'含' if ok else '不含'}「{text}」"


def answer_has(t: Trial, all_of: list) -> tuple[bool, str]:
    miss = [x for x in all_of if not _has(t.answer, x)]
    return not miss, f"缺 {miss}" if miss else "关键词齐全"


def answer_lacks(t: Trial, patterns: list[str]) -> tuple[bool, str]:
    hit = [p for p in patterns if re.search(p, t.answer)]
    return not hit, f"出现了 {hit}" if hit else "没有禁止内容"


def stop_reason(t: Trial, value: str, index: int = 0) -> tuple[bool, str]:
    got = t.stop_reasons[index] if len(t.stop_reasons) > index else None
    return got == value, f"第 {index + 1} 个 run 的 stop_reason={got}"


def ledger(t: Trial, count: int, amount: float | None = None, to: str | None = None) -> tuple[bool, str]:
    rows = [r for r in t.ledger if (amount is None or float(r["amount"]) == amount)
            and (to is None or r["to_name"] == to)]
    return len(t.ledger) == count and len(rows) == count, f"ledger {len(t.ledger)} 行，符合条件 {len(rows)} 行"


def approval(t: Trial, tool: str, args: dict, status: str | None = None) -> tuple[bool, str]:
    rows = [a for a in t.approvals if a["tool_name"] == tool and _subset(args, a["args"])
            and (status is None or a["status"] == status)]
    return len(rows) == 1, f"符合条件的审批记录 {len(rows)} 行"


def children(t: Trial, count: int, overlap: bool = True) -> tuple[bool, str]:
    done = [c for c in t.children if c["status"] == "completed"]
    # 并行 = 最晚开始的那个，早于最早结束的那个
    par = len(done) < 2 or max(c["start"] for c in done) < min(c["end"] for c in done)
    return len(done) == count and (par or not overlap), f"完成的子 run {len(done)} 个，并行={par}"


def compacted(t: Trial) -> tuple[bool, str]:
    return t.compacted, "发生了压缩" if t.compacted else "没有压缩"


def continued(t: Trial, min_count: int = 1) -> tuple[bool, str]:
    """O7 机制断言：真的撞上了 max_tokens 并续写过。没撞上的话这道题测不到续写，单列成机制未达标。"""
    return t.continuations >= min_count, f"续写 {t.continuations} 次"


CHECKS: dict[str, Callable[..., tuple[bool, str]]] = {
    f.__name__: f for f in (tool_called, tool_attempted, tool_not_succeeded, result_has, answer_has, answer_lacks,
                            stop_reason, ledger, approval, children, compacted, continued)}

# ---------- LLM 评委：忠实度 ----------

JUDGE_SYSTEM = """你是严格的事实核查员。给你「检索原文」和「回答」，把回答拆成若干条事实性断言，逐条打标签：
- direct：原文直接写了
- inferred：由原文的一处或几处内容经过简单推理得出（如「22 点后全额」→「22 点前按普通标准」）
- unsupported：原文找不到依据、与原文矛盾，或加了原文没有的数字、条件、风险、建议
不算断言、不要列出：当前时间日期、复述用户的问题、客套话、出处标注。不要因为回答写得长或语气肯定就放宽。
只输出 JSON：{"claims": [{"claim": "...", "label": "direct|inferred|unsupported",
"quote": ["原文中逐字摘录的依据（推论写出所有前提）"]}]}，unsupported 的 quote 为空列表。"""

OK_LABELS = ("direct", "inferred")


def parse_verdict(raw: str, evidence: str) -> dict:
    """解析评委输出（用户选 B 三档：只有 unsupported 判失败，推论单独计数不扣分），
    并用代码复核引文：标了 direct / inferred 但引文不全在原文里的，按 unsupported 算（防评委手松）。"""
    start = raw.find("{")
    try:  # raw_decode 从第一个 { 开始读一个完整 JSON，后面再有解释文字也不影响（贪婪正则会把后文的 } 吞进来）
        claims = json.JSONDecoder().raw_decode(raw[start:])[0]["claims"] if start >= 0 else None
    except (json.JSONDecodeError, KeyError, TypeError):
        claims = None
    if not isinstance(claims, list):
        return {"passed": None, "error": "judge_parse_error", "raw": raw[:2000]}
    src = _norm(evidence)

    def verified(c: dict) -> bool:
        q = c.get("quote")
        quotes = [q] if isinstance(q, str) else (q if isinstance(q, list) else [])
        return c.get("label") in OK_LABELS and bool(quotes) and all(_norm(str(x)) in src for x in quotes)
    bad = [c.get("claim") for c in claims if not verified(c)]
    inferred = [c.get("claim") for c in claims if verified(c) and c["label"] == "inferred"]
    return {"passed": not bad, "unsupported": bad, "inferred": inferred, "n_claims": len(claims)}


async def judge_faithful(answer: str, evidence: str, complete: Callable[..., Awaitable[str]]) -> dict:
    """complete 注入：测试用假评委，真跑用 llm.complete(model=JUDGE_MODEL)。评委本身出错 → judge_error。"""
    if not evidence:
        return {"passed": False, "unsupported": ["没有检索原文：没调 kb_search 却给了答案"]}
    try:
        raw = await complete(system=JUDGE_SYSTEM, prompt=f"【检索原文】\n{evidence}\n\n【回答】\n{answer}")
    except Exception as e:  # 网络、限流等：不是被测回答的错
        return {"passed": None, "error": f"judge_error: {type(e).__name__}"}
    return parse_verdict(raw, evidence)


def _combine(flags: list[bool | None]) -> bool | None:
    return False if False in flags else (None if None in flags else True)


async def grade(case: dict, t: Trial, complete: Callable[..., Awaitable[str]] | None = None,
                disabled: frozenset[str] = frozenset()) -> dict:
    """一次 trial 的判定。用户选 (b)：断言分两种——
    结果断言（用户要的东西拿到没有）决定 passed；机制断言（带 mechanism=功能名，如 subagents）单列，
    本配置关掉了那个功能就跳过。确定性断言有一条失败就是失败；全过但评委出错 → None（不进分母）。"""
    outcome, mech = [], []
    for chk in case["checks"]:
        feat = chk.get("mechanism")
        if feat in disabled:
            continue
        if chk["kind"] == "code":
            ok, why = CHECKS[chk["name"]](t, **chk.get("args", {}))
            g = {"check": chk["name"], "passed": ok, "detail": why}
        else:  # model：目前只有忠实度，原文 = 本次 trial 里 kb_search 返回的全部内容
            results = [str(c["result"]) for c in t.calls if c["name"] == "kb_search" and not c["is_error"]]
            g = {"check": "faithful", **await judge_faithful(t.answer, "\n".join(results), complete),
                 # O8：引用校验和评委并列记录、不改 passed：评委判「有没有依据」，它判「标的出处对不对」
                 "citations": citations.check(t.answer, results)}
        (mech if feat else outcome).append(g | ({"mechanism": feat} if feat else {}))
    return {"passed": _combine([g["passed"] for g in outcome]),
            "mechanism": _combine([g["passed"] for g in mech]) if mech else None, "grades": outcome + mech}


def pass_metrics(flags: list[bool | None]) -> dict:
    """一道题 k 次 trial 的指标：pass@1 = 通过比例（不是只看第一次），
    pass^k = 每次都过；None（评委出错）不进分母，全是 None 就是这题没评成。"""
    ok = [f for f in flags if f is not None]
    if not ok:
        return {"pass@1": None, "pass^k": None, "eligible": 0, "judge_errors": len(flags)}
    return {"pass@1": round(sum(ok) / len(ok), 3), "pass^k": all(ok), "eligible": len(ok),
            "judge_errors": len(flags) - len(ok)}
