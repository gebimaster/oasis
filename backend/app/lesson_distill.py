"""P4 第 3 步：提炼器。把一段有信号的轨迹写成结构化假设（FGSE）：前提 → 行为 → 预期效果 → 验证方式。

- 输入只有 lesson_trace.render 给的结构化轨迹（用户选 A），看不到工具结果正文。
- 失败的 run 也提炼（WikiTrace：只用成功轨迹会丢掉一半信息）。原因摘要是推测，不是查证过的原因，读取时会标出来。
- 这里只校验「形状」：字段齐不齐、证据编号是不是轨迹里列出来的。内容安不安全交给 L0（lesson_gate），两件事分开。
- 每段轨迹最多 2 条（WikiTrace 是最多 3 条；我们的 run 短，2 条够了，也少花 L1 的钱）。
"""

import json
import logging
from uuid import UUID

from app import db, lesson_store as ls, lesson_trace as lt, llm

log = logging.getLogger(__name__)
MAX_LESSONS, MAX_FIELD = 2, 300
FIELDS = ("precondition", "behavior", "expected", "rationale")

SYSTEM = """你是经验提炼器。给你一段 agent 运行轨迹（用户原话、工具调用的参数和成败，没有工具结果正文），
写出「下次遇到同类情况该怎么做」的经验。每条经验是一个可以被检验的假设：
- precondition：什么情况下适用。要具体到工具和情形（「调用 search_flights 时」），不要写成「所有需要城市的工具」这种过宽的前提。
- behavior：该怎么做，一两句话，可以直接照做。
- expected：照做以后预期看到什么。
- verify：如果把轨迹里第一句用户原话原样再问一遍，用什么条件判断这次做对了。只能用这些检查：
  {{"name":"tool_called","args":{{"name":"工具名","args":{{参数子集}}}}}}  某工具成功调用过，且参数包含这些值
  {{"name":"tool_errors_max","args":{{"name":"工具名","max":0}}}}         某工具最多失败几次
  {{"name":"tool_calls_max","args":{{"name":"工具名","max":1}}}}          某工具最多成功调用几次（防重复提交）
  {{"name":"answer_has","args":{{"all_of":["必须出现的词"]}}}}            回答里必须出现
  {{"name":"answer_lacks","args":{{"patterns":["不许出现的正则"]}}}}       回答里不许出现
- tools：这条经验涉及的工具名（只能是轨迹里出现过的）。
- rationale：一两句话，说明这段轨迹为什么失败、或者是怎么救回来的。失败原因是你的推测，照实写「可能是」。
- evidence：依据的编号，只能用轨迹里列出的：工具调用写方括号里的 id（如 "toolu_01"），用户原话写 "#消息号"（如 "#45"）。
规则：
1. 经验讲的是「怎么用工具、怎么做事」，不是关于这位用户本人的信息（那是用户记忆的事，这里不记）。
2. 写成通用的做法，不要照抄这一次的具体数值；但 verify 要针对第一句用户原话写，可以用具体数值。
3. 轨迹里有「以下是改对之后」时，behavior 只能写那一段里实际成功的做法（照着改对后的参数写），
   不要加轨迹里没有成功验证过的步骤（比如去查别的工具）；tools 只写照做时真正要调用的工具。
4. 轨迹里没有值得记的（比如工具本身挂了、和做法无关），输出空列表。宁缺毋滥，最多 {max_n} 条。
只输出 JSON：{{"lessons":[{{"precondition":"…","behavior":"…","expected":"…","verify":[…],"tools":["…"],"rationale":"…","evidence":["…"]}}]}}"""


def parse(text: str) -> list:
    i, j = text.find("{"), text.rfind("}")
    got = json.loads(text[i:j + 1]).get("lessons") if i >= 0 and j > i else None
    if not isinstance(got, list):
        raise ValueError("提炼结果不是 {lessons: [...]}")
    return got


def shape(raw: list, ep: lt.Episode) -> tuple[list[dict], int]:
    """逐条校验形状，返回 (合格的 [{h, evidence}], 丢掉的个数)。证据编号换成 (消息 id, tool_use_id)。"""
    by_tool = {c["tool_use_id"]: c["msg"] for c in ep.calls if c["msg"] is not None}
    by_msg = {f"#{mid}": mid for mid, _ in ep.said}
    good = []
    for x in raw[:MAX_LESSONS]:
        if not isinstance(x, dict):
            continue
        ok = all(isinstance(x.get(k), str) and 0 < len(x[k].strip()) <= MAX_FIELD for k in FIELDS)
        verify, tools_, ev = x.get("verify"), x.get("tools"), x.get("evidence")
        ok = ok and isinstance(verify, list) and verify and all(
            isinstance(v, dict) and isinstance(v.get("name"), str) and isinstance(v.get("args", {}), dict)
            for v in verify)
        ok = ok and isinstance(tools_, list) and tools_ and all(isinstance(t, str) for t in tools_)
        ok = ok and isinstance(ev, list) and ev and all(e in by_tool or e in by_msg for e in ev)
        if not ok:
            continue
        h = {k: x[k].strip() for k in FIELDS} | {"verify": verify, "tools": tools_, "outcome": ep.outcome}
        evidence = [(by_tool[e], e) if e in by_tool else (by_msg[e], "") for e in ev]
        good.append({"h": h, "evidence": evidence})
    return good, len(raw) - len(good)


async def distill(ep: lt.Episode, *, complete=None) -> tuple[list[dict], int]:
    """调一次模型，返回 (形状合格的候选, 丢掉的个数)。complete 默认是真模型，测试注入假的。"""
    text, stop = await (complete or llm.complete_with_stop)(
        system=SYSTEM.format(max_n=MAX_LESSONS), prompt=lt.render(ep), max_tokens=3000)
    if stop == "max_tokens":
        raise ValueError("提炼输出被截断")
    return shape(parse(text), ep)


async def scan(run_id: UUID, *, complete=None) -> dict | None:
    """扫一个 run：没信号记 skipped；有信号就提炼、写候选。每个 run 只扫一次，扫过的返回 None。"""
    pool = db.pool()
    ep = await lt.episode(run_id)
    if ep is None:
        return None
    if not await pool.fetchval("INSERT INTO lesson_scans (run_id, user_id, status, signals) VALUES ($1, $2, $3, $4) "
                               "ON CONFLICT DO NOTHING RETURNING true", run_id, ep.user_id,
                               "running" if ep.signals else "skipped", ep.signals):
        return None
    if not ep.signals:
        return {"signals": [], "candidates": []}
    try:
        good, dropped = await distill(ep, complete=complete)
        ids = []
        for g in good:
            ids.append(await ls.add_candidate(ep.user_id, run_id, g["h"], g["evidence"]))
            await ls.index(ids[-1], g["h"]["precondition"])
    except Exception as e:  # 失败了记下来，不重试（批量脚本可以手动清掉这行重扫）
        log.warning("lesson distill %s failed: %s", run_id, type(e).__name__)
        await pool.execute("UPDATE lesson_scans SET status = 'failed', result = $2 WHERE run_id = $1",
                           run_id, {"error": type(e).__name__})
        return {"signals": ep.signals, "error": type(e).__name__}
    out = {"signals": ep.signals, "candidates": ids, "dropped": dropped}
    await pool.execute("UPDATE lesson_scans SET status = 'done', result = $2 WHERE run_id = $1", run_id, out)
    return out
