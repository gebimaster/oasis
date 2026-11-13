"""M5 人在回路：挂起时整批落库，人决定后由续跑 run 执行，并把整批结果拼回一条 tool_result 消息。

状态机（tool_approval.status）：
  pending ──批准──> approved ──claim──> executing ──> succeeded / failed
     │                                     └──崩溃 / 超时 / 被停止──> outcome_unknown ──人核实──> (resolution)
     ├──拒绝──> rejected
     └──用户直接发新消息──> superseded
思路：持久化暂停 + 冷启动续跑；执行日志里「有执行意图」不等于「可以重试」（claim / recover）。
"""

import asyncio
import hashlib
import json
import logging
from uuid import UUID

from app import db, rules, tools

log = logging.getLogger(__name__)

# 给模型看的结果文字：统一以【未执行】/【结果未知】开头，让模型分清「没做」「做了但不知道结果」和「工具坏了」
NOT_RUN_REJECTED = "[未执行] 用户拒绝了这个操作。不要换个方式重试，除非用户再次明确要求。"
NOT_RUN_SUPERSEDED = "[未执行] 用户没有确认这个操作，而是发了新消息；不要假设它已执行，按用户的新消息继续。"
NOT_RUN_APPROVED = "[未执行] 用户批准后服务中断，操作没有开始执行；不要假设它已执行。"
# O10 复核③：排在要确认的调用后面、本来放行的调用：用户没确认前面那个就发了新消息，它也没执行
NOT_RUN_DEFERRED = "[未执行] 它排在一个要用户确认的操作后面，用户没有确认就发了新消息，所以它也没有执行。"
UNKNOWN = "[结果未知] 执行过程中服务中断或调用超时，操作可能已经生效也可能没有。"
UNKNOWN_TAIL = {
    "done": "用户核实后确认：操作已经生效。不要重复执行。",
    "not_done": "用户核实后确认：操作没有生效。是否重做由用户决定，不要自行重试。",
    None: "用户还没有核实。不要重试；请提醒用户先核实外部状态。",
}
# O10：系统按幂等键查到的结论，措辞和人核实的分开（模型和用户都该知道是谁确认的）
RECONCILED_TAIL = {
    "done": "系统按这次调用的编号查询确认：操作已经生效。不要重复执行。",
    "not_done": "系统按这次调用的编号查询确认：操作没有生效。是否重做由用户决定，不要自行重试。",
}
# O10：「查不到」要等多久才能信。转账在本进程里是一个数据库事务，进程死了就不会再提交；但真实的外部 API
# 可能还在处理一个已经发出去的请求，晚几秒才落账。所以执行结束（记成结果未知）之后这么久以内，查不到只当「还不知道」
RECONCILE_GRACE = float(__import__("os").getenv("RECONCILE_GRACE_SECONDS", "60"))
# 系统查到结论时走和人核实同一条 CAS 更新（db.decide_approval），只是记成 reconcile
RECONCILE_SQL = {v: (f"resolution = '{v}', resolved_by = 'reconcile'", "status = 'outcome_unknown' AND resolution IS NULL")
                 for v in ("done", "not_done")}


def canonical(args) -> str:
    """参数原文：键排序、无空格。存进库的就是这串文字，sha256 也对它算。"""
    return json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_rows(items: list[dict]) -> list[dict]:
    """agent_loop 给的批次 → 待插入的行：参数在这里一次性定型并算 hash。"""
    return [{**i, "args_json": (a := canonical(i["args"])), "args_sha256": sha256(a)} for i in items]


def result_block(row: dict) -> dict:
    """一行审批记录 → 给模型的 tool_result。"""
    status = row["status"]
    if status in ("succeeded", "failed", "denied"):
        content, is_error = row["result"]["content"], row["result"]["is_error"]
    elif status == "outcome_unknown":
        tail = (RECONCILED_TAIL if row.get("resolved_by") == "reconcile" else UNKNOWN_TAIL)[row["resolution"]]
        content, is_error = f"{UNKNOWN}{tail}", True
    elif status == "superseded" and row.get("decision") == rules.ALLOW:  # 被作废的 deferred
        content, is_error = NOT_RUN_DEFERRED, True
    else:
        text = {"rejected": NOT_RUN_REJECTED, "approved": NOT_RUN_APPROVED}.get(status, NOT_RUN_SUPERSEDED)
        content, is_error = text, True
    return {"type": "tool_result", "tool_use_id": row["tool_use_id"], "content": content, "is_error": is_error}


async def _update(w, run_id: UUID, row: dict, status: str) -> None:
    ev = {"type": "approval_update", "tool_use_id": row["tool_use_id"], "status": status}
    await w.emit(lambda: db.append_event(run_id, ev))


async def execute_approved(row: dict, run_id: UUID, w) -> dict:
    """执行一条已批准的调用，最多一次。返回执行后的最新行。"""
    claimed = await db.claim(row["id"])
    if claimed is None:  # 没抢到：别人在执行 / 已执行 / 状态变了。绝不执行，读回最新状态
        return await db.get_approval(row["id"])
    await _update(w, run_id, row, "executing")
    if sha256(claimed["args_json"]) != claimed["args_sha256"]:  # 记录被改过：不执行
        await db.finish_approval(row["id"], "failed", {"content": "[未执行] 参数校验失败，拒绝执行", "is_error": True})
    else:
        # 只用库里的参数；执行前把存在性 / 权限 / schema / 规则全部重查一遍（等审批期间权限可能被收回、
        # 可能新加了 deny 规则）。规则只认 deny：ask 已经由人批准过了，新规则只能收紧、不能放宽
        call = tools.prepare({"id": row["tool_use_id"], "name": row["tool_name"],
                              "input": json.loads(claimed["args_json"])})
        tools.call_id.set(row["tool_use_id"])  # O10：幂等键。循环里的调用由 tools._run 设，这条路径直接 invoke，要自己设
        if call.error:
            status = "denied" if call.decision == rules.DENY else "failed"
            await db.finish_approval(row["id"], status, {"content": call.error, "is_error": True})
        else:
            try:
                content, is_error, timed_out, _ = await tools.invoke(call.tool, call.args)  # 错误码只给循环熔断用
            except asyncio.CancelledError:  # 用户点停止时正在执行：外部可能已经做了，记结果未知
                await asyncio.shield(db.finish_approval(row["id"], "outcome_unknown", None))
                raise
            # 超时 = 外部可能还在执行（M4 遗留）；其他异常是工具明确报错，记 failed
            status = "outcome_unknown" if timed_out else ("failed" if is_error else "succeeded")
            await db.finish_approval(row["id"], status, {"content": content, "is_error": is_error})
    row = await db.get_approval(row["id"])
    await _update(w, run_id, row, row["status"])
    return row


async def reconcile(row: dict, user_id: UUID) -> str | None:
    """O10：结果未知的调用按幂等键查一次。返回 done / not_done，查不出结论（工具没有查询、刚执行不久查不到）返回 None。
    查到「转了」马上信；「没转」要等 RECONCILE_GRACE 之后才信（外部可能还在处理）。"""
    tool = tools._REGISTRY.get(row["tool_name"])
    if tool is None or tool.reconcile is None or not row.get("keyed"):  # 复核①：不带键执行的（O10 之前）查不到不代表没做
        return None
    try:
        verdict = await tool.reconcile(user_id, json.loads(row["args_json"]), row["tool_use_id"])
    except Exception as e:  # 复核⑧：查询本身出错 = 查不出结论，不能让续跑或接口崩掉
        log.warning("reconcile %s failed: %s", row["tool_use_id"], type(e).__name__)
        return None
    if verdict == "not_done" and (row.get("age_s") is None or row["age_s"] < RECONCILE_GRACE):
        return None  # 不知道是什么时候出的事（老数据没记 finished_at）也按「刚出事」处理：宁可让人核实
    return verdict


async def auto_reconcile(conversation_id: UUID, rows: list[dict]) -> list[dict]:
    """续跑开头对结果未知、还没人核实的行各查一次；查到结论就写进 resolution（resolved_by = reconcile）。"""
    user = await db.conversation_user(conversation_id)
    out = []
    for r in rows:
        if r["status"] == "outcome_unknown" and r["resolution"] is None and (v := await reconcile(r, user["id"])):
            await db.resolve_by_reconcile(r["id"], v)
            r = await db.get_approval(r["id"])
        out.append(r)
    return out


async def resolve_open(conversation_id: UUID) -> None:
    """O10 复核②：用户直接发新消息、批次要被收口之前先查一次，收口时给模型的话才是查到的结论，不是「还没核实」。
    只写结论、不开续跑（新消息本身就要开一个 run）。"""
    rows = await db.open_batch(conversation_id)
    if any(r["status"] == "outcome_unknown" and r["resolution"] is None for r in rows):
        await auto_reconcile(conversation_id, rows)


async def reconcile_pending() -> list[tuple[UUID, UUID]]:
    """O10 复核②：后台对账（runner.lease_loop 每轮调）。接管把执行中的调用记成结果未知后不会续跑，
    续跑开头那次自动查永远轮不到它；这里扫一遍，查到结论就走和人核实同一条 CAS 更新。
    返回 (会话, 续跑 run) 列表：批次因此不再等人时由调用方启动续跑。"""
    started = []
    for row in await db.unresolved_unknown():
        verdict = await reconcile(row, row["user_id"])
        if verdict is None:
            continue
        try:
            r = await db.decide_approval(row["conversation_id"], row["tool_use_id"], *RECONCILE_SQL[verdict])
        except db.RunConflict:  # 会话正忙：下一轮再来
            continue
        if r and r.get("updated") and r.get("run_id"):
            started.append((row["conversation_id"], r["run_id"]))
    return started


async def resume(conversation_id: UUID, run_id: UUID, w) -> bool:
    """续跑 run 开头调用。没有未收口的批次就什么也不做。
    返回 True 表示批次还在等人（执行时超时 → 结果未知），本 run 应该再次挂起，不进模型循环。"""
    rows = await db.open_batch(conversation_id)
    if not rows:
        return False
    # O10 复核③：deferred 也在这里按 position 顺序执行（批准了前面要确认的那个之后，后面的才轮到）
    rows = [await execute_approved(r, run_id, w) if r["status"] in ("approved", "deferred") else r for r in rows]
    rows = await auto_reconcile(conversation_id, rows)  # O10：能按键查清的不用等人
    if any(r["status"] == "pending" or (r["status"] == "outcome_unknown" and r["resolution"] is None)
           for r in rows):
        return True
    await w.emit(lambda: db.close_batch(conversation_id, run_id, [r["id"] for r in rows],
                                        [result_block(r) for r in rows]))
    return False


def blocks_for_new_message(rows: list[dict]) -> list[dict]:
    """用户没点确认、直接发了新消息：pending 已在同一事务里改成 superseded，整批结果放在新消息最前面。"""
    return [result_block(r) for r in rows]
