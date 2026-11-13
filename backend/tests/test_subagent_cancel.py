"""M7 第 5 步：先写库再取消、写工具结果未知、停止/重启后给父补真 tool_result。"""

import asyncio
import contextlib

from app import db, llm, runner, subagent, subagent_tool, transfer  # noqa: F401
from test_db import _conversation, with_fresh_db
from test_subagent_runs import fake, patched  # noqa: F401  patched 是 fixture


def test_db_is_written_before_cancel_signal(patched):
    patched.setattr(subagent, "SUBAGENT_TIMEOUT", 30)

    async def body():
        both, seen = asyncio.Event(), []
        stream, _ = fake([{"title": "乙1", "task": "slow-1"}, {"title": "乙2", "task": "slow-2"}], both, 2)

        async def spy(**kw):  # 子任务收到取消信号的那一刻，去库里看自己的状态
            try:
                async for ev in stream(**kw):
                    yield ev
            except asyncio.CancelledError:
                seen.append(await db.pool().fetchval(
                    "SELECT status FROM runs WHERE parent_run_id IS NOT NULL AND status <> 'completed' LIMIT 1"))
                raise
        patched.setattr(llm, "stream_message", spy)
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "分头查"}])
        runner.start(cid, run_id)
        await both.wait()
        await runner.cancel(run_id)
        with contextlib.suppress(asyncio.CancelledError):
            await runner._tasks[run_id]
        assert seen == ["cancelling", "cancelling"]  # 信号到达时，停止意图已经在库里

        # 父历史里那条 spawn_subagents 被补上了真 tool_result，下一轮模型不会只看到「中断」
        msgs = await db.list_messages(cid)
        tr = next(b for m in msgs if m["role"] == "user" for b in m["content"] if b.get("tool_use_id") == "sp1")
        assert tr["content"].count("已取消（用户停止）") == 2
    with_fresh_db(body)


def test_write_tool_in_timed_out_child_is_outcome_unknown(patched, monkeypatch):
    monkeypatch.setenv("TRANSFER_DELAY_SECONDS", "30")  # 转账卡住：子任务一定先超时（0.5s）

    async def body():
        uid, cid = await _conversation()
        await db.grant_permissions(uid, ["transfer:write"])
        await db.add_rule(uid, "transfer", [{"field": "amount", "op": "lte", "value": 500}], "allow")  # 不走审批

        async def stream(**kw):
            if kw["system"][-1]["text"] == subagent.CONTRACT:
                content = [{"type": "tool_use", "id": "tr1", "name": "transfer", "input": {"to": "张三", "amount": 300}}]
                stop = "tool_use"
            elif len(kw["messages"]) <= 2:
                content = [{"type": "tool_use", "id": "sp1", "name": "spawn_subagents",
                            "input": {"tasks": [{"title": "付款", "task": "给张三转 300"}]}}]
                stop = "tool_use"
            else:
                content, stop = [{"type": "text", "text": "收到"}], "end_turn"
            yield {"type": "final", "content": content, "stop_reason": stop}
        patched.setattr(llm, "stream_message", stream)
        run_id = await db.start_run(cid, [{"type": "text", "text": "付款"}])
        runner.start(cid, run_id)
        await runner._tasks[run_id]

        ex = await db.pool().fetchrow("SELECT tool_name, status FROM tool_executions")
        assert dict(ex) == {"tool_name": "transfer", "status": "outcome_unknown"}
        result = (await db.list_messages(cid))[2]["content"][0]["content"]
        assert "执行中超时" in result and "结果未知" in result and '"amount": 300' in result
    with_fresh_db(body)


def test_restart_reconciles_children_and_closes_parent():
    async def body():
        _, cid = await _conversation()
        parent = await db.start_run(cid, [{"type": "text", "text": "分头查"}])
        tasks = [{"title": t, "task": t} for t in ("甲", "乙", "丙", "丁")]
        await db.insert_message(cid, parent, "assistant", [{"type": "tool_use", "id": "sp1", "name": "spawn_subagents",
                                                            "input": {"tasks": tasks}}])
        kids = [await db.start_child_run(cid, parent, "sp1", t["title"], t["task"], i) for i, t in enumerate(tasks[:3])]
        await db.insert_message(kids[0][0], kids[0][1], "assistant", [{"type": "text", "text": "甲的结论"}])
        await db.finish_run(kids[0][1], "completed", "end_turn")
        await db.start_execution(kids[1][1], "tr1", "transfer", {"to": "张三", "amount": 300})  # 崩溃时正在转账
        # 用户点了停止、写完库就崩了：父和丙是 cancelling；乙还是 running；丁还在排队（没有子 run）
        await db.pool().execute("UPDATE runs SET status = 'cancelling' WHERE id = ANY($1::uuid[])", [parent, kids[2][1]])

        await db.pool().execute("UPDATE runs SET lease_until = now() - interval '1 second'")  # O1：模拟属主崩溃、租约过期
        assert await db.recover_expired_runs(close_blocks=subagent_tool.closing_blocks) == 3
        status = {r["title"]: (r["status"], r["stop_reason"]) for r in await db.child_runs(parent)}
        assert status == {"甲": ("completed", "end_turn"), "乙": ("failed", "interrupted"),
                          "丙": ("cancelled", "parent_cancel")}
        assert ((await db.get_run(parent))["status"], (await db.get_run(parent))["stop_reason"]) == ("cancelled", "user_cancel")
        assert await db.pool().fetchval("SELECT status FROM tool_executions") == "outcome_unknown"

        tr = (await db.list_messages(cid))[-1]["content"][0]
        assert tr["tool_use_id"] == "sp1"
        for want in ("甲 — 完成", "甲的结论", "乙 — 已中断（服务重启）", "丙 — 已取消", "丁 — 没有开始", "结果未知"):
            assert want in tr["content"]
        types = [e["type"] for e in await db.list_events(parent, 0)]
        assert types[-2:] == ["message_saved", "run_end"]  # 补的结果在 run_end 之前，续播的前端能看到
    with_fresh_db(body)
