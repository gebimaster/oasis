"""M5：挂起 → 人决定 → 续跑执行一次；并发、拒绝、作废、结果未知、篡改。需要 docker compose 里的 PG。"""

import asyncio

from app import approvals, db, runner
from test_agent_loop import fake_model
from test_db import with_fresh_db

PERMS = ["weather:read", "transfer:write"]
WEATHER = {"type": "tool_use", "id": "w1", "name": "get_weather", "input": {"city": "北京"}}
PAY = {"type": "tool_use", "id": "p1", "name": "transfer", "input": {"to": "alice", "amount": 500}}
DONE = ([{"type": "text", "text": "好的"}], "end_turn")
APPROVE, REJECT = ("status = 'approved'", "status = 'pending'"), ("status = 'rejected'", "status = 'pending'")


async def _run(cid, run_id, script):
    """跑一个 run 到结束，返回假模型收到的每轮参数。"""
    stream, calls = fake_model(script)
    runner.start(cid, run_id, model_stream=stream)
    await runner._tasks[run_id]
    return calls


async def _suspended(calls=(WEATHER, PAY)):
    """建用户 + 会话，模型一次调读工具和转账 500（high、没规则命中 → ask），跑到挂起。"""
    uid = await db.create_user("u", "t-u")
    await db.grant_permissions(uid, PERMS)
    cid = await db.create_conversation(uid)
    run_id = await db.start_run(cid, [{"type": "text", "text": "查天气并给 alice 转 500"}])
    await _run(cid, run_id, [(list(calls), "tool_use")])
    return cid, run_id


async def _ledger():
    return await db.pool().fetchval("SELECT count(*) FROM ledger")


async def _status(tool_use_id):
    return await db.pool().fetchval("SELECT status FROM tool_approval WHERE tool_use_id = $1", tool_use_id)


def test_suspend_runs_reads_now_and_ends_run():
    async def body():
        cid, run_id = await _suspended()
        run = await db.get_run(run_id)
        assert (run["status"], run["stop_reason"]) == ("completed", "awaiting_approval")  # 不占会话锁
        assert await _status("w1") == "succeeded" and await _status("p1") == "pending"  # 读工具不陪着等
        types = [e["type"] for e in await db.list_events(run_id, 0)]
        assert types[-2:] == ["approval_request", "run_end"]
        assert await _ledger() == 0
    with_fresh_db(body)


def test_two_concurrent_approvals_execute_once():
    """两个标签页同时批准：真并发打到 PG，只有一个改成功、只开一个续跑，账本只多一行。"""
    async def body():
        cid, _ = await _suspended()
        r1, r2 = await asyncio.gather(db.decide_approval(cid, "p1", *APPROVE), db.decide_approval(cid, "p1", *APPROVE))
        assert sorted([r1["updated"], r2["updated"]]) == [False, True]
        run_ids = [r["run_id"] for r in (r1, r2) if r["run_id"]]
        assert len(run_ids) == 1
        calls = await _run(cid, run_ids[0], [DONE])
        assert await _ledger() == 1 and await _status("p1") == "succeeded"
        results = calls[0]["messages"][-1]["content"]          # 续跑时模型收到的整批结果，顺序与 tool_use 一致
        assert [b["tool_use_id"] for b in results if b["type"] == "tool_result"] == ["w1", "p1"]
        assert "已向 alice 转账" in results[1]["content"]
    with_fresh_db(body)


def test_claim_is_atomic_under_real_interleaving():
    """确定性地交错：A 抢到但还没提交时 B 也来抢，B 被行锁挡住，A 提交后 B 重新判断 WHERE → 0 行。"""
    async def body():
        cid, _ = await _suspended()
        await db.decide_approval(cid, "p1", *APPROVE)
        aid = await db.pool().fetchval("SELECT id FROM tool_approval WHERE tool_use_id = 'p1'")
        a = await db.pool().acquire()
        tx = a.transaction()
        await tx.start()
        assert await a.fetchrow("UPDATE tool_approval SET status = 'executing' WHERE id = $1 AND status = 'approved' "
                                "RETURNING id", aid)
        b = asyncio.create_task(db.claim(aid))
        await asyncio.sleep(0.3)
        assert not b.done()                                     # B 在等 A 的行锁
        await tx.commit()
        await db.pool().release(a)
        assert await b is None                                  # B 没抢到，不会执行
        many = await asyncio.gather(*(db.claim(aid) for _ in range(10)))
        assert many == [None] * 10
    with_fresh_db(body)


def test_reject_tells_model_and_run_finishes():
    async def body():
        cid, _ = await _suspended()
        r = await db.decide_approval(cid, "p1", *REJECT)
        calls = await _run(cid, r["run_id"], [DONE])
        result = calls[0]["messages"][-1]["content"][1]
        assert result["is_error"] and result["content"] == approvals.NOT_RUN_REJECTED
        assert (await db.get_run(r["run_id"]))["stop_reason"] == "end_turn" and await _ledger() == 0
        again = await db.decide_approval(cid, "p1", *APPROVE)   # 拒绝之后再点批准：落空
        assert again["updated"] is False and again["status"] == "rejected"
    with_fresh_db(body)


def test_new_message_supersedes_pending():
    """用户没点确认，直接说「算了」：pending 作废，整批 tool_result 单独存一条、紧接着存用户消息；之后再批准落空。
    O4：两条分开存，压缩才认得出「算了别转了」是当前提问。"""
    async def body():
        cid, _ = await _suspended()
        await db.start_run(cid, [{"type": "text", "text": "算了别转了"}],
                           close_batch_blocks=approvals.blocks_for_new_message)
        results, question = (await db.list_messages(cid))[-2:]
        assert [b["type"] for b in results["content"]] == ["tool_result", "tool_result"]
        assert results["content"][1]["content"] == approvals.NOT_RUN_SUPERSEDED
        assert question["content"] == [{"type": "text", "text": "算了别转了"}]
        assert await db.pool().fetchval("SELECT array_agg(DISTINCT result_message_id) FROM tool_approval "
                                        "WHERE conversation_id = $1", cid) == [results["id"]]
        late = await db.decide_approval(cid, "p1", *APPROVE)
        assert late["updated"] is False and late["status"] == "superseded" and await _ledger() == 0
    with_fresh_db(body)


def test_crash_while_executing_becomes_unknown_and_never_reruns():
    async def body():
        cid, _ = await _suspended()
        await db.decide_approval(cid, "p1", *APPROVE)
        await db.pool().execute("UPDATE tool_approval SET status = 'executing' WHERE tool_use_id = 'p1'")  # 崩在执行中
        await db.pool().execute("UPDATE runs SET lease_until = now() - interval '1 second'")  # O1：模拟属主崩溃、租约过期
        assert await db.recover_expired_runs() == 1
        assert await _status("p1") == "outcome_unknown"
        assert await db.ready_batches() == []                   # 等人核实：不自动续跑
        r = await db.decide_approval(cid, "p1", "resolution = 'done'",
                                     "status = 'outcome_unknown' AND resolution IS NULL")
        calls = await _run(cid, r["run_id"], [DONE])
        assert approvals.UNKNOWN_TAIL["done"] in calls[0]["messages"][-1]["content"][1]["content"]
        assert await _ledger() == 0                             # 没有重新执行
    with_fresh_db(body)


def test_approved_but_not_started_resumes_after_restart():
    """approved 还没 claim 就崩了：一定没开始执行，重启后自动续跑，执行一次。"""
    async def body():
        cid, _ = await _suspended()
        r = await db.decide_approval(cid, "p1", *APPROVE)
        await db.pool().execute("UPDATE runs SET lease_until = now() - interval '1 second'")  # O1：续跑没来得及跑，属主就崩了
        await db.recover_expired_runs()
        [(c, run_id)] = await db.ready_batches()
        await _run(c, run_id, [DONE])
        assert await _ledger() == 1 and await _status("p1") == "succeeded"
    with_fresh_db(body)


def test_tampered_args_are_not_executed():
    async def body():
        cid, _ = await _suspended()
        await db.pool().execute("UPDATE tool_approval SET args_json = '{\"amount\":99999,\"to\":\"mallory\"}' "
                                "WHERE tool_use_id = 'p1'")
        r = await db.decide_approval(cid, "p1", *APPROVE)
        await _run(cid, r["run_id"], [DONE])
        assert await _status("p1") == "failed" and await _ledger() == 0
    with_fresh_db(body)


def test_user_stop_leaves_stopped_marker():
    """M1 遗留：点停止后，半截回答存成 stopped 消息；下一轮拼给模型时带「已停止」说明。"""
    from app import context
    from test_events import _run_and_collect, stalling_model

    async def body():
        uid = await db.create_user("u", "t-u")
        cid = await db.create_conversation(uid)
        run_id = await db.start_run(cid, [{"type": "text", "text": "写一篇长文"}])
        await _run_and_collect(cid, run_id, stalling_model(5), cancel_after=0.3)
        last = (await db.list_messages(cid))[-1]
        assert last["status"] == "stopped" and last["content"][0]["text"] == "我查"
        _, msgs = context.build_request(user_name="u", summary=None, history=await db.list_messages(cid),
                                        now=__import__("datetime").datetime.now().astimezone(), final_turn=False)
        assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"][-1]["text"] == context.STOPPED_NOTE
    with_fresh_db(body)


def test_deny_rule_added_while_waiting_blocks_execution():
    """等审批期间管理员加了「给 alice 转账一律拒绝」：批准后续跑重查规则，不执行，模型收到规则拒绝。"""
    async def body():
        cid, _ = await _suspended()
        await db.add_rule(None, "transfer", [{"field": "to", "op": "eq", "value": "alice"}], "deny")
        r = await db.decide_approval(cid, "p1", *APPROVE)
        calls = await _run(cid, r["run_id"], [DONE])
        assert await _status("p1") == "denied" and await _ledger() == 0
        assert "DENIED_BY_RULE" in calls[0]["messages"][-1]["content"][1]["content"]
    with_fresh_db(body)
