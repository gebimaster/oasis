"""O1：run 归属。租约 + 心跳、只接管过期的、fencing、旧属主醒来悄悄退出。需要本地 PG。"""

import asyncio
from datetime import timedelta
from uuid import uuid4

import pytest

from app import db, runner, trace
from app.hub import END, hub
from test_db import _conversation, with_fresh_db
from test_events import _db_events, stalling_model

EXPIRED = "UPDATE runs SET lease_until = now() - interval '1 second' WHERE id = $1"


async def _as_other_instance(coro_fn):
    """假装自己是实例 B：换掉进程身份跑一段，再换回来。"""
    mine, db.INSTANCE_ID = db.INSTANCE_ID, uuid4()
    try:
        return await coro_fn()
    finally:
        db.INSTANCE_ID = mine


def test_fencing_rejects_writes_from_old_owner():
    """手工把属主改成别人：原属主再写事件、存消息、收尾，全部被拒，库里一行不多。"""
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        await db.append_event(run_id, {"type": "text_delta", "text": "前半句"})
        await db.pool().execute("UPDATE runs SET owner_id = gen_random_uuid() WHERE id = $1", run_id)
        with pytest.raises(db.LeaseLost):
            await db.append_event(run_id, {"type": "text_delta", "text": "后半句"})
        with pytest.raises(db.LeaseLost):  # 消息和 message_saved 同一事务：事件被拒，消息也回滚
            await db.insert_message(cid, run_id, "assistant", [{"type": "text", "text": "整段"}])
        assert await db.finish_run(run_id, "completed", "end_turn") is None
        assert [m["role"] for m in await db.list_messages(cid)] == ["user"]
        assert [e["seq"] for e in await _db_events(run_id)] == [1]
        assert (await db.get_run(run_id))["status"] == "running"
    with_fresh_db(body)


def test_scan_takes_over_only_expired_runs():
    """别的实例正在跑的（租约没过期）不动；过期的连同它的工具执行、span 一起收尾。"""
    async def body():
        uid, live_c = await _conversation()
        dead_c = await db.create_conversation(uid)
        live = await db.start_run(live_c, [{"type": "text", "text": "活的"}])
        dead = await db.start_run(dead_c, [{"type": "text", "text": "死的"}])
        await db.pool().execute("UPDATE runs SET owner_id = gen_random_uuid() WHERE id = $1", live)  # 别人的、还在续租
        for r in (live, dead):
            await trace.start("run", "run", r)
            await db.start_execution(r, f"t-{r}", "transfer", {"amount": 1})
        await db.pool().execute(EXPIRED, dead)

        assert await db.recover_expired_runs() == 1
        d, lv = await db.get_run(dead), await db.get_run(live)
        assert (d["status"], d["stop_reason"], d["owner_id"]) == ("failed", "interrupted", db.INSTANCE_ID)
        assert lv["status"] == "running" and lv["owner_id"] != db.INSTANCE_ID
        span = dict(await db.pool().fetch("SELECT run_id, status FROM spans"))
        assert span == {dead: "interrupted", live: "running"}
        exe = dict(await db.pool().fetch("SELECT run_id, status FROM tool_executions"))
        assert exe == {dead: "outcome_unknown", live: "executing"}
        assert await db.recover_expired_runs() == 0  # 再扫一次不重复收尾
    with_fresh_db(body)


def test_renew_only_runs_with_live_tasks_and_their_children():
    """续租按「本进程还有 task 的 run」：僵尸 run（task 没了）和别人的 run 都不续。"""
    async def body():
        uid, cid = await _conversation()
        parent = await db.start_run(cid, [{"type": "text", "text": "父"}])
        _, child = await db.start_child_run(cid, parent, "sp1", "甲", "子任务")
        zombie = await db.start_run(await db.create_conversation(uid), [{"type": "text", "text": "僵尸"}])
        other = await db.start_run(await db.create_conversation(uid), [{"type": "text", "text": "别人的"}])
        await db.pool().execute("UPDATE runs SET owner_id = gen_random_uuid() WHERE id = $1", other)
        await db.pool().execute("UPDATE runs SET lease_until = now() + interval '1 second'")

        assert await db.renew_leases([parent, other]) == 2  # other 点名了也不续：不是我的
        left = dict(await db.pool().fetch("SELECT id, lease_until - now() > interval '20 seconds' FROM runs"))
        assert left == {parent: True, child: True, zombie: False, other: False}
    with_fresh_db(body)


def test_old_owner_wakes_up_after_takeover_and_exits_quietly(caplog):
    """A 卡住期间被 B 接管（B 写了 run_end）。A 醒来第一次写就被拒：不再写任何事件，task 正常结束，订阅者收到 END。
    O2：A 上的订阅者也能收到 B 写的 run_end（hub 从库里拉，不管是谁写的）。
    被接管不是崩溃：不能走 internal_error 那条路、打一条带异常栈的 crashed（运维会当成 bug 告警）。"""
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        q = hub.subscribe(run_id)
        runner.start(cid, run_id, model_stream=stalling_model(0.6))
        task = runner._tasks[run_id]
        await asyncio.sleep(0.3)  # 「我查」已经写库，模型正卡着
        await db.pool().execute(EXPIRED, run_id)
        assert await _as_other_instance(db.recover_expired_runs) == 1

        await task
        assert task.exception() is None
        events = await _db_events(run_id)
        assert events[-1]["type"] == "run_end" and events[-1]["stop_reason"] == "interrupted"  # 接管者写的就是最后一条
        assert not any(m["role"] == "assistant" and m["status"] == "completed" for m in await db.list_messages(cid))
        seen = []
        while (ev := await q.get()) is not END:
            seen.append(ev["type"])
        assert seen[-1] == "run_end"  # O2：B 写的 run_end，A 上的订阅者照样收到（M2–O1 时只能靠重连续播）
        assert not [r for r in caplog.records if "crashed" in r.getMessage()]
        assert any("lost its lease" in r.getMessage() for r in caplog.records)
        hub.unsubscribe(run_id, q)
    with_fresh_db(body)


def test_second_instance_start_takes_no_table_lock_when_schema_unchanged():
    """A 正在写（持有 runs 的行锁事务）时 B 启动：schema 没变就不跑 DDL，1 秒锁超时内完成。
    整份重跑会排队等 ACCESS EXCLUSIVE 锁，和 A「先 messages 后 runs」的事务实测死锁。"""
    async def body():
        schema = (db.Path(db.__file__).parent / "schema.sql").read_text(encoding="utf-8")
        a, b = await db.pool().acquire(), await db.pool().acquire()
        try:
            async with a.transaction():
                await a.execute("UPDATE runs SET last_seq = last_seq WHERE false")  # A 的事务拿着 runs 的表锁（RowExclusive）
                await b.execute("SET lock_timeout = '1s'")
                await db._migrate(b, schema)
        finally:
            await b.execute("RESET lock_timeout")
            await db.pool().release(a)
            await db.pool().release(b)
    with_fresh_db(body)


def test_lease_loop_keeps_live_run_and_takes_over_dead_one(monkeypatch):
    """租约 0.6 秒、每 0.2 秒一轮：跑 1.5 秒的 run 一直被续租，正常完成；没有 task 的 run 不用重启就被接管。"""
    monkeypatch.setattr(db, "LEASE", timedelta(seconds=0.6))
    monkeypatch.setattr(runner, "LEASE_TICK", 0.2)

    async def body():
        uid, cid = await _conversation()
        dead = await db.start_run(await db.create_conversation(uid), [{"type": "text", "text": "进程崩了"}])
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        loop = asyncio.create_task(runner.lease_loop())
        try:
            runner.start(cid, run_id, model_stream=stalling_model(1.5))
            await runner._tasks[run_id]
        finally:
            loop.cancel()
        live = await db.get_run(run_id)
        assert (live["status"], live["stop_reason"]) == ("completed", "end_turn")
        assert ((await db.get_run(dead))["status"], (await db.get_run(dead))["stop_reason"]) == ("failed", "interrupted")
    with_fresh_db(body)
