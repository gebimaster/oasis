"""O10 第二步：transfer 的幂等键（= tool_use_id）+ 按键对账。需要 docker compose 里的 PG。"""

import asyncio

import pytest
from fastapi import HTTPException

from app import approvals, db, main, runner, tools
from test_approvals import APPROVE, DONE, _ledger, _run, _status, _suspended
from test_db import with_fresh_db


async def _pay(uid, key, to="alice", amount=500):
    tools.current.set(tools.ToolContext(user_id=uid, permissions=frozenset({"transfer:write"})))
    tools.call_id.set(key)
    return await tools.invoke(tools._REGISTRY["transfer"], {"to": to, "amount": amount})


def test_same_key_twice_charges_once():
    async def body():
        uid = await db.create_user("u", "t-u")
        first, err1, *_ = await _pay(uid, "p1")
        again, err2, *_ = await _pay(uid, "p1")
        assert not err1 and not err2 and await _ledger() == 1
        assert "没有重复扣款" in again and first.split("流水号 ")[1][:1] == again.split("流水号 ")[1][:1]
        conflict, err3, *_ = await _pay(uid, "p1", amount=900)  # 同键不同参数：报错、不执行
        assert err3 and "IDEMPOTENCY_CONFLICT" in conflict and await _ledger() == 1
        other = await db.create_user("v", "t-v")
        await _pay(other, "p1")  # 键按用户隔离：别人的同名键是另一笔
        assert await _ledger() == 2
    with_fresh_db(body)


async def _crashed_in_execution(after_insert: bool):
    """批准 → 执行到一半崩了：after_insert=True 时转账已经提交（只是没来得及写执行结果），否则还没开始插入。
    租约过期后对账把它记成结果未知（M5 / O1 的老路径）。"""
    cid, _ = await _suspended()
    uid = (await db.conversation_user(cid))["id"]
    await db.decide_approval(cid, "p1", *APPROVE)
    await db.claim(await db.pool().fetchval("SELECT id FROM tool_approval WHERE tool_use_id = 'p1'"))  # 走真的抢执行权（带 keyed）
    if after_insert:
        await _pay(uid, "p1")
    await db.pool().execute("UPDATE runs SET lease_until = now() - interval '1 second'")
    assert await db.recover_expired_runs() == 1 and await _status("p1") == "outcome_unknown"
    return cid, uid


async def _reconcile_via_api(cid):
    r = await main.decide("p1", main.DecisionBody(action="reconcile"), conversation_id=cid)
    if r.get("run_id"):
        await runner._tasks[r["run_id"]]
    return r


def test_crash_after_insert_reconciles_to_done_and_never_reruns(monkeypatch):
    async def body():
        cid, _ = await _crashed_in_execution(after_insert=True)
        stream_calls = []

        async def stream(**kw):
            stream_calls.append(kw)
            yield {"type": "final", "content": DONE[0], "stop_reason": "end_turn"}
        monkeypatch.setattr(runner, "run_loop", _wrap_stream(stream))
        r = await _reconcile_via_api(cid)
        assert r["resolution"] == "done"
        row = await db.pool().fetchrow("SELECT resolution, resolved_by FROM tool_approval WHERE tool_use_id = 'p1'")
        assert dict(row) == {"resolution": "done", "resolved_by": "reconcile"}
        assert await _ledger() == 1  # 只有崩溃前那一笔，没有重做
        tail = stream_calls[0]["messages"][-1]["content"]
        assert any(approvals.RECONCILED_TAIL["done"] in str(b.get("content")) for b in tail)
    with_fresh_db(body)


def test_crash_before_insert_stays_unknown_within_grace_then_not_done(monkeypatch):
    async def body():
        cid, _ = await _crashed_in_execution(after_insert=False)
        with pytest.raises(HTTPException) as e:  # 刚出事：查不到不等于没转（外部可能还在处理）
            await _reconcile_via_api(cid)
        assert e.value.status_code == 409 and await _status("p1") == "outcome_unknown"
        monkeypatch.setattr(approvals, "RECONCILE_GRACE", 0)  # 等够了
        monkeypatch.setattr(runner, "run_loop", _wrap_stream(None))
        r = await _reconcile_via_api(cid)
        assert r["resolution"] == "not_done" and await _ledger() == 0
    with_fresh_db(body)


def test_resume_reconciles_automatically(monkeypatch):
    """续跑开头对结果未知的行自动查一次：能按键查清的，不用等人点。"""
    async def body():
        cid, _ = await _crashed_in_execution(after_insert=True)
        rows = await approvals.auto_reconcile(cid, await db.open_batch(cid))
        assert [r["resolution"] for r in rows if r["tool_use_id"] == "p1"] == ["done"]
    with_fresh_db(body)


def test_tools_without_lookup_still_need_a_human():
    async def body():
        cid, uid = await _crashed_in_execution(after_insert=False)
        row = next(r for r in await db.open_batch(cid) if r["tool_use_id"] == "p1")
        assert await approvals.reconcile({**row, "tool_name": "get_weather"}, uid) is None
    with_fresh_db(body)


def _wrap_stream(stream):
    """把假模型塞进 runner 用的 run_loop（测试里 runner._run 没法从 decide 传 model_stream 进去）。"""
    from app import agent_loop

    async def fake_stream(**kw):
        yield {"type": "final", "content": DONE[0], "stop_reason": "end_turn"}

    def run_loop(*a, **kw):
        kw["model_stream"] = stream or fake_stream
        return agent_loop.run_loop(*a, **kw)
    return run_loop


class _Writer:
    """approvals.resume 只用到 w.emit：直接执行写库函数。"""
    async def emit(self, write):
        await write()


def test_background_reconcile_after_takeover_resumes_the_batch(monkeypatch):
    """复核②：接管把它记成结果未知后不会续跑，续跑开头的那次自动查永远轮不到。后台对账（lease_loop 每轮）扫到它、
    按键查清，批次不再等人就开续跑，续跑把查到的结论交给模型。原来那个测试在生产到不了的状态上调 resume，证明不了什么。"""
    async def body():
        cid, _ = await _crashed_in_execution(after_insert=True)
        monkeypatch.setattr(runner, "run_loop", _wrap_stream(None))
        started = await approvals.reconcile_pending()
        assert [c for c, _ in started] == [cid]
        runner.start(*started[0])
        await runner._tasks[started[0][1]]
        msgs = await db.list_messages(cid)
        assert any(approvals.RECONCILED_TAIL["done"] in str(b.get("content")) for m in msgs
                   if isinstance(m["content"], list) for b in m["content"])
        assert await _ledger() == 1
    with_fresh_db(body)


def test_rows_executed_without_a_key_are_never_concluded():
    """复核①：O10 之前（或旧实例）执行的转账账本里没有键，按键查不到不代表没转：不下结论，交给人。"""
    async def body():
        cid, uid = await _crashed_in_execution(after_insert=False)
        await db.pool().execute("UPDATE tool_approval SET keyed = false, finished_at = now() - interval '10 minutes'")
        await db.pool().execute("INSERT INTO ledger (user_id, to_name, amount) VALUES ($1, 'alice', 500)", uid)  # 旧代码那一笔
        row = next(r for r in await db.open_batch(cid) if r["tool_use_id"] == "p1")
        assert await approvals.reconcile(row, uid) is None
        assert await approvals.reconcile_pending() == []
    with_fresh_db(body)


def test_stuck_owner_cannot_commit_after_takeover():
    """复核④：卡住（没死）的旧进程在被接管之后醒来，转账不能再落账：围栏和插入在同一条语句里。"""
    async def body():
        cid, uid = await _crashed_in_execution(after_insert=False)  # 已被接管、记成结果未知
        content, is_error, *_ = await _pay(uid, "p1")
        assert is_error and "FENCED" in content and await _ledger() == 0
    with_fresh_db(body)


def test_new_message_closes_the_batch_with_the_looked_up_result(monkeypatch):
    """复核②：用户不点按钮、直接发新消息：收口前先按键查一次，交给模型的是查到的结论，不是「还没核实」。"""
    async def body():
        cid, _ = await _crashed_in_execution(after_insert=True)
        monkeypatch.setattr(runner, "run_loop", _wrap_stream(None))
        r = await main.send_message(main.SendBody(text="钱转了吗"), conversation_id=cid)
        await runner._tasks[r["run_id"]]
        msgs = await db.list_messages(cid)
        closing = next(m for m in msgs if m["role"] == "user" and isinstance(m["content"], list)
                       and any(b.get("tool_use_id") == "p1" for b in m["content"]))
        assert approvals.RECONCILED_TAIL["done"] in str(closing["content"])
    with_fresh_db(body)


def test_approved_transfer_carries_its_tool_use_id_as_key():
    """审批续跑路径直接调 tools.invoke（不经 tools._run），幂等键要它自己设；否则这一笔没有键，结果未知时查不到。"""
    async def body():
        cid, _ = await _suspended()
        r = await db.decide_approval(cid, "p1", *APPROVE)
        await _run(cid, r["run_id"], [DONE])
        assert await db.pool().fetchval("SELECT idem_key FROM ledger") == "p1"
    with_fresh_db(body)



def test_calls_after_a_confirmation_wait_for_it(monkeypatch):
    """复核③：模型给的是 [转账（要确认）, 查天气（放行）]：查天气不能抢在转账被批准之前执行，
    批准后续跑按 position 顺序：先转账、再查天气。"""
    from app import tools as tools_mod
    order = []
    orig = tools_mod.invoke

    async def spy(tool, args):
        order.append(tool.name)
        return await orig(tool, args)
    monkeypatch.setattr(tools_mod, "invoke", spy)

    async def body():
        from test_approvals import PAY, WEATHER
        cid, _ = await _suspended(calls=(PAY, WEATHER))
        assert order == []  # 挂起时一个都没执行
        assert await _status("w1") == "deferred"
        r = await db.decide_approval(cid, "p1", *APPROVE)
        await _run(cid, r["run_id"], [DONE])
        assert order == ["transfer", "get_weather"] and await _status("w1") == "succeeded"
    with_fresh_db(body)



def test_timeout_after_insert_during_resume_is_settled_right_away(monkeypatch):
    """续跑里执行批准过的转账：账已经落了、工具却超时（记成结果未知）。续跑开头那次自动查当场查到「转了」，
    不用再挂起等人（查到「转了」不需要等 60 秒）。"""
    import dataclasses
    t = tools._REGISTRY["transfer"]

    async def slow_after_insert(args):
        out = await t.handler(args)
        await asyncio.sleep(5)  # 落账之后卡住，被工具总超时打断
        return out
    monkeypatch.setitem(tools._REGISTRY, "transfer", dataclasses.replace(t, handler=slow_after_insert, timeout=0.3))

    async def body():
        cid, _ = await _suspended()
        r = await db.decide_approval(cid, "p1", *APPROVE)
        await _run(cid, r["run_id"], [DONE])
        row = await db.pool().fetchrow("SELECT status, resolution, resolved_by FROM tool_approval WHERE tool_use_id = 'p1'")
        assert dict(row) == {"status": "outcome_unknown", "resolution": "done", "resolved_by": "reconcile"}
        assert await _ledger() == 1
    with_fresh_db(body)


def test_lookup_errors_mean_no_conclusion(monkeypatch):
    """复核⑧：按键查询本身出错（库断了），当成查不出结论，不能让续跑或接口崩掉。"""
    import dataclasses

    async def broken(*a):
        raise ConnectionError("db down")
    monkeypatch.setitem(tools._REGISTRY, "transfer", dataclasses.replace(tools._REGISTRY["transfer"], reconcile=broken))

    async def body():
        cid, uid = await _crashed_in_execution(after_insert=True)
        row = next(r for r in await db.open_batch(cid) if r["tool_use_id"] == "p1")
        assert await approvals.reconcile(row, uid) is None
    with_fresh_db(body)
