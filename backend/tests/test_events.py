"""M2 事件日志验证：seq 连续、先存后推、定时 flush、终态事件。需要先 `docker compose up -d`。"""

import asyncio
import time

import pytest

from app import db, main, runner
from app import hub as hub_mod
from app.hub import END, RESET, hub
from test_db import TEST_URL, _conversation, with_fresh_db


async def _db_events(run_id):
    rows = await db.pool().fetch("SELECT seq, data FROM run_events WHERE run_id = $1 ORDER BY seq", run_id)
    return [{**r["data"], "seq": r["seq"]} for r in rows]


def stalling_model(stall: float):
    """先吐「我查」，停顿 stall 秒，再吐「一下」后结束：模拟模型中途思考。"""
    async def stream(**_):
        yield {"type": "text_delta", "text": "我"}
        yield {"type": "text_delta", "text": "查"}
        await asyncio.sleep(stall)
        yield {"type": "text_delta", "text": "一下"}
        yield {"type": "final", "content": [{"type": "text", "text": "我查一下"}], "stop_reason": "end_turn"}
    return stream


async def _run_and_collect(cid, run_id, stream, cancel_after: float | None = None):
    q = hub.subscribe(run_id)
    t0 = time.monotonic()
    runner.start(cid, run_id, model_stream=stream)
    if cancel_after is not None:
        await asyncio.sleep(cancel_after)
        await runner.cancel(run_id)
    got = []
    while (ev := await q.get()) is not END:
        got.append((time.monotonic() - t0, ev))
    hub.unsubscribe(run_id, q)
    await asyncio.sleep(0)  # 让 task 的 done 回调跑完
    return got


def test_published_equals_db_and_ticker_flushes_during_stall():
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "北京天气？"}])
        got = await _run_and_collect(cid, run_id, stalling_model(0.5))
        events = [ev for _, ev in got]
        assert [e["type"] for e in events] == ["text_delta", "text_delta", "message_saved", "run_end"]
        assert [e["text"] for e in events[:2]] == ["我查", "一下"]   # 停顿前的字被合成一批
        assert got[0][0] < 0.4                                       # 定时器在停顿期间就推了，不等 0.5s
        assert [e["seq"] for e in events] == [1, 2, 3, 4]            # 连续、无洞
        assert events == await _db_events(run_id)                    # 推出去的 == 库里的
    with_fresh_db(body)


def test_cancel_still_flushes_and_ends():
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        got = await _run_and_collect(cid, run_id, stalling_model(5), cancel_after=0.25)
        events = [ev for _, ev in got]
        # M5：用户停止会多存一条 stopped 消息（半截文字），在 run_end 之前
        assert [e["type"] for e in events] == ["text_delta", "message_saved", "run_end"]
        assert events[-1]["status"] == "cancelled"
        assert events == await _db_events(run_id)
    with_fresh_db(body)


def test_recover_writes_run_end_with_next_seq():
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        await db.append_event(run_id, {"type": "text_delta", "text": "半句"})
        await db.append_event(run_id, {"type": "tool_start", "name": "get_weather"})
        await db.pool().execute("UPDATE runs SET lease_until = now() - interval '1 second'")  # O1：模拟属主崩溃、租约过期
        assert await db.recover_expired_runs() == 1       # 模拟崩溃后重启
        assert await db.finish_run(run_id, "completed", "end_turn") is None  # 不会再写第二个 run_end
        events = await _db_events(run_id)
        assert [e["seq"] for e in events] == [1, 2, 3, 4]
        assert events[2]["type"] == "message_saved"  # M3：先补半截消息，再 run_end
        assert events[-1] == {"type": "run_end", "status": "failed", "stop_reason": "interrupted",
                              "error": None, "seq": 4}
    with_fresh_db(body)


def long_text_then_stall():
    """一次吐满 FLUSH_CHARS 个字：主循环自己持锁 flush（不是定时器），取消正好打在这次写库上。"""
    async def stream(**_):
        yield {"type": "text_delta", "text": "字" * runner.FLUSH_CHARS}
        await asyncio.sleep(5)
        yield {"type": "final", "content": [{"type": "text", "text": "不会到这"}], "stop_reason": "end_turn"}
    return stream


@pytest.mark.parametrize("where", ["before_seq", "after_commit"])
def test_cancel_mid_commit_keeps_push_order(monkeypatch, where):
    """O2：卡住一次写库的同时取消 run。不变量：推送的 seq 单调递增、都在 END 之前，run_end 是库里最后一条。
    before_seq：还没拿号就卡住（修前：run_end 之后又写进一条）；after_commit：已提交、没推送（修前：推送倒序）。"""
    async def body():
        gate, entered = asyncio.Event(), asyncio.Event()
        real = db.append_event

        async def gated(run_id, ev):
            if ev["type"] != "text_delta" or entered.is_set():
                return await real(run_id, ev)
            entered.set()
            if where == "before_seq":
                await gate.wait()
            out = await real(run_id, ev)
            if where == "after_commit":
                await gate.wait()
            return out
        monkeypatch.setattr(db, "append_event", gated)

        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        q = hub.subscribe(run_id)
        runner.start(cid, run_id, model_stream=long_text_then_stall())
        task = runner._tasks[run_id]
        await entered.wait()
        await runner.cancel(run_id)
        await asyncio.wait([task], timeout=0.5)  # 修前：取消路径不等被卡住的写入，这 0.5 秒里就收尾了
        gate.set()
        await asyncio.wait([task], timeout=5)
        await asyncio.sleep(0.1)  # 让被卡住的那次推送也进队列（修前它排在 END 后面）
        got = []
        while not q.empty():
            got.append(q.get_nowait())
        hub.unsubscribe(run_id, q)

        assert got and got[-1] is END                              # END 之后不能再有事件
        seqs = [e["seq"] for e in got[:-1]]
        assert seqs == sorted(set(seqs))                           # 单调递增、不重
        events = await _db_events(run_id)
        assert events[-1]["type"] == "run_end"                     # run_end 是最后一条
        assert got[:-1] == events                                  # 推出去的 == 库里的
    with_fresh_db(body)


async def _listener_pid(exclude=None):
    """hub 那条专用 LISTEN 连接在库里的 pid（asyncpg 的 add_listener 最后执行的就是这句）。"""
    for _ in range(100):
        pid = await db.pool().fetchval("SELECT pid FROM pg_stat_activity WHERE datname = current_database() "
                                       "AND query = 'LISTEN \"run_events\"' AND pid IS DISTINCT FROM $1", exclude)
        if pid:
            return pid
        await asyncio.sleep(0.05)
    raise AssertionError("LISTEN connection not up")


async def _take(q, n):
    return [await asyncio.wait_for(q.get(), 2) for _ in range(n)]


def test_other_instance_gets_events_via_notify_and_catches_up_after_listen_dies(monkeypatch):
    """O2：other 模拟实例 B，它只能靠 NOTIFY 知道有新事件（本机的 wake 只到全局 hub）。
    杀掉 B 的 LISTEN 连接，断开期间写的事件靠「重连后唤醒所有拉取任务、按游标补拉」补齐，不缺不重。"""
    async def body():
        reconnect = asyncio.Event()
        real_connect, calls = hub_mod.asyncpg.connect, []

        async def gated_connect(*a, **kw):  # 第二次连接（重连）等测试放行：断开期间的写入一定没有通知
            calls.append(1)
            if len(calls) > 1:
                await reconnect.wait()
            return await real_connect(*a, **kw)
        monkeypatch.setattr(hub_mod.asyncpg, "connect", gated_connect)

        other = hub_mod.RunHub()
        listener = asyncio.create_task(other.listen(TEST_URL))
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        pid = await _listener_pid()
        q = other.subscribe(run_id)
        for i in range(3):
            await db.append_event(run_id, {"type": "text_delta", "text": str(i)})
        assert [e["seq"] for e in await _take(q, 3)] == [1, 2, 3]

        await db.pool().execute("SELECT pg_terminate_backend($1)", pid)
        while len(calls) < 2:  # 等 B 发现断开、开始重连
            await asyncio.sleep(0.01)
        for i in range(3, 5):  # 断开期间：没有人在 LISTEN，这几条（连同 run_end）的通知都丢了
            await db.append_event(run_id, {"type": "text_delta", "text": str(i)})
        await db.finish_run(run_id, "completed", "end_turn")  # 最后一批都在断开期间：重连后不会再有新通知
        await asyncio.sleep(0.2)
        assert q.empty()
        reconnect.set()
        got = await _take(q, 4)  # 只能靠「重连后唤醒所有拉取任务」补齐，否则流永远卡着
        assert [e["seq"] for e in got[:3]] == [4, 5, 6] and got[2]["type"] == "run_end" and got[3] is END
        listener.cancel()
    with_fresh_db(body)


def test_slow_consumer_is_disconnected_then_resumes_from_cursor(monkeypatch):
    """O2：SSE 生成器停着不拉（客户端读得慢，响应写不出去），事件超过队列上限。
    断言：队列不超过上限、只剩 RESET；流不发 run_end 就结束；带 Last-Event-ID 重连后补齐、不重。"""
    async def body():
        monkeypatch.setattr(hub_mod, "QUEUE_MAX", 5)
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        await db.append_event(run_id, {"type": "text_delta", "text": "0"})
        run = await db.get_run(run_id)
        sse = main.stream_run(run=run, after_seq=None, last_event_id=None)
        assert (await anext(sse)).data["seq"] == 1                 # 回放了第一条，然后客户端不读了
        q = next(iter(hub._runs[run_id].subs))
        for i in range(1, 12):
            await db.append_event(run_id, {"type": "text_delta", "text": str(i)})
            hub.wake(run_id)
            await asyncio.sleep(0.01)
            assert q.qsize() <= 5
        assert q.qsize() == 1 and q.get_nowait() is RESET
        assert run_id not in hub._runs                             # 唯一的订阅者被断开：拉取任务也退出了
        q.put_nowait(RESET)  # 放回去，让生成器自己读到
        rest = [e async for e in sse]
        assert rest == []                                          # 断开：没有 run_end，前端会重连
        await db.finish_run(run_id, "completed", "end_turn")
        again = [e.data async for e in main.stream_run(run=run, after_seq=None, last_event_id=1)]
        assert [e["seq"] for e in again] == list(range(2, 14)) and again[-1]["type"] == "run_end"
    with_fresh_db(body)


def test_o7_retract_and_continuation_go_through_the_same_outlet():
    """O7：retract / continued 是普通事件，拿 seq 落库，推出去的 == 库里的；续写提示存成一条 user 消息。
    retract 之前那段缓冲里的字先 flush（emit 先写缓冲），所以回放时 retract 一定排在它要撤的字后面。"""
    attempts = []

    async def stream(**_):
        attempts.append(1)
        if len(attempts) == 1:
            yield {"type": "text_delta", "text": "我查"}
            raise ValueError("bad json")  # 入参 JSON 坏了 → 撤回、重发
        if len(attempts) == 2:
            yield {"type": "text_delta", "text": "前半"}
            yield {"type": "final", "content": [{"type": "text", "text": "前半"}], "stop_reason": "max_tokens"}
            return
        yield {"type": "text_delta", "text": "后半"}
        yield {"type": "final", "content": [{"type": "text", "text": "后半"}], "stop_reason": "end_turn"}

    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "写报告"}])
        events = [ev for _, ev in await _run_and_collect(cid, run_id, stream)]
        assert [e["type"] for e in events] == ["text_delta", "retract", "text_delta", "message_saved",
                                               "message_saved", "continued", "text_delta", "message_saved", "run_end"]
        assert events == await _db_events(run_id)
        assert [e["seq"] for e in events] == list(range(1, 10))
        roles = [(m["role"], m["content"][0]["text"][:10]) for m in await db.list_messages(cid)]
        assert roles[1:] == [("assistant", "前半"), ("user", '<loop_note'), ("assistant", "后半")]
        assert events[-1]["status"] == "completed"
    with_fresh_db(body)



def test_o7_stop_after_retract_does_not_store_retracted_text():
    """O7 复核①：停止时存的半截文字只取最后一个 message_saved / retract 之后的。修前存成「我查我查一下」。"""
    attempts = []

    async def stream(**_):
        attempts.append(1)
        if len(attempts) == 1:
            yield {"type": "text_delta", "text": "我查"}
            raise ValueError("bad json")
        yield {"type": "text_delta", "text": "我查一下"}
        await asyncio.sleep(5)  # 卡在重试的这次里，等用户点停止

    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "北京天气？"}])
        events = [ev for _, ev in await _run_and_collect(cid, run_id, stream, cancel_after=0.4)]
        assert [e["type"] for e in events] == ["text_delta", "retract", "text_delta", "message_saved", "run_end"]
        stopped = [m for m in await db.list_messages(cid) if m["status"] == "stopped"]
        assert [b["text"] for b in stopped[0]["content"] if b["type"] == "text"] == ["我查一下"]
    with_fresh_db(body)
