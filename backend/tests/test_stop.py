"""O1 第 2 步：跨实例停止。意图只落库，属主轮询发现；协作取消超时就在库里强制收尾、属主换墓碑。需要本地 PG。"""

import asyncio
import time

from app import db, runner, tools
from test_db import _conversation, with_fresh_db
from test_events import _db_events, stalling_model


def _fast(monkeypatch, force_after=0.5):
    monkeypatch.setattr(runner, "STOP_POLL", 0.1)
    monkeypatch.setattr(runner, "FORCE_AFTER", force_after)


def test_stop_requested_on_other_instance_reaches_owner(monkeypatch):
    """B 上点停止 = B 只把意图写进库（它没有 task）。A 的 stop_loop 在一个轮询间隔内发现并取消，按用户停止收尾。"""
    _fast(monkeypatch)

    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        poll = asyncio.create_task(runner.stop_loop())
        try:
            runner.start(cid, run_id, model_stream=stalling_model(5))
            task = runner._tasks[run_id]
            await asyncio.sleep(0.2)
            t0 = time.monotonic()
            assert await db.request_cancel(run_id) == [run_id]  # 实例 B 的停止接口做的就是这一步
            await asyncio.wait([task], timeout=3)
            assert task.done() and time.monotonic() - t0 < 1
        finally:
            poll.cancel()
        run = await db.get_run(run_id)
        assert (run["status"], run["stop_reason"], run["error"]) == ("cancelled", "user_cancel", None)
        # 「是用户停的」来自库里的 cancelling，不是本进程内存：照样存了 stopped 消息
        assert (await db.list_messages(cid))[-1]["status"] == "stopped"
    with_fresh_db(body)


def _stubborn_tool(started: asyncio.Event, seconds: float) -> tools.Tool:
    """吞掉所有取消、睡满 seconds 才返回：模拟写得不好的第三方 SDK。"""
    async def handler(_args):
        started.set()
        end = asyncio.get_running_loop().time() + seconds
        while (left := end - asyncio.get_running_loop().time()) > 0:
            try:
                await asyncio.sleep(left)
            except asyncio.CancelledError:
                pass  # 故意吞掉
        return "做完了"
    return tools.Tool(name="stubborn", description="test", input_schema={"type": "object", "properties": {}},
                      handler=handler, permission=tools.PUBLIC, risk=tools.LOW, timeout=60, record_execution=True)


def _calls_tool_once(name):
    async def stream(messages, **_):
        if messages[-1]["role"] == "user" and any(b.get("type") == "tool_result" for b in messages[-1]["content"]):
            yield {"type": "final", "content": [{"type": "text", "text": "好了"}], "stop_reason": "end_turn"}
            return
        yield {"type": "text_delta", "text": "我来执行"}
        yield {"type": "final", "stop_reason": "tool_use", "content": [
            {"type": "text", "text": "我来执行"}, {"type": "tool_use", "id": "s1", "name": name, "input": {}}]}
    return stream


def test_force_cancel_when_tool_swallows_cancel(monkeypatch):
    """工具吞了取消：0.5 秒后库里强制收尾（cancelled + force_cancel、stopped 消息、执行记录结果未知），
    属主换成墓碑；工具 2 秒后醒来，task 的写入全被拒，run_end 之后一条事件都没有。"""
    _fast(monkeypatch)

    async def body():
        started = asyncio.Event()
        tools.register(_stubborn_tool(started, 2.0))
        poll = asyncio.create_task(runner.stop_loop())
        try:
            _, cid = await _conversation()
            run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
            runner.start(cid, run_id, model_stream=_calls_tool_once("stubborn"))
            task = runner._tasks[run_id]
            await started.wait()
            await db.request_cancel(run_id)
            await asyncio.sleep(1.0)  # 轮询 0.1 + 看门狗 0.5 + 余量；工具还在睡
            assert not task.done()
            run = await db.get_run(run_id)
            assert (run["status"], run["stop_reason"], run["error"]) == ("cancelled", "user_cancel", "force_cancel")
            assert run["owner_id"] != db.INSTANCE_ID  # 墓碑
            assert await db.pool().fetchval("SELECT status FROM tool_executions") == "outcome_unknown"
            assert (await db.list_messages(cid))[-1]["status"] == "stopped"
            ended_at = (await _db_events(run_id))[-1]

            await asyncio.wait([task], timeout=5)  # 工具醒来 → 写库被拒 → 悄悄退出
            assert task.done() and task.cancelled()  # 不是带着 LeaseLost 异常退出
            assert (await _db_events(run_id))[-1] == ended_at and ended_at["type"] == "run_end"
            assert run_id not in runner._tasks
        finally:
            poll.cancel()
            tools.unregister("stubborn")
    with_fresh_db(body)
