"""M7 第 4 步：子 run 真落库、父子关系、父流里的胶囊状态、级联取消。假模型按 system 区分父和子。"""

import asyncio
import contextlib

import pytest

from app import db, llm, runner, subagent
from test_db import _conversation, with_fresh_db

TASKS = [{"title": "甲", "task": "ok-甲"}, {"title": "乙", "task": "slow-乙"}, {"title": "丙", "task": "boom-丙"}]


def fake(tasks, child_started: asyncio.Event, n_started: int):
    """父：第 1 轮派子任务，第 2 轮收尾。子：ok 直接答，slow 卡死，boom 抛异常。"""
    calls, gate, started = [], asyncio.Event(), []

    async def stream(**kw):
        calls.append(kw)
        if kw["system"][-1]["text"] == subagent.CONTRACT:
            task = kw["messages"][1]["content"][0]["text"]
            if task.startswith("slow"):
                started.append(task)
                if len(started) == n_started:
                    child_started.set()
                await gate.wait()  # 永远不放行：只能被超时或取消结束
            if task.startswith("boom"):
                raise RuntimeError("upstream secret body")
            content, stop = [{"type": "text", "text": f"结论：{task}"}], "end_turn"
        elif sum(c["system"][-1]["text"] != subagent.CONTRACT for c in calls) == 1:
            content = [{"type": "tool_use", "id": "sp1", "name": "spawn_subagents", "input": {"tasks": tasks}}]
            stop = "tool_use"
        else:
            content, stop = [{"type": "text", "text": "汇总完毕"}], "end_turn"
        yield {"type": "final", "content": content, "stop_reason": stop,
               "usage": {"input_tokens": 10, "output_tokens": 5}}
    return stream, calls


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(subagent, "SUBAGENT_TIMEOUT", 0.5)  # 只有 slow 会碰到；它卡在 Event 上，结果不赌时序
    return monkeypatch


async def _start(parent_text="帮我分头调研"):
    _, cid = await _conversation()
    run_id = await db.start_run(cid, [{"type": "text", "text": parent_text}])
    runner.start(cid, run_id)
    return cid, run_id


def test_children_persisted_and_partial_results(patched):
    async def body():
        stream, calls = fake(TASKS, asyncio.Event(), 99)
        patched.setattr(llm, "stream_message", stream)
        cid, run_id = await _start()
        await runner._tasks[run_id]

        assert (await db.get_run(run_id))["status"] == "completed"
        kids = await db.child_runs(run_id)
        got = {k["title"]: (k["status"], k["stop_reason"]) for k in kids}
        assert got == {"甲": ("completed", "end_turn"), "乙": ("failed", "timeout"), "丙": ("failed", "internal_error")}
        jia = next(k for k in kids if k["title"] == "甲")  # 三个子 run 并发建，created_at 先后不固定，按名字取
        assert jia["usage"] == {"input_tokens": 10, "output_tokens": 5}  # 留给 M8
        parents = await db.pool().fetch("SELECT parent_conversation_id, parent_tool_use_id FROM conversations c "
                                        "JOIN runs r ON r.conversation_id = c.id WHERE r.parent_run_id = $1", run_id)
        assert {(p["parent_conversation_id"], p["parent_tool_use_id"]) for p in parents} == {(cid, "sp1")}

        # 父模型拿到的汇总：部分成功如实写，异常原文不进去
        result = (await db.list_messages(cid))[2]["content"][0]["content"]
        assert "完成 1，未完成 2" in result and "结论：ok-甲" in result and "执行中超时" in result
        assert "RuntimeError" in result and "secret" not in result

        # 子智能体只看到任务说明（不含父的提问），工具里没有 spawn_subagents
        child = next(c for c in calls if c["system"][-1]["text"] == subagent.CONTRACT)
        assert "帮我分头调研" not in str(child["messages"])
        assert "spawn_subagents" not in [t["name"] for t in child["tools"]]

        # 父流里每个子任务：queued → running（带 child_run_id）→ 终态，seq 递增
        evs = [e for e in await db.list_events(run_id, 0) if e["type"] == "subagent_update"]
        for i, final in enumerate(["completed", "timeout", "failed"]):
            mine = [e for e in evs if e["index"] == i]
            assert [e["status"] for e in mine] == ["queued", "running", final]
        assert {e["child_run_id"] for e in evs if e["status"] == "running"} == {str(k["id"]) for k in kids}
        assert [e["seq"] for e in evs] == sorted(e["seq"] for e in evs)
        # 子 run 自己的事件流能续播：消息 + run_end
        assert [e["type"] for e in await db.list_events(jia["id"], 0)][-2:] == ["message_saved", "run_end"]
    with_fresh_db(body)


def test_stop_parent_cascades_to_children(patched):
    patched.setattr(subagent, "SUBAGENT_TIMEOUT", 30)

    async def body():
        both = asyncio.Event()
        stream, _ = fake([{"title": "乙1", "task": "slow-1"}, {"title": "乙2", "task": "slow-2"}], both, 2)
        patched.setattr(llm, "stream_message", stream)
        cid, run_id = await _start()
        await both.wait()  # 两个子任务都卡在模型里了
        await runner.cancel(run_id)
        with contextlib.suppress(asyncio.CancelledError):
            await runner._tasks[run_id]
        assert (await db.get_run(run_id))["status"] == "cancelled"
        kids = await db.child_runs(run_id)
        assert [(k["status"], k["stop_reason"]) for k in kids] == [("cancelled", "parent_cancel")] * 2
    with_fresh_db(body)
