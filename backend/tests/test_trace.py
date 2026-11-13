"""M8 第 2 步：trace。钉住「子 run 挂在 spawn 工具 span 下」「失败的 LLM 调用也有 span」「停止后不留 running」。
假模型复用 M7 的 fake：父派 甲（正常）/ 乙（超时）/ 丙（模型抛异常）。"""

import asyncio
import contextlib
from datetime import datetime, timedelta
from uuid import uuid4

from app import db, llm, runner, trace, trace_api
from test_db import with_fresh_db
from test_subagent_runs import TASKS, _start, fake, patched  # noqa: F401  patched 是 fixture


def test_build_tree_orders_by_time_and_keeps_orphans_as_roots():
    t0 = datetime(2026, 10, 30)
    a, b, c, orphan = uuid4(), uuid4(), uuid4(), uuid4()
    rows = [{"id": c, "parent_id": a, "started_at": t0 + timedelta(seconds=2), "ended_at": None},
            {"id": b, "parent_id": a, "started_at": t0 + timedelta(seconds=1), "ended_at": t0 + timedelta(seconds=1.5)},
            {"id": a, "parent_id": None, "started_at": t0, "ended_at": t0 + timedelta(seconds=3)},
            {"id": orphan, "parent_id": uuid4(), "started_at": t0, "ended_at": t0}]
    roots = trace.build_tree(rows)
    assert {r["id"] for r in roots} == {a, orphan}
    root = next(r for r in roots if r["id"] == a)
    assert [n["id"] for n in root["children"]] == [b, c]  # 按开始时间，不按插入顺序
    assert root["duration_ms"] == 3000 and root["children"][0]["duration_ms"] == 500
    assert root["children"][1]["duration_ms"] is None  # 还没结束


def shape(node, depth=0):
    """树 → 缩进的 (kind, name, status) 列表；子 run 按 run_id 排不稳定，按名字和状态排序后比。"""
    out = [(depth, node["kind"], node["name"], node["status"])]
    for ch in node["children"]:
        out += shape(ch, depth + 1)
    return out


def test_span_tree_of_parent_with_subagents(patched):  # noqa: F811
    async def body():
        stream, _ = fake(TASKS, asyncio.Event(), 99)
        patched.setattr(llm, "stream_message", stream)
        _, run_id = await _start()
        await runner._tasks[run_id]

        rows = await db.pool().fetch("SELECT * FROM spans")
        assert {r["trace_id"] for r in rows} == {run_id}  # 子 run 的 span 也在父的 trace 里
        assert all(r["status"] != "running" and r["ended_at"] for r in rows)
        [root] = await trace.load(run_id)
        assert [(c["kind"], c["name"]) for c in root["children"]] == [
            ("llm", "llm #1"), ("tool", "spawn_subagents"), ("llm", "llm #2")]
        assert root["status"] == "ok" and root["attrs"]["stop_reason"] == "end_turn"
        llm1, spawn, _ = root["children"]
        assert llm1["attrs"]["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert llm1["attrs"]["hit_rate"] == 0.0  # O3：每轮的命中率记在 llm span 上（假模型没有首字时间）
        assert spawn["attrs"]["tool_use_id"] == "sp1"

        kids = {k["run_id"]: k for k in spawn["children"]}  # 子 run 挂在 spawn 工具 span 下（用户选 B）
        titles = {r["id"]: r["title"] for r in await db.child_runs(run_id)}
        by_title = {titles[rid]: k for rid, k in kids.items()}
        assert by_title["甲"]["status"] == "ok" and by_title["甲"]["children"][0]["status"] == "ok"
        assert by_title["乙"]["status"] == "error" and by_title["乙"]["attrs"]["stop_reason"] == "timeout"
        assert by_title["乙"]["children"][0]["status"] == "cancelled"  # 卡在模型里被超时取消
        boom = by_title["丙"]["children"][0]  # 模型调用本身失败：没有消息，但 span 在
        assert boom["kind"] == "llm" and boom["status"] == "error" and "RuntimeError" in boom["error"]
        io = await trace_api.tool_io([run_id])  # 查看页按 tool_use_id 关联全文（span 里不复制）
        assert io["sp1"]["input"]["tasks"] == TASKS and "完成 1，未完成 2" in str(io["sp1"]["result"])
        for k in kids.values():  # 子 run 的时间段落在 spawn 工具 span 里面
            assert spawn["started_at"] <= k["started_at"] and k["ended_at"] <= spawn["ended_at"]
    with_fresh_db(body)


def test_stop_leaves_no_running_span_and_recover_marks_interrupted(patched):  # noqa: F811
    patched.setattr(runner.subagent, "SUBAGENT_TIMEOUT", 30)

    async def body():
        both = asyncio.Event()
        stream, _ = fake([{"title": "乙1", "task": "slow-1"}, {"title": "乙2", "task": "slow-2"}], both, 2)
        patched.setattr(llm, "stream_message", stream)
        _, run_id = await _start()
        await both.wait()
        await runner.cancel(run_id)
        with contextlib.suppress(asyncio.CancelledError):
            await runner._tasks[run_id]
        statuses = {(r["kind"], r["status"]) for r in await db.pool().fetch("SELECT * FROM spans")}
        assert ("run", "cancelled") in statuses and ("tool", "cancelled") in statuses
        assert not any(s == "running" for _, s in statuses)

        # 模拟崩溃留下的 running span：启动对账改成 interrupted
        await trace.start("tool", "half", run_id)  # O1：span 跟着 run 走，run 也得是「崩溃时还在跑、租约过期」
        await db.pool().execute("UPDATE runs SET status = 'running', lease_until = now() - interval '1 second' "
                                "WHERE id = $1", run_id)
        assert await db.recover_expired_runs() == 1
        assert await db.pool().fetchval("SELECT status FROM spans WHERE name = 'half'") == "interrupted"
    with_fresh_db(body)


def test_first_token_counts_thinking_and_final_carries_it(monkeypatch):
    """O3：首字时间 = 请求发出 → 第一个内容增量（thinking 也算），跟着 final 走，trace 记到 llm span。"""
    from types import SimpleNamespace as NS
    now = [10.0]  # 假时钟随事件推进：哪个事件到了，就是那一刻（不按调用次数，否则只认正文也会读到 10.25）

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def __aiter__(self):
            async def gen():
                for t, ev in ((10.1, NS(type="message_start")),
                              (10.25, NS(type="content_block_delta", delta=NS(type="thinking_delta"))),
                              (10.5, NS(type="content_block_delta", delta=NS(type="text_delta"))),
                              (10.5, NS(type="text", text="好"))):
                    now[0] = t
                    yield ev
            return gen()

        async def get_final_message(self):
            return NS(content=[], stop_reason="end_turn", usage=NS(input_tokens=1, output_tokens=1))

    monkeypatch.setenv("MODEL", "claude-opus-5-5")  # 固定走 Anthropic 协议
    monkeypatch.setattr(llm.anthropic_api, "_get_client", lambda m: NS(messages=NS(stream=lambda **kw: FakeStream())))
    monkeypatch.setattr(llm.anthropic_api, "time", NS(monotonic=lambda: now[0]))  # 只换 llm 看到的时钟，asyncio 也在用 time.monotonic

    async def collect():
        return [ev async for ev in llm.stream_message(system=[], messages=[], tools=[], tool_choice={"type": "auto"})]
    evs = asyncio.run(collect())
    assert evs[-1]["first_token_ms"] == 250  # 不是看到正文的 500
