"""M7 第 3 步：并发上限、超时、部分失败、汇总截断。run_child 是假的，用 Event 卡住，不靠 sleep 赌时序。"""

import asyncio

import pytest

from app import subagent


def subs(n):
    return [{"title": f"任务{i}", "task": f"做第 {i} 件事"} for i in range(n)]


def test_concurrency_limit_holds():
    async def main():
        sem, gate, full = asyncio.Semaphore(2), asyncio.Event(), asyncio.Event()
        active = peak = 0

        async def child(sub, cm):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                full.set()
            await gate.wait()  # 卡住，直到测试放行
            active -= 1
            return {"status": "completed", "text": sub["title"]}

        job = asyncio.create_task(subagent.spawn(subs(5), child, sem=sem, timeout=10))
        await full.wait()
        for _ in range(5):  # 多让几次调度：排队的拿不到名额，怎么让都进不来
            await asyncio.sleep(0)
        assert active == 2
        gate.set()
        return peak, await job

    peak, results = asyncio.run(main())
    assert peak == 2 and [r["status"] for r in results] == ["completed"] * 5


def test_partial_failure_and_timeout_reported_separately():
    async def main():
        sem = asyncio.Semaphore(1)
        hold = asyncio.Event()  # 永远不 set：第 0 个占着名额不放，直到超时

        async def child(sub, cm):
            if sub["title"] == "任务0":
                await hold.wait()
            if sub["title"] == "任务1":
                raise RuntimeError("secret upstream body")
            return {"status": "completed", "text": "ok"}

        # 名额只有 1 个，任务0 卡死 → 任务0「执行中超时」，后面的「排队时超时」
        stuck = await subagent.spawn(subs(2), child, sem=sem, timeout=0.05)
        free = await subagent.spawn(subs(3)[1:], child, sem=asyncio.Semaphore(2), timeout=5)
        return stuck, free

    stuck, free = asyncio.run(main())
    assert [(r["status"], r["started"]) for r in stuck] == [("timeout", True), ("timeout", False)]
    assert [r["status"] for r in free] == ["failed", "completed"]  # 一个失败不拖垮别的
    text = subagent.render(stuck + free)
    assert "执行中超时" in text and "排队时超时" in text and "完成 1，未完成 3" in text
    assert "RuntimeError" in text and "secret" not in text  # 异常原文不给模型


def test_timeout_is_visible_inside_child():
    """第 5 步要靠它区分「超时」和「用户停止」：被超时取消时，子协程里 cm.expired() 已经是 True。"""
    seen = {}

    async def child(sub, cm):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            seen["expired"] = cm.expired()
            raise

    [r] = asyncio.run(subagent.spawn(subs(1), child, sem=asyncio.Semaphore(1), timeout=0.05))
    assert r["status"] == "timeout" and seen == {"expired": True}


def test_parent_cancel_cascades_to_children():
    async def main():
        cancelled, started, n = [], asyncio.Event(), 0

        async def child(sub, cm):
            nonlocal n
            try:
                n += 1
                if n == 3:  # 三个都进到执行里了再取消
                    started.set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(sub["title"])
                raise

        job = asyncio.create_task(subagent.spawn(subs(3), child, sem=asyncio.Semaphore(3), timeout=10))
        await started.wait()
        job.cancel()  # 父被取消：gather 取消所有子任务，run_one 不吞 CancelledError
        with pytest.raises(asyncio.CancelledError):
            await job
        return cancelled

    assert sorted(asyncio.run(main())) == ["任务0", "任务1", "任务2"]


def test_result_is_clipped():
    long = "字" * (subagent.RESULT_CHARS + 500)
    text = subagent.render([{"title": "长", "status": "completed", "started": True, "text": long}])
    assert f"已截断，原文 {len(long)} 字" in text and len(text) < subagent.RESULT_CHARS + 200


def test_context_budget_forces_final_turn():
    """第 6 步：上下文超预算 → 下一轮当最后一轮（不许调工具），结束原因 budget。缓存命中也要算进上下文。"""
    from app import context
    from app.agent_loop import run_loop
    from test_agent_loop import fake_model
    call = {"type": "tool_use", "id": "c1", "name": "calculate", "input": {"a": 1, "b": 2, "op": "add"}}
    stream, calls = fake_model([([call], "tool_use"), ([{"type": "text", "text": "结论：3"}], "end_turn")])

    async def with_usage(**kw):  # input 只有 100，但缓存命中 25000：DeepSeek 的 input_tokens 不含缓存
        async for ev in stream(**kw):
            if ev["type"] == "final":
                ev["usage"] = {"input_tokens": 100, "cache_read_input_tokens": 25000, "output_tokens": 20}
            yield ev

    async def go():
        return [ev async for ev in run_loop([{"role": "user", "content": "算 1+2"}], model_stream=with_usage,
                                            max_turns=6, context_limit=20000)]
    evs = asyncio.run(go())
    assert calls[1]["tool_choice"] == {"type": "none"}
    assert context.FINAL_TURN_NOTE.strip() in str(calls[1]["messages"][-1])
    assert evs[-1] == {"type": "run_end", "status": "completed", "stop_reason": "budget"}
    text = subagent.render([{"title": "算", "status": "completed", "started": True, "text": "结论：3",
                             "stop_reason": "budget"}])
    assert "预算用完" in text


def test_repeated_action_is_translated_for_parent():
    """O6：子 run 因原样重复被强制收尾（failed/repeated_action），父模型看到的是一句人话，不是原样的代号。"""
    text = subagent.render([{"title": "查", "status": "failed", "started": True, "text": "火星没有数据",
                             "stop_reason": "repeated_action", "error": "repeated_action"}])
    assert "反复" in text and "（repeated_action）" not in text and "已完成部分：火星没有数据" in text


def test_tool_unavailable_is_translated_for_parent():
    """O6 第二步：子 run 因依赖不可用被强制收尾（completed/tool_unavailable），父模型知道这份汇报是提前交的。"""
    text = subagent.render([{"title": "查", "status": "completed", "started": True, "text": "没拿到数据",
                             "stop_reason": "tool_unavailable"}])
    assert "服务不可用" in text and "可能不完整" in text


def test_closing_blocks_keep_stop_reason_for_translation():
    """O6 复核⑥：父 run 被停止 / 接管时，子 run 结果从库里读（_from_row），也要带 stop_reason，翻译才生效。"""
    from app import subagent_tool
    task = {"title": "查"}
    a = lambda t: {"role": "assistant", "content": [{"type": "text", "text": t}]}
    rep = subagent_tool._from_row(task, {"status": "failed", "stop_reason": "repeated_action", "msgs": [a("火星没有数据")]})
    down = subagent_tool._from_row(task, {"status": "completed", "stop_reason": "tool_unavailable", "msgs": [a("没拿到")]})
    text = subagent.render([rep, down])
    assert "反复" in text and "（repeated_action）" not in text
    assert "服务不可用" in text and "可能不完整" in text


def test_closing_blocks_join_a_continued_report():
    """O7 复核④：父被停止 / 接管时从库里读子 run 的汇报，续写过的也要拼回来（原来只取最后一条，丢前半截）。"""
    from app import context, subagent_tool
    a = lambda t: {"role": "assistant", "content": [{"type": "text", "text": t}]}
    msgs = [{"role": "user", "content": "简报"}, a("前半"), context.loop_note("continue", "x"), a("后半")]
    assert subagent_tool._from_row({"title": "查"}, {"status": "completed", "stop_reason": "end_turn",
                                                    "msgs": msgs})["text"] == "前半后半"
