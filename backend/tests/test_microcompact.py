"""O9 第二步：清旧工具结果。确定的投影、分批、最近的不动、排在 LLM 摘要前面、水位存库。"""

import asyncio

from app import budget, db, microcompact, runner, tools
from app.agent_loop import run_loop
from test_db import _conversation, with_fresh_db

PAGE = "这是一大段抓回来的网页正文。" * 300  # 约 4500 字，一条约 4000+ token 估算


def fetch_round(i: int, mid: int, content: str = PAGE) -> list[dict]:
    """第 i 次 get_weather（clearable）：assistant 的 tool_use + user 的 tool_result，带消息 id（模拟已落库）。"""
    return [{"id": mid, "role": "assistant", "content": [{"type": "tool_use", "id": f"w{i}", "name": "get_weather",
                                                         "input": {"city": f"城市{i}"}}]},
            {"id": mid + 1, "role": "user", "content": [{"type": "tool_result", "tool_use_id": f"w{i}",
                                                         "content": content}]}]


def history(n: int, content: str = PAGE) -> list[dict]:
    h = [{"id": 1, "role": "user", "content": "帮我对比这些城市"}]
    for i in range(n):
        h += fetch_round(i, 10 + 2 * i, content)
    return h


def results(h):
    return [b["content"] for m in h if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"]


def test_oldest_are_replaced_with_fixed_placeholders_recent_untouched():
    h = history(8)
    w = microcompact.next_watermark(h, 0)
    assert w == 10 + 2 * 4 + 1  # 8 条里最近 3 条不动，前 5 条成一批
    out = results(microcompact.project(h, w))
    assert all(x.startswith("[已清理的旧结果]") for x in out[:5]) and out[5:] == [PAGE] * 3
    assert results(h) == [PAGE] * 8  # 原 history（库里的原文）不动


def test_projection_is_byte_for_byte_stable():
    h = history(8)
    h[2 * 5 + 2]["content"][0]["content"] = "晴"  # 第 6 次的结果很小：窗口后移时新进来的这条不够一批
    w = microcompact.next_watermark(h, 0)
    assert microcompact.project(h, w) == microcompact.project(h, w)
    # 第 N+1 轮：新加一条小结果，不够一批，水位不动，前面的投影逐字节不变
    h2 = h + fetch_round(99, 100, "晴")
    assert microcompact.next_watermark(h2, w) is None
    assert microcompact.project(h2, w)[:len(h)] == microcompact.project(h, w)


def test_small_batch_is_not_worth_touching_the_prefix():
    h = history(8, content="晴，20°C")
    assert microcompact.next_watermark(h, 0) is None


def test_errors_and_non_clearable_tools_are_kept():
    h = history(5)
    h[2]["content"][0]["is_error"] = True                    # 错误结果很短、模型要据此调整，不清
    h.insert(3, {"id": 11, "role": "assistant", "content": []})
    w = 10 + 2 * 4 + 1
    out = microcompact.project(h, w)
    assert results(out)[0] == PAGE
    spawn = [{"id": 200, "role": "assistant", "content": [{"type": "tool_use", "id": "s", "name": "spawn_subagents", "input": {}}]},
             {"id": 201, "role": "user", "content": [{"type": "tool_result", "tool_use_id": "s", "content": PAGE}]}]
    assert results(microcompact.project(spawn, 10**6)) == [PAGE]  # 子智能体汇报重做很贵：不清


def test_kb_placeholder_keeps_block_headers():
    kb = "[KB_RESULT] ...\n--- 《差旅报销制度》第1段\n打车上限 200 元\n--- 《错误码手册》 · 设备类 · 第5段\nE1024"
    p = microcompact.placeholder("kb_search", kb)
    assert "《差旅报销制度》第1段" in p and "设备类 · 第5段" in p and "200" not in p


def test_call_tool_unwrapped_for_clearable_check():
    h = [{"id": 1, "role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": tools.CALL_TOOL,
                                                     "input": {"tool_name": "get_weather", "arguments": {}}}]},
         {"id": 2, "role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": PAGE}]}]
    assert results(microcompact.project(h, 2))[0].startswith("[已清理的旧结果] 这是较早一次 get_weather")


def test_loop_clears_before_summarizing_and_reports_tokens():
    """超过触发线：先清旧结果（不调模型），清完低于触发线就不调摘要。"""
    h = history(8)
    summarized = []

    async def compactor(*a):
        summarized.append(1)
        return None

    async def stream(**kw):
        stream.sent = kw["messages"]
        yield {"type": "final", "content": [{"type": "text", "text": "对比完了"}], "stop_reason": "end_turn"}
    total = budget.estimate_json(results(h))

    async def go():
        return [e async for e in run_loop(h, model_stream=stream, compactor=compactor,
                                          limits={"trigger": total // 2, "target": total // 3})]
    events = asyncio.run(go())
    cleared = next(e for e in events if e["type"] == "cleared")
    assert cleared["until_id"] == 19 and cleared["tokens_after"] < cleared["tokens_before"]
    assert summarized == []
    assert sum(1 for x in results(stream.sent) if x.startswith("[已清理")) == 5


def test_watermark_persists_and_next_run_projects_the_same_prefix():
    """水位存库（只升不降），下一个 run 读出来拼出同样的投影：清过的不会被放回来。"""
    async def body():
        _, cid = await _conversation()
        first = await db.start_run(cid, [{"type": "text", "text": "对比城市"}])
        for i in range(8):
            await db.insert_message(cid, first, "assistant", [{"type": "tool_use", "id": f"w{i}", "name": "get_weather",
                                                               "input": {"city": f"城市{i}"}}])
            await db.insert_message(cid, first, "user", [{"type": "tool_result", "tool_use_id": f"w{i}", "content": PAGE}])
        await db.finish_run(first, "completed", "end_turn")
        sent = []

        async def stream(**kw):
            sent.append(kw["messages"])
            yield {"type": "final", "content": [{"type": "text", "text": "好"}], "stop_reason": "end_turn"}
        total = budget.estimate_json([PAGE] * 8)
        for q in ("第二问", "第三问"):
            run_id = await db.start_run(cid, [{"type": "text", "text": q}])
            await runner._run(cid, run_id, model_stream=stream, limits={"trigger": total // 2, "target": total // 3})
        cl = await db.get_cleared(cid)
        assert cl["cleared_until_id"] > 0 and cl["cleared_at_id"] >= cl["cleared_until_id"]
        # 第 3 个 run 没有再清（水位够了），前缀和第 2 个 run 的投影逐字节一致
        assert sent[1][:len(sent[0]) - 1] == sent[0][:-1]
        await db.save_cleared(cid, run_id, {"type": "cleared", "until_id": 1, "at_id": 1})  # 只升不降
        assert (await db.get_cleared(cid))["cleared_until_id"] == cl["cleared_until_id"]
    with_fresh_db(body)



def test_stale_usage_is_dropped_after_clearing():
    """清理前的 usage 是那份大请求的：不丢掉的话，估算还是按它打底，清完照样判超线、又去调摘要。"""
    h = history(8)
    total = budget.estimate_json(results(h))
    h[-2]["usage"] = {"input_tokens": total, "output_tokens": 10}  # 最近一条 assistant 带着清理前的大 usage
    summarized = []

    async def compactor(*a):
        summarized.append(1)
        return None

    async def stream(**kw):
        yield {"type": "final", "content": [{"type": "text", "text": "好"}], "stop_reason": "end_turn"}

    async def go():
        return [e async for e in run_loop(h, model_stream=stream, compactor=compactor,
                                          limits={"trigger": total // 2, "target": total // 3})]
    events = asyncio.run(go())
    assert any(e["type"] == "cleared" for e in events) and summarized == []



# ---------------- 复核补的用例 ----------------

def test_malformed_call_tool_does_not_crash_clearing():
    """复核①：tool_name 是列表、input 不是对象的 call_tool 原样留在历史里，算水位和投影都不能崩。"""
    bad = [{"id": 3, "role": "assistant", "content": [
               {"type": "tool_use", "id": "x1", "name": tools.CALL_TOOL, "input": {"tool_name": ["get_weather"], "arguments": {}}},
               {"type": "tool_use", "id": "x2", "name": tools.CALL_TOOL, "input": "坏了"}]},
           {"id": 4, "role": "user", "content": [{"type": "tool_result", "tool_use_id": "x1", "content": "e", "is_error": True},
                                                 {"type": "tool_result", "tool_use_id": "x2", "content": "e", "is_error": True}]}]
    h = history(8)[:1] + bad + history(8)[1:]
    w = microcompact.next_watermark(h, 0)
    assert w is not None and microcompact.project(h, w)


def test_projection_does_not_depend_on_the_live_registry(monkeypatch):
    """复核②：重启时少注册了某个工具（WEB_TOOLS=0、沙箱没开、MCP 没了），清过的不能被放回来。"""
    h = history(8)
    w = microcompact.next_watermark(h, 0)
    before = microcompact.project(h, w)
    monkeypatch.delitem(tools._REGISTRY, "get_weather")
    assert microcompact.project(h, w) == before


def test_web_results_are_not_cleared():
    """复核⑤：抓取 / 搜索的旧结果不清：清掉以后再抓同一链接要重新确认，搜索结果的链接也没了。"""
    h = [{"id": 1, "role": "assistant", "content": [{"type": "tool_use", "id": "f", "name": "fetch", "input": {"url": "https://x"}}]},
         {"id": 2, "role": "user", "content": [{"type": "tool_result", "tool_use_id": "f", "content": PAGE}]}]
    assert results(microcompact.project(h, 10**6)) == [PAGE]


def test_summary_tier_sizes_and_summarizes_the_projection():
    """复核③：清完还超线才轮到摘要：摘要那一级拿到的是投影（清掉的模型本来就看不到），返回的保留部分换回原文。"""
    h = history(8)
    seen = {}

    async def compactor(hist, summary, room, measure):
        seen["n_cleared"] = sum(1 for x in results(hist) if x.startswith("[已清理"))
        return "摘要", hist[-4:]

    async def stream(**kw):
        stream.sent = kw["messages"]
        yield {"type": "final", "content": [{"type": "text", "text": "好"}], "stop_reason": "end_turn"}
    total = budget.estimate_json(results(h))

    async def go():
        return [e async for e in run_loop(h, model_stream=stream, compactor=compactor,
                                          limits={"trigger": total // 10, "target": total // 20})]
    events = asyncio.run(go())
    assert seen["n_cleared"] == 5
    assert next(e for e in events if e["type"] == "compacted")["kept_messages"] == 4
    assert results(stream.sent)[-1] == PAGE  # 最近的原文仍在（保留部分是原文，不是投影的副本）


def _rounds(n=8):
    out = []
    for i in range(n):
        out.append(("assistant", [{"type": "tool_use", "id": f"w{i}", "name": "get_weather", "input": {"city": f"城市{i}"}}]))
        out.append(("user", [{"type": "tool_result", "tool_use_id": f"w{i}", "content": PAGE}]))
    return out


def test_next_run_uses_the_persisted_watermark_even_when_not_over_the_line():
    """复核④：下一个 run 不超线、不会再算水位，也必须按库里的水位投影（不能每次从 0 重算）。"""
    async def body():
        _, cid = await _conversation()
        first = await db.start_run(cid, [{"type": "text", "text": "对比"}])
        for role, content in _rounds():
            await db.insert_message(cid, first, role, content)
        await db.finish_run(first, "completed", "end_turn")
        sent = []

        async def stream(**kw):
            sent.append(kw["messages"])
            yield {"type": "final", "content": [{"type": "text", "text": "好"}], "stop_reason": "end_turn"}
        total = budget.estimate_json([PAGE] * 8)
        run_id = await db.start_run(cid, [{"type": "text", "text": "第二问"}])
        await runner._run(cid, run_id, model_stream=stream, limits={"trigger": total // 2, "target": total // 3})
        run_id = await db.start_run(cid, [{"type": "text", "text": "第三问"}])
        await runner._run(cid, run_id, model_stream=stream, limits={"trigger": 10**9, "target": 10**9})
        assert sum(1 for x in results(sent[1]) if x.startswith("[已清理")) == 5
    with_fresh_db(body)


def test_stale_usage_is_dropped_by_the_next_run_after_a_failed_clearing_run(monkeypatch):
    """复核④：清理的那个 run 没产出任何 assistant 就失败了，库里最新的 usage 还是清理前的大请求；
    下一个 run 必须按 cleared_at_id 丢掉它，否则一估算又超线、又去调摘要。"""
    from app import compaction, llm
    summarized = []

    async def fake_summarize(old, msgs):
        summarized.append(1)
        return "摘要"
    monkeypatch.setattr(compaction, "summarize", fake_summarize)

    async def body():
        _, cid = await _conversation()
        first = await db.start_run(cid, [{"type": "text", "text": "对比"}])
        total = budget.estimate_json([PAGE] * 8)
        for role, content in _rounds():
            await db.insert_message(cid, first, role, content)
        await db.insert_message(cid, first, "assistant", [{"type": "text", "text": "对比完了"}],
                                usage={"input_tokens": total, "output_tokens": 10})
        await db.finish_run(first, "completed", "end_turn")
        limits = {"trigger": total // 2, "target": total // 3}

        async def failing(**kw):
            raise llm.LLMError("boom")
            yield
        run_id = await db.start_run(cid, [{"type": "text", "text": "第二问"}])
        await runner._run(cid, run_id, model_stream=failing, limits=limits)
        assert (await db.get_cleared(cid))["cleared_at_id"] > 0 and summarized == []

        async def ok(**kw):
            yield {"type": "final", "content": [{"type": "text", "text": "好"}], "stop_reason": "end_turn"}
        run_id = await db.start_run(cid, [{"type": "text", "text": "第三问"}])
        await runner._run(cid, run_id, model_stream=ok, limits=limits)
        assert summarized == []
    with_fresh_db(body)
