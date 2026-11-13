"""P1-C2：分段增量摘要。固定段落 + 有旧摘要时按合并规则更新；「关键标识与 ref」由代码从工具调用生成，从原文读。"""

import asyncio
from uuid import uuid4

from app import budget, compaction, llm, tools
from app.agent_loop import run_loop
from app.compaction import REFS_HEADING, make_compactor, tool_log, with_refs

REF = "out_0123456789abcdef"
PREVIEW = "开头……\n\n[输出过长，已截断] 全文 20000 字。需要中间的内容，用 read_tool_output 按 ref=" + REF + " 和 offset 取回\n\n……结尾"


def use(i, name, inp):
    return {"type": "tool_use", "id": f"u{i}", "name": name, "input": inp}


def res(i, content, **kw):
    return {"type": "tool_result", "tool_use_id": f"u{i}", "content": content, **kw}


CALLS = [
    {"role": "assistant", "content": [use(1, "get_weather", {"city": "苏州"}), use(2, "get_stock", {"symbol": "AAPL"}),
                                      use(3, tools.CALL_TOOL, {"tool_name": "get_air_quality", "arguments": {"city": "苏州"}}),
                                      use(4, tools.READ_TOOL, {"ref": REF})]},
    {"role": "user", "content": [res(1, "苏州 晴"), res(2, PREVIEW), res(3, "[TOOL_ERROR] 服务不可用", is_error=True),
                                 res(4, "中间的内容")]},
]


def test_tool_log_lines():
    assert tool_log(CALLS) == ['- get_weather {"city": "苏州"}',
                               '- get_stock {"symbol": "AAPL"} → ref=' + REF,  # 截断过的带 ref
                               '- get_air_quality {"city": "苏州"} → 失败']   # call_tool 记真名；回取那次不记
    long = [{"role": "assistant", "content": [use(9, "kb_search", {"query": "长" * 300})]}]
    assert tool_log(long)[0].endswith("…") and len(tool_log(long)[0]) < 160


def test_model_written_refs_dropped_old_refs_kept():
    model = "## 目标\n- 年会\n" + REFS_HEADING + "\n- get_stock → ref=out_ffffffffffffffff"  # 模型抄错的
    old = "## 目标\n- 旧\n\n" + REFS_HEADING + "\n- recall {\"query\": \"预算\"}"
    out = with_refs(model, old, CALLS)
    assert out.startswith("## 目标\n- 年会\n\n" + REFS_HEADING)
    assert "out_ffff" not in out and REF in out
    assert out.index("recall") < out.index("get_weather")                 # 旧的在前，新的接在后面


def test_refs_dedupe_keep_latest_and_cap():
    old = REFS_HEADING + "\n" + "\n".join(f"- t{i} {{}}" for i in range(compaction.MAX_REFS))
    again = [{"role": "assistant", "content": [use(1, "t0", {}), use(2, "new", {})]}]  # 去重后 31 条，超 1 条
    lines = with_refs("## 目标\n- x", old, again).split(REFS_HEADING)[1].strip().splitlines()
    assert len(lines) == compaction.MAX_REFS
    assert lines[-2:] == ["- t0 {}", "- new {}"] and lines.count("- t0 {}") == 1  # 重复的只留最后一次
    assert lines[0] == "- t2 {}"                                         # 超了丢最旧的（t1）


def measure(summary, history):
    return budget.estimate_json([summary, [m["content"] for m in history]])


def test_compactor_reads_refs_from_raw_not_projection():
    """O9 清过的结果在投影里是占位符、没有 ref；ref 段要从原文读。"""
    raw = [{"role": "user", "content": "查一下股价"},
           {"role": "assistant", "content": [use(2, "get_stock", {"symbol": "AAPL"})]},
           {"role": "user", "content": [res(2, PREVIEW + "x" * 3000)]},
           {"role": "assistant", "content": [{"type": "text", "text": "查到了"}]},
           {"role": "user", "content": "好的"}]
    projected = [*raw[:2], {"role": "user", "content": [res(2, "[已清理的旧结果] 这是较早一次 get_stock 的输出")]}, *raw[3:]]

    async def fake(old, msgs, **kw):
        return "## 目标\n- 查股价"
    out = asyncio.run(make_compactor(uuid4(), 0, fake)(projected, None, 500, measure, raw=raw))
    assert out and "ref=" + REF in out[0]


def fake_llm(monkeypatch):
    sent = []

    async def stream(**kw):
        sent.append(kw)
        yield {"type": "final", "content": [{"type": "text", "text": "## 目标\n- 年会"}], "stop_reason": "end_turn"}
    monkeypatch.setattr(llm, "stream_message", stream)
    return sent


def test_merge_rules_only_when_old_summary(monkeypatch):
    sent = fake_llm(monkeypatch)

    def prompt(s, h, note):
        return {"system": [], "messages": [{"role": "user", "content": [{"type": "text", "text": note}]}], "tools": []}
    asyncio.run(compaction.summarize("旧摘要", CALLS, prompt=prompt))
    asyncio.run(compaction.summarize(None, CALLS, prompt=prompt))
    merge, fresh = (s["messages"][-1]["content"][-1]["text"] for s in sent)
    assert compaction.MERGE_RULES in merge and compaction.MERGE_RULES not in fresh
    assert "## 未回答的用户请求" in merge and "## 未回答的用户请求" in fresh  # 两种都用固定段落


def test_loop_passes_raw_history_alongside_projection():
    h = [{"id": 1, "role": "user", "content": "查天气"},
         {"id": 2, "role": "assistant", "content": [use(1, "get_weather", {"city": "苏州"})]},
         {"id": 3, "role": "user", "content": [res(1, PREVIEW + "y" * 5000)]},
         {"id": 4, "role": "assistant", "content": [{"type": "text", "text": "晴"}]},
         {"id": 5, "role": "user", "content": "谢谢"}]
    seen = {}

    async def compactor(hist, summary, room, measure, prompt=None, raw=None):
        seen["projected"], seen["raw"] = str(hist[2]["content"]), str(raw[2]["content"])
        return None

    async def stream(**kw):
        yield {"type": "final", "content": [{"type": "text", "text": "好"}], "stop_reason": "end_turn"}

    async def go():
        return [e async for e in run_loop(h, model_stream=stream, compactor=compactor, cleared_until=3,
                                          limits={"trigger": 0, "target": 10**9})]
    asyncio.run(go())
    assert "[已清理" in seen["projected"] and REF not in seen["projected"]
    assert REF in seen["raw"]
