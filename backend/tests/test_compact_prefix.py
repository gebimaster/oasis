"""P1-C1：摘要请求复用正常请求的前缀（同 system、同 tools、tool_choice auto），只把运行时尾部换成压缩指令。"""

import asyncio
from datetime import datetime, timezone

from app import compaction, context, llm, tools
from app.agent_loop import run_loop

NOW = datetime(2026, 11, 12, 10, 0, tzinfo=timezone.utc)
Q = {"role": "user", "content": "帮我查北京天气"}
CALL = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "北京"}}]}
RESULT = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "北京 晴 12℃"}]}
ANSWER = {"role": "assistant", "content": [{"type": "text", "text": "北京今天晴，12℃。"}]}


def build(history, instruction=None):
    return context.build_request(user_name="小王", summary="旧摘要", history=history, now=NOW, final_turn=False,
                                 instruction=instruction)


def test_instruction_replaces_tail_after_tool_result():
    """形状 a：最后是 tool_result，指令接在同一条 user 里；前面逐字相同，只有尾部换掉。"""
    sys_main, main = build([Q, CALL, RESULT])
    sys_sum, summ = build([Q, CALL, RESULT], instruction="压缩")
    assert sys_sum == sys_main and summ[:-1] == main[:-1]
    assert summ[-1]["content"][:-1] == main[-1]["content"][:-1]          # tool_result 原样
    assert summ[-1]["content"][-1] == {"type": "text", "text": "压缩"}   # 运行时尾部（时间）换成了指令
    assert "当前时间" not in str(summ)


def test_instruction_after_assistant_starts_new_user_message():
    """形状 b：摘要源以回答结尾（按真提问切），指令新起一条 user，不能拼进 assistant。"""
    _, main = build([Q, ANSWER])
    _, summ = build([Q, ANSWER], instruction="压缩")
    assert summ[:-1] == main and summ[-1] == {"role": "user", "content": [{"type": "text", "text": "压缩"}]}


def fake_llm(monkeypatch, replies):
    """按顺序回放 (content, stop_reason)，记下每次请求。"""
    sent = []

    async def stream(**kw):
        sent.append(kw)
        content, stop = replies[len(sent) - 1]
        yield {"type": "final", "content": content, "stop_reason": stop, "usage": {}}
    monkeypatch.setattr(llm, "stream_message", stream)
    return sent


def prompt(s, h, instruction):
    system, msgs = build(h, instruction)
    return {"system": system, "messages": msgs, "tools": tools.list_tools()}


TEXT = [{"type": "thinking", "thinking": "想想"}, {"type": "text", "text": "摘要：用户在查北京天气"}]
TOOL = [{"type": "tool_use", "id": "x", "name": "recall", "input": {"query": "天气"}}]


def test_summary_request_keeps_tools_and_auto(monkeypatch):
    sent = fake_llm(monkeypatch, [(TEXT, "end_turn")])
    out = asyncio.run(compaction.summarize("旧摘要", [Q, CALL, RESULT], prompt=prompt))
    assert out == "摘要：用户在查北京天气"                                   # 只取正文，thinking 不进摘要
    main_sys, main = build([Q, CALL, RESULT])
    assert sent[0]["tool_choice"] == {"type": "auto"}                        # 不能是 none：会把工具从 prompt 里拿掉
    assert sent[0]["tools"] == tools.list_tools() and sent[0]["system"] == main_sys
    assert sent[0]["messages"][:-1] == main[:-1]


def test_tool_use_retries_once_with_stronger_note(monkeypatch):
    sent = fake_llm(monkeypatch, [(TOOL, "tool_use"), (TEXT, "end_turn")])
    out = asyncio.run(compaction.summarize(None, [Q, CALL, RESULT], prompt=prompt))
    assert out == "摘要：用户在查北京天气" and len(sent) == 2
    first, second = (s["messages"] for s in sent)
    assert first[:-1] == second[:-1] and first[-1]["content"][:-1] == second[-1]["content"][:-1]  # 前缀不变
    assert second[-1]["content"][-1]["text"] == compaction.compact_note(merge=False, retry=True)
    blocks = [b for m in second if isinstance(m["content"], list) for b in m["content"]]
    assert all(b.get("id") != "x" for b in blocks)                          # 被丢掉的 tool_use 没有混进重试请求


def test_tool_use_twice_falls_back_to_alone(monkeypatch):
    sent = fake_llm(monkeypatch, [(TOOL, "tool_use"), (TOOL, "tool_use")])
    alone = []

    async def complete_with_stop(**kw):
        alone.append(kw)
        return "退路摘要：北京天气", "end_turn"
    monkeypatch.setattr(llm, "complete_with_stop", complete_with_stop)
    assert asyncio.run(compaction.summarize(None, [Q, CALL, RESULT], prompt=prompt)) == "退路摘要：北京天气"
    assert len(sent) == 2 and len(alone) == 1


def test_loop_prompt_matches_its_main_request():
    """循环给 compactor 的 prompt 和它自己发出的正常请求同一前缀：同 system、同 tools，消息只差尾部。"""
    got = {}

    async def compactor(h, s, room, measure, prompt=None):
        got["req"] = prompt(s, h, "压缩")
        return None  # 不压，接着照发正常请求

    calls = []

    async def stream(**kw):
        calls.append(kw)
        yield {"type": "final", "content": ANSWER["content"], "stop_reason": "end_turn", "usage": {}}

    async def go():
        return [e async for e in run_loop([Q], model_stream=stream, compactor=compactor, summary="旧摘要",
                                          limits={"trigger": 0, "target": 10**9})]
    asyncio.run(go())
    req, main = got["req"], calls[0]
    assert req["system"] == main["system"] and req["tools"] == main["tools"]
    assert req["messages"][:-1] == main["messages"][:-1]
    assert req["messages"][-1]["content"][:-1] == main["messages"][-1]["content"][:-1]
    assert req["messages"][-1]["content"][-1]["text"] == "压缩"
