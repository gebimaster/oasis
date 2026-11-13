"""O9 第一步：大结果在写库时统一截断一次、全文另存、按 ref 回取（只能读本人的）。"""

import asyncio

import pytest

from app import db, tools
from app.agent_loop import run_loop
from test_db import _conversation, with_fresh_db

BIG = "".join(f"第{i:05d}段：这是一段很长的中文输出。" for i in range(2500))  # 约 5 万字


@pytest.fixture
def big_tool():
    async def big(args):
        return BIG
    tools.register(tools.Tool(name="big", description="测试用", input_schema={"type": "object"}, handler=big,
                              permission=tools.PUBLIC, risk=tools.LOW))
    yield
    tools.unregister("big")


def call(name="big", args=None, id_="t1"):
    return tools.prepare({"id": id_, "name": name, "input": args or {}}, check_rules=False)


def test_without_identity_only_truncates(big_tool):
    content, is_error = asyncio.run(tools.run(call()))
    assert not is_error and len(content) < tools.OUTPUT_LIMIT
    assert content.startswith(BIG[:tools.OUTPUT_HEAD]) and content.endswith(BIG[-tools.OUTPUT_TAIL:])
    assert "全文没有保存" in content and "out_" not in content


def test_small_results_are_untouched():
    assert asyncio.run(tools.shrink("x", "短")) == "短"
    assert asyncio.run(tools.shrink("x", "a" * tools.OUTPUT_LIMIT)) == "a" * tools.OUTPUT_LIMIT  # 正好在线上不截


def test_preview_is_deterministic():
    """同一份全文永远得到同一段预览：历史里这条结果的字节不随轮次变，前缀缓存才可能命中。"""
    assert tools.output_preview(BIG, "out_0123456789abcdef") == tools.output_preview(BIG, "out_0123456789abcdef")
    assert tools.output_ref(None, None, "t1", BIG) == tools.output_ref(None, None, "t1", BIG)
    assert tools.output_ref(None, None, "t1", BIG) != tools.output_ref(None, None, "t2", BIG)
    assert tools.output_ref("u1", None, "", BIG) != tools.output_ref("u2", None, "", BIG)  # 复核⑨：按用户区分


def test_saved_full_text_can_be_read_back_only_by_its_owner(big_tool):
    async def body():
        uid, cid = await _conversation()
        other = await db.create_user("bob", "token-bob")
        tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, permissions=frozenset()))
        content, _ = await tools.run(call())
        ref = content.split("ref=")[1].split(" ")[0]
        assert ref.startswith("out_") and tools.READ_TOOL in content
        # 回取指定段落
        got, err = await tools.run(call(tools.READ_TOOL, {"ref": ref, "offset": 30000, "length": 100}, "t2"))
        assert not err and got.endswith(BIG[30000:30100]) and f"共 {len(BIG)} 字" in got
        # 越界的 offset 不报错，返回空段
        got, err = await tools.run(call(tools.READ_TOOL, {"ref": ref, "offset": 10**7}, "t3"))
        assert not err and got.endswith("\n")
        # 别人的 ref：和不存在的说同一句话
        tools.current.set(tools.ToolContext(user_id=other, permissions=frozenset()))
        theirs, err = await tools.run(call(tools.READ_TOOL, {"ref": ref}, "t4"))
        missing, err2 = await tools.run(call(tools.READ_TOOL, {"ref": "out_0000000000000000"}, "t5"))
        assert err and err2 and theirs.replace(ref, "X") == missing.replace("out_0000000000000000", "X")
        # 同一结果再写一次是幂等的（ON CONFLICT DO NOTHING），不报错
        tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, permissions=frozenset()))
        assert (await tools.run(call()))[0] == content
    with_fresh_db(body)


def test_history_keeps_the_same_truncated_bytes_across_turns(big_tool):
    """截断发生在执行时（写库前）：第 2、3 轮请求里这条结果的字节完全一样。"""
    seen = []

    async def stream(**kw):
        seen.append(kw["messages"])
        n = len(seen)
        if n == 1:
            content, stop = [{"type": "tool_use", "id": "b1", "name": "big", "input": {}}], "tool_use"
        elif n == 2:
            content, stop = [{"type": "tool_use", "id": "b2", "name": "get_weather", "input": {"city": "北京"}}], "tool_use"
        else:
            content, stop = [{"type": "text", "text": "好"}], "end_turn"
        yield {"type": "final", "content": content, "stop_reason": stop}

    async def go():
        return [e async for e in run_loop([{"role": "user", "content": "查"}], model_stream=stream)]
    events = asyncio.run(go())
    result = lambda msgs: next(b["content"] for m in msgs if isinstance(m["content"], list)
                               for b in m["content"] if b.get("tool_use_id") == "b1")
    assert result(seen[1]) == result(seen[2]) and len(result(seen[1])) < tools.OUTPUT_LIMIT
    assert next(e for e in events if e["type"] == "tool_end" and e["tool_use_id"] == "b1")["content"] == result(seen[1])



def test_saving_full_text_does_not_count_against_the_tool_timeout(monkeypatch):
    """复核⑥：handler 已经做完，存全文慢了不能判成超时（高风险工具会被记成结果未知、结果被丢掉）。"""
    import uuid

    async def slow_save(*a):
        await asyncio.sleep(0.3)
    monkeypatch.setattr(db, "save_tool_output", slow_save)

    async def big(args):
        return BIG
    t = tools.Tool(name="w", description="", input_schema={"type": "object"}, handler=big, permission=tools.PUBLIC,
                   risk=tools.HIGH, timeout=0.1)
    tools.current.set(tools.ToolContext(user_id=uuid.uuid4(), permissions=frozenset()))
    content, is_error, timed_out, _ = asyncio.run(tools.invoke(t, {}))
    assert not is_error and not timed_out and "ref=out_" in content


def test_huge_error_messages_are_truncated_too():
    """复核⑧：异常里带着整个响应体也要截，不然 2MB 从报错这条路进了历史。"""
    async def boom(args):
        raise RuntimeError(BIG)
    t = tools.Tool(name="b", description="", input_schema={"type": "object"}, handler=boom, permission=tools.PUBLIC,
                   risk=tools.LOW)
    content, is_error, *_ = asyncio.run(tools.invoke(t, {}))
    assert is_error and len(content) < tools.OUTPUT_LIMIT and content.startswith(tools.ERROR_PREFIX)
