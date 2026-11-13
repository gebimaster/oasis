"""M4 第 2 步：按用户权限过滤。无权限的工具在每个暴露点都看不到，调用时再查一次，回话和「不存在」一样。"""

import asyncio

import pytest

from app import db, tool_search, tools  # noqa: F401
from app.agent_loop import run_loop
from test_agent_loop import fake_model
from test_db import with_fresh_db

GUEST = frozenset({"weather:read"})  # 没有 stock:read


def as_user(perms, fn):
    async def main():
        tools.current.set(tools.ToolContext(permissions=perms))
        return await fn()
    return asyncio.run(main())


def test_hidden_from_every_exposure_point():
    def check():
        names = [t["name"] for t in tools.list_tools()]
        return names, tools.deferred_catalog(), tools.definitions(["get_stock", "get_weather"])

    names, catalog, defs = as_user(GUEST, lambda: asyncio.sleep(0, check()))
    assert "call_tool" in names                                   # 还有能用的延迟工具 get_weather
    assert "get_weather" in catalog and "get_stock" not in catalog
    assert [d["name"] for d in defs] == ["get_weather"]           # 权限收回后，摘要后补回的定义也不带它


def test_tool_search_cannot_reach_hidden():
    ctx = tools.ToolContext(permissions=GUEST)
    out, is_error = as_user(GUEST, lambda: tools.execute("tool_search", {"query": "股票"}))
    assert "get_stock" not in out                                 # 关键词搜不到
    async def select():
        tools.current.set(ctx)
        return await tools.execute("tool_search", {"query": "select:get_stock"})
    out, is_error = asyncio.run(select())
    assert is_error and "no_match" in out and ctx.loaded == []    # select 精确名也绕不过


def test_call_hidden_looks_like_unknown():
    unknown = as_user(GUEST, lambda: tools.execute("no_such_tool", {}))
    for name, args in [("get_stock", {"symbol": "AAPL"}),                                       # 真名直调
                       ("call_tool", {"tool_name": "get_stock", "arguments": {"symbol": "AAPL"}})]:  # 经代理
        out, is_error = as_user(GUEST, lambda: tools.execute(name, args))
        assert is_error and out == unknown[0].replace("no_such_tool", "get_stock")  # 和「不存在」一字不差
    assert as_user(frozenset({"stock:read"}), lambda: tools.execute("get_stock", {"symbol": "AAPL"}))[1] is False


def test_no_permissions_sees_only_public():
    names = as_user(frozenset(), lambda: asyncio.sleep(0, [t["name"] for t in tools.list_tools()]))
    assert "calculate" in names and "call_tool" not in names      # 一个延迟工具都用不了，代理也不给
    assert as_user(frozenset(), lambda: asyncio.sleep(0, tools.deferred_catalog())) == ""


def test_register_requires_permission():
    async def h(_):
        return ""
    with pytest.raises(TypeError):                                # 漏写 permission：构造就失败
        tools.Tool(name="x", description="", input_schema={}, handler=h)
    with pytest.raises(ValueError):
        tools.register(tools.Tool(name="x", description="", input_schema={}, handler=h, permission="", risk="low"))


def test_loop_survives_model_calling_hidden_tool():
    """模型硬调没暴露的工具：拿到 is_error 的 tool_result，run 正常结束，不崩。"""
    call = {"type": "tool_use", "id": "t1", "name": "call_tool",
            "input": {"tool_name": "get_stock", "arguments": {"symbol": "AAPL"}}}
    stream, calls = fake_model([([call], "tool_use"), ([{"type": "text", "text": "查不到"}], "end_turn")])

    async def collect():
        tools.current.set(tools.ToolContext(permissions=GUEST))
        return [ev async for ev in run_loop([{"role": "user", "content": "AAPL?"}], model_stream=stream)]
    events = asyncio.run(collect())
    end = next(e for e in events if e["type"] == "tool_end")
    assert end["is_error"] and "unknown tool 'get_stock'" in end["content"]
    assert events[-1] == {"type": "run_end", "status": "completed", "stop_reason": "end_turn"}
    assert "get_stock" not in str(calls[0]["messages"][0])       # 发给模型的名录里本来就没有它


def test_permissions_roundtrip():
    async def body():
        uid = await db.create_user("guest", "tok-g")
        await db.grant_permissions(uid, ["weather:read"])
        await db.grant_permissions(uid, ["weather:read"])           # 幂等
        assert await db.get_permissions(uid) == ["weather:read"]
    with_fresh_db(body)
