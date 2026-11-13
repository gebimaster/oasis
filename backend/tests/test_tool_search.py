"""M4 第 1 步：常驻/延迟两档、tool_search（select 精确加载 / 关键词只出候选）、call_tool 代理、压缩后补回定义。"""

import asyncio
import json
from datetime import datetime, timezone

from app import db, tool_search, tools  # noqa: F401  import tool_search 即注册
from app.context import LOADED_NOTE, build_request
from conftest import ALL_PERMISSIONS
from test_db import with_fresh_db


def run(coro_fn, ctx=None):
    """在一个设置了 ToolContext 的上下文里跑，模拟 runner。"""
    async def main():
        tools.current.set(ctx or tools.ToolContext(permissions=ALL_PERMISSIONS))
        return await coro_fn()
    return asyncio.run(main())


def search(query, ctx=None):
    return run(lambda: tools.execute("tool_search", {"query": query}), ctx)


def test_tools_array_fixed_and_catalog_lists_deferred():
    before = tools.list_tools()
    names = [t["name"] for t in before]
    assert "tool_search" in names and names[-1] == "call_tool"
    catalog = tools.deferred_catalog()
    assert "get_weather" in catalog and "get_stock" in catalog and "calculate" not in catalog
    ctx = tools.ToolContext(permissions=ALL_PERMISSIONS)
    search("select:get_stock", ctx)
    assert tools.list_tools() == before          # 加载后 tools 数组一个字节都不变：前缀缓存不受影响
    assert tools.deferred_catalog() == catalog   # 名录也不变


def test_select_loads_definitions_and_records():
    ctx = tools.ToolContext(permissions=ALL_PERMISSIONS)
    out, is_error = search("select:get_stock, get_weather", ctx)
    assert not is_error and ctx.loaded == ["get_stock", "get_weather"]
    defs = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
    assert [d["name"] for d in defs] == ["get_stock", "get_weather"]
    assert defs[0]["input_schema"]["required"] == ["symbol"]
    search("select:get_stock", ctx)              # 重复加载不重复记录
    assert ctx.loaded == ["get_stock", "get_weather"]


def test_select_needs_exact_deferred_name():
    for q in ("select:get_sto", "select:calculate", "select:"):  # 半截名、常驻工具、空
        ctx = tools.ToolContext(permissions=ALL_PERMISSIONS)
        out, is_error = search(q, ctx)
        assert is_error and "no_match" in out and ctx.loaded == []


def test_keyword_only_returns_candidates():
    ctx = tools.ToolContext(permissions=ALL_PERMISSIONS)
    out, is_error = search("股票 价格", ctx)
    assert not is_error and "get_stock" in out and "get_weather" not in out
    assert ctx.loaded == []                      # 关键词不加载
    out, _ = search("股价 股票 stock price", ctx)  # 中英混搜（真实模型这样搜过）：至少中一个词就算候选
    assert "get_stock" in out and ctx.loaded == []
    out, _ = search("火车票", ctx)
    assert "没有和" in out


def test_call_tool_unwraps_and_validates_target():
    ok = run(lambda: tools.execute("call_tool", {"tool_name": "get_stock", "arguments": {"symbol": "aapl"}}))
    assert ok == ("AAPL 最新价 231.5 美元（示例数据）", False)
    # 参数按目标工具的 schema 校验
    out, is_error = run(lambda: tools.execute("call_tool", {"tool_name": "get_stock", "arguments": {"code": "AAPL"}}))
    assert is_error and "INVALID_INPUT" in out
    # 代理本身的格式错、嵌套调用、未知工具
    for bad in ({"tool_name": "get_stock"}, {"tool_name": "call_tool", "arguments": {}},
                {"tool_name": "rm_rf", "arguments": {}}):
        out, is_error = run(lambda: tools.execute("call_tool", bad))
        assert is_error and out.startswith(tools.ERROR_PREFIX), out


def test_deferred_callable_by_real_name_without_loading():
    # DeepSeek 实测允许模型直接用真名调用不在 tools 数组里的工具；放行，但照样做参数校验
    assert run(lambda: tools.execute("get_stock", {"symbol": "TSLA"}))[1] is False


def test_loaded_defs_follow_summary_only():
    now = datetime(2026, 10, 29, tzinfo=timezone.utc)
    defs = tools.definitions(["get_stock"])
    args = dict(user_name="小刘", history=[{"role": "user", "content": "hi"}], now=now, final_turn=False,
                catalog=tools.deferred_catalog(), loaded_defs=defs)
    _, msgs = build_request(summary=None, **args)
    assert "<deferred_tools>" in msgs[0]["content"]               # 名录在第 2 层
    assert not any(LOADED_NOTE in str(m["content"]) for m in msgs)  # 没压缩过：定义还在历史的 tool_result 里
    _, msgs = build_request(summary="之前查过股票", **args)
    assert msgs[1]["content"].startswith("<summary>") and LOADED_NOTE in msgs[1]["content"]
    assert '"name": "get_stock"' in msgs[1]["content"]


def test_loaded_tools_persisted():
    async def body():
        uid = await db.create_user("alice", "tok-a")
        cid = await db.create_conversation(uid)
        assert await db.get_loaded_tools(cid) == []
        tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, permissions=ALL_PERMISSIONS))
        await tools.execute("tool_search", {"query": "select:get_weather"})
        await tools.execute("tool_search", {"query": "select:get_stock"})
        assert await db.get_loaded_tools(cid) == ["get_weather", "get_stock"]  # 保持加载顺序
    with_fresh_db(body)
