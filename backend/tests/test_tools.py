"""工具层验证：不依赖数据库和模型，直接跑 `uv run pytest`。"""

import asyncio
import time

from app import tools


def _use(id, name, input):
    # tool_use 块的 dict 形式（llm 层已把 SDK 对象转成 dict）
    return {"type": "tool_use", "id": id, "name": name, "input": input}


def test_list_tools_is_api_format():
    names = {t["name"] for t in tools.list_tools()}
    assert {"calculate", "call_tool"} <= names  # recall 等在各自模块 import 后才注册，取决于导入顺序
    assert not {"get_weather", "get_stock"} & names  # M4：延迟工具不进 tools 数组
    for t in tools.list_tools():
        assert set(t) == {"name", "description", "input_schema", "eager_input_streaming"}


def test_execute_ok():
    assert asyncio.run(tools.execute("calculate", {"a": 6, "b": 7, "op": "mul"})) == ("42", False)


def test_unknown_tool_is_error_not_exception():
    content, is_error = asyncio.run(tools.execute("rm_rf", {}))
    assert is_error and "unknown tool" in content


def test_invalid_input_is_error():
    # 模拟流式入参被截断：缺了必填字段 op
    content, is_error = asyncio.run(tools.execute("calculate", {"a": 1, "b": 2}))
    assert is_error and "INVALID_INPUT" in content


def test_handler_exception_is_error():
    content, is_error = asyncio.run(tools.execute("calculate", {"a": 1, "b": 0, "op": "div"}))
    assert is_error and "ZeroDivisionError" in content


def test_execute_all_parallel_and_ordered():
    uses = [
        _use("t1", "get_weather", {"city": "北京"}),
        _use("t2", "get_weather", {"city": "火星"}),  # 这个会失败
        _use("t3", "get_weather", {"city": "上海"}),
    ]
    start = time.perf_counter()
    results = asyncio.run(tools.execute_all([tools.prepare(u) for u in uses]))
    elapsed = time.perf_counter() - start

    # 3 个各睡 0.5s，并行应远小于串行的 1.5s
    assert elapsed < 1.0, f"没有并行执行，耗时 {elapsed:.2f}s"
    # 顺序与 tool_use 一致，每个都有配对的 tool_result，失败的那个标 is_error
    assert [r["tool_use_id"] for r in results] == ["t1", "t2", "t3"]
    assert [r["is_error"] for r in results] == [False, True, False]
    assert all(r["type"] == "tool_result" for r in results)


def test_every_error_has_text_prefix():
    # 不依赖 is_error 字段，光看文字也能认出失败（兼容会忽略 is_error 的接口）
    cases = [
        ("rm_rf", {}),
        ("calculate", {"a": 1, "b": 2}),
        ("calculate", {"a": 1, "b": 0, "op": "div"}),
    ]
    for name, args in cases:
        content, is_error = asyncio.run(tools.execute(name, args))
        assert is_error and content.startswith(tools.ERROR_PREFIX), content


# ---------------- O6 第二步：依赖不可用的结构化错误码 ----------------

def test_unavailable_sets_error_code_others_dont():
    """工具主动抛 tools.Unavailable：文字照样给模型看（不带异常类名），错误码另走 Call.error_code，不进 tool_result 块。
    普通异常（含 ConnectionError、TimeoutError）不带码：只有工具自己确认「服务挂了」才算。"""
    from tool_codes import execute_with_code

    async def down(args):
        raise tools.Unavailable("UNAVAILABLE 服务暂时不可用")

    async def conn(args):
        raise ConnectionError("连不上")

    async def slow(args):
        raise TimeoutError("慢")
    for name, h in (("t_down", down), ("t_conn", conn), ("t_slow", slow)):
        tools.register(tools.Tool(name=name, description="x", input_schema={"type": "object"}, handler=h,
                                  permission=tools.PUBLIC, risk=tools.LOW))
    try:
        async def go():
            return [await execute_with_code(n, {}) for n in ("t_down", "t_conn", "t_slow")]
        (c1, e1, k1), (c2, e2, k2), (c3, e3, k3) = asyncio.run(go())
        assert e1 and c1 == f"{tools.ERROR_PREFIX} UNAVAILABLE 服务暂时不可用" and k1 == tools.UNAVAILABLE
        assert e2 and "ConnectionError" in c2 and k2 is None
        assert e3 and k3 is None
        blocks = asyncio.run(tools.execute_all([tools.prepare({"id": "x", "name": "t_down", "input": {}})]))
        assert set(blocks[0]) == {"type", "tool_use_id", "content", "is_error"}  # 码不进发给 API 的块
    finally:
        for n in ("t_down", "t_conn", "t_slow"):
            tools.unregister(n)


def test_air_quality_demo_is_always_unavailable():
    """O6 评测用的演示工具：服务固定挂着；权限默认谁都没有，评测题单独给。"""
    from tool_codes import execute_with_code
    tools.current.set(tools.ToolContext(permissions=frozenset({"air:read"})))
    content, is_error, code = asyncio.run(execute_with_code("get_air_quality", {"city": "北京"}))
    assert is_error and code == tools.UNAVAILABLE and "不可用" in content
    tools.current.set(tools.ToolContext())
    assert "get_air_quality" not in tools.deferred_catalog()


def test_unavailable_can_also_mark_outcome_unknown():
    """O6 复核乙（选 b）：Unavailable(outcome_unknown=True) 既带 UNAVAILABLE 码（计入熔断），又让 timed_out=True
    （有副作用 / 要记录的工具，执行记录写 outcome_unknown）。默认是 False：确定没执行。"""
    async def gone(args):
        raise tools.Unavailable("UNAVAILABLE 断开了", outcome_unknown=True)

    async def down(args):
        raise tools.Unavailable("UNAVAILABLE 连不上")
    t1 = tools.Tool(name="g", description="", input_schema={"type": "object"}, handler=gone,
                    permission=tools.PUBLIC, risk=tools.LOW)
    t2 = tools.Tool(name="d", description="", input_schema={"type": "object"}, handler=down,
                    permission=tools.PUBLIC, risk=tools.LOW)
    assert asyncio.run(tools.invoke(t1, {}))[1:] == (True, True, tools.UNAVAILABLE)
    assert asyncio.run(tools.invoke(t2, {}))[1:] == (True, False, tools.UNAVAILABLE)
