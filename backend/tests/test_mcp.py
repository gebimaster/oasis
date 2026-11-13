"""M4 第 3 步：MCP。真起 demo_server 子进程，验证发现/注册、调用、业务错误、参数校验、权限、超时、进程被杀后的行为和自动重连。"""

import asyncio
import os
import subprocess
import types

from app import mcp_client, tools
from tool_codes import execute_with_code

PERMS = frozenset({"mcp:demo:echo_upper", "mcp:demo:unit_convert", "mcp:demo:slow_task"})
CONVERT = "mcp__demo__unit_convert"


def test_normalize_and_render():
    assert mcp_client.normalize("Get-Stock.Price!") == "get_stock_price"
    long = types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="x" * 5000),
                                          types.SimpleNamespace(type="image")])
    out = mcp_client.render(long)
    assert out.startswith(mcp_client.UNTRUSTED)                     # 外部结果带不可信前缀
    assert "已截断" in out and len(out) < mcp_client.MAX_RESULT_CHARS + 200


def _kill_server_child():
    """找到本进程拉起的 demo_server.py 子进程并强杀，模拟 server 崩溃。"""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq %d -and "
          "$_.CommandLine -like '*demo_server.py*' } | Select-Object -ExpandProperty ProcessId" % os.getpid())
    pids = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.split()
    assert pids, "没找到 demo_server 子进程"
    for pid in pids:
        subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)


def test_mcp_end_to_end_and_crash_recovery():
    async def main():
        mcp_client.PING_INTERVAL, mcp_client.CALL_TIMEOUT = 0.5, 1  # 测试里调快
        tools.current.set(tools.ToolContext(permissions=PERMS))
        conn = mcp_client.start_all()["demo"]
        try:
            await asyncio.wait_for(conn.connected.wait(), 20)
            assert conn.registered == {"mcp__demo__echo_upper", CONVERT, "mcp__demo__slow_task"}
            # 第二轮复核②：钉住真实 demo 名单。名单写空、拼错（如 unit-convert），对应工具就会回到 HIGH
            risks = {n: tools._REGISTRY[n].risk for n in conn.registered}
            assert risks == {"mcp__demo__echo_upper": tools.LOW, CONVERT: tools.LOW, "mcp__demo__slow_task": tools.HIGH}
            assert CONVERT in tools.deferred_catalog()

            out, err = await tools.execute("call_tool", {"tool_name": CONVERT,
                                                         "arguments": {"value": 10, "from_unit": "km", "to_unit": "mile"}})
            assert not err and "6.2137 mile" in out and out.startswith(mcp_client.UNTRUSTED)
            out, err = await tools.execute(CONVERT, {"value": 1, "from_unit": "x", "to_unit": "y"})
            assert err and "不支持 x→y" in out                       # server 的业务错误透传给模型
            out, err = await tools.execute(CONVERT, {"value": "十"})
            assert err and "INVALID_INPUT" in out                     # 我们先按远端 schema 校验，不发网络请求
            out, err = await tools.execute("mcp__demo__slow_task", {"seconds": 3})
            assert err and "未返回" in out                             # 调用超时

            tools.current.set(tools.ToolContext(permissions=frozenset({"mcp:demo:echo_upper"})))  # guest
            assert CONVERT not in tools.deferred_catalog()
            out, err = await tools.execute(CONVERT, {"value": 1, "from_unit": "km", "to_unit": "mile"})
            assert err and "unknown tool" in out
            tools.current.set(tools.ToolContext(permissions=PERMS))

            await asyncio.to_thread(_kill_server_child)
            args = {"value": 1, "from_unit": "kg", "to_unit": "lb"}
            out, err, code = await execute_with_code(CONVERT, args)
            # 挂了：is_error，不崩。刚杀掉、客户端还没发现时拿到的是「调用中断开」（CONNECTION_CLOSED），看时序
            assert err and ("暂时不可用" in out or "调用失败" in out or "断开" in out), out
            assert code == tools.UNAVAILABLE  # O6：三种都算依赖不可用（断开那种同时标结果未知，复核乙）
            assert CONVERT in tools.deferred_catalog()                      # 名录不删：第 2 层缓存不受影响

            for _ in range(80):                                              # 自动重连（ping 发现 + 1s 退避）
                if conn.status == "down":
                    break
                await asyncio.sleep(0.1)
            await asyncio.wait_for(conn.connected.wait(), 20)
            out, err = await tools.execute(CONVERT, args)
            assert not err and "2.2046 lb" in out
        finally:
            await mcp_client.stop_all()
            for name in list(conn.registered):
                tools.unregister(name)
    asyncio.run(main())


def test_unavailable_codes_without_real_server():
    """O6：不起子进程，逐条确认各种失败的错误码。连不上、服务端内部错误（-32603 等其余 MCPError）、调用中断开 → UNAVAILABLE；
    调用超时 → 结果未知，不算不可用；参数错误、方法不存在、请求无效、解析错误 → 调用方的问题，不算（第三轮复核 F 改准）。"""
    from mcp import MCPError
    from mcp.types import REQUEST_TIMEOUT
    conn = mcp_client.ServerConn("fake", None)
    handler = conn._handler("t")

    class Session:
        def __init__(self, code):
            self.code = code

        async def call_tool(self, *a, **kw):
            raise MCPError(self.code, "boom")

    async def go(session, status):
        conn.session, conn.status = session, status
        tool = tools.Tool(name="m", description="", input_schema={"type": "object"}, handler=handler,
                          permission=tools.PUBLIC, risk=tools.LOW)
        return await tools.invoke(tool, {})
    down = asyncio.run(go(None, "down"))
    failed = asyncio.run(go(Session(-32603), "up"))
    timed_out = asyncio.run(go(Session(REQUEST_TIMEOUT), "up"))
    assert down[3] == tools.UNAVAILABLE and "暂时不可用" in down[0]
    assert failed[3] == tools.UNAVAILABLE and "调用失败" in failed[0]
    assert timed_out[3] is None and timed_out[2]  # 超时：timed_out=True，外部可能还在执行
    # 复核乙（你选 b）：调用中连接断了 = 服务挂了，同时这次结果未知：计入熔断，timed_out=True 让执行记录写 outcome_unknown
    from mcp.types import CONNECTION_CLOSED, INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR
    closed = asyncio.run(go(Session(CONNECTION_CLOSED), "up"))
    assert closed[3] == tools.UNAVAILABLE and closed[2] and "可能已经执行" in closed[0]
    # 复核乙：参数错误、方法不存在是这次调用本身的问题，不算依赖不可用，也不是结果未知
    for code in (INVALID_PARAMS, METHOD_NOT_FOUND, INVALID_REQUEST, PARSE_ERROR):
        bad = asyncio.run(go(Session(code), "up"))
        assert bad[1] and bad[3] is None and not bad[2], (code, bad)


def _fake_remote(name, read_only_hint=None):
    ann = types.SimpleNamespace(read_only_hint=read_only_hint) if read_only_hint is not None else None
    return types.SimpleNamespace(name=name, description="x", inputSchema={"type": "object"},
                                 input_schema={"type": "object"}, annotations=ann)


def test_read_only_list_is_ours_not_the_servers(monkeypatch):
    """O6 复核④（你选的 D2）：只读名单写在 oasis 自己的配置里（READ_ONLY），server 自报的 read_only_hint 一概不看。
    名单里的工具 risk=LOW，不走审批；其余照旧 HIGH。"""
    monkeypatch.setitem(mcp_client.READ_ONLY, "fakesrv", frozenset({"lookup"}))
    conn = mcp_client.ServerConn("fakesrv", None)
    try:
        conn._register([_fake_remote("lookup"), _fake_remote("liar", read_only_hint=True), _fake_remote("write")])
        reg = tools._REGISTRY
        assert reg["mcp__fakesrv__lookup"].risk == tools.LOW
        assert reg["mcp__fakesrv__liar"].risk == tools.HIGH  # 自称只读也不信
        assert reg["mcp__fakesrv__write"].risk == tools.HIGH
    finally:
        conn._register([])
    assert "demo" in mcp_client.READ_ONLY and "slow_task" not in mcp_client.READ_ONLY["demo"]


def test_mcp_down_trips_breaker_without_rules(monkeypatch):
    """O6 复核④：没有任何用户规则时，只读 MCP 工具挂了也能被熔断数到（原来 HIGH → 审批挂起，走不到计数）。"""
    from app.agent_loop import run_loop
    from test_agent_loop import TEXT, fake_model
    monkeypatch.setitem(mcp_client.READ_ONLY, "fakesrv", frozenset({"lookup"}))
    conn = mcp_client.ServerConn("fakesrv", None)  # 没连上：status 是 down
    conn._register([_fake_remote("lookup")])
    try:
        tools.current.set(tools.ToolContext(permissions=frozenset({"mcp:fakesrv:lookup"})))
        call = lambda i: {"type": "tool_use", "id": f"m{i}", "name": "mcp__fakesrv__lookup", "input": {"q": i}}
        stream, calls = fake_model([([call(1)], "tool_use"), ([call(2)], "tool_use"), TEXT])

        async def go():
            return [ev async for ev in run_loop([{"role": "user", "content": "查一下"}], model_stream=stream)]
        events = asyncio.run(go())
        assert events[-1]["stop_reason"] == "tool_unavailable" and len(calls) == 3
    finally:
        conn._register([])
