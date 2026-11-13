"""M4 演示用 MCP server（stdio）。由 oasis 后端作为子进程拉起，不要直接对外。

mcp 2.x：FastMCP 已改名为 mcp.server.mcpserver.MCPServer（见 SDK 的 server/fastmcp.py）。
"""

import asyncio

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # 故意抛的业务错误：消息会传给客户端（其他异常只回一句通用错误）

server = MCPServer("demo")

_RATES = {("km", "mile"): 0.621371, ("mile", "km"): 1.609344, ("kg", "lb"): 2.204623, ("lb", "kg"): 0.453592}


@server.tool()
def echo_upper(text: str) -> str:
    """把文字转成大写（演示用）。"""
    return text.upper()


@server.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算，支持 km↔mile、kg↔lb。"""
    rate = _RATES.get((from_unit, to_unit))
    if rate is None:
        raise ToolError(f"不支持 {from_unit}→{to_unit}，可选：{sorted(_RATES)}")
    return f"{value} {from_unit} = {round(value * rate, 4)} {to_unit}"


@server.tool()
async def slow_task(seconds: float) -> str:
    """故意等待指定秒数再返回，用来验证调用超时。"""
    await asyncio.sleep(seconds)
    return f"睡了 {seconds} 秒"


if __name__ == "__main__":
    server.run("stdio")
