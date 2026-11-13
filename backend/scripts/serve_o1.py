"""O1 双实例实测用：同一个库（oasis_o1）、不同端口，模型换成假的慢模型，不花钱。

用法：uv run python scripts/serve_o1.py 8011
    FAKE_SECONDS：假模型吐字多久（默认 60 秒，每 0.5 秒一个字）
    FAKE_MODE=stubborn：调一个吞取消的工具；FAKE_MODE=real：不替换，用 .env 里的真实模型（花钱）
"""

import asyncio
import os
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DATABASE_URL", "postgresql://oasis:oasis@127.0.0.1:5433/oasis_o1")
os.environ.setdefault("SANDBOX", "0")    # 不连 Cube
os.environ.setdefault("WEB_TOOLS", "0")  # 不花搜索费

from app import llm  # noqa: E402

FAKE_SECONDS = float(os.getenv("FAKE_SECONDS", "60"))


async def slow_stream(**_):
    """每 0.5 秒吐一个字，模拟一个很长的回答。"""
    text = ""
    for i in range(int(FAKE_SECONDS * 2)):
        await asyncio.sleep(0.5)
        ch = str(i % 10)
        text += ch
        yield {"type": "text_delta", "text": ch}
    yield {"type": "final", "content": [{"type": "text", "text": text}], "stop_reason": "end_turn"}


async def stubborn_stream(messages, **_):
    """第 2 步：第一轮调 stubborn 工具，拿到结果后收尾。"""
    if any(b.get("type") == "tool_result" for b in messages[-1]["content"]):
        yield {"type": "final", "content": [{"type": "text", "text": "好了"}], "stop_reason": "end_turn"}
        return
    yield {"type": "text_delta", "text": "我来执行"}
    yield {"type": "final", "stop_reason": "tool_use", "content": [
        {"type": "text", "text": "我来执行"}, {"type": "tool_use", "id": "s1", "name": "stubborn", "input": {}}]}


async def stubborn(_args):
    """吞掉所有取消、睡满 STUBBORN_SECONDS 才返回：模拟写得不好的第三方 SDK。"""
    end = asyncio.get_running_loop().time() + float(os.getenv("STUBBORN_SECONDS", "25"))
    while (left := end - asyncio.get_running_loop().time()) > 0:
        try:
            await asyncio.sleep(left)
        except asyncio.CancelledError:
            print("stubborn: swallowed a cancel", flush=True)
    print("stubborn: woke up", flush=True)
    return "做完了"


if os.getenv("FAKE_MODE") == "stubborn":
    from app import tools  # noqa: E402
    tools.register(tools.Tool(name="stubborn", description="test", input_schema={"type": "object", "properties": {}},
                              handler=stubborn, permission=tools.PUBLIC, risk=tools.LOW, timeout=120,
                              record_execution=True))
    llm.stream_message = stubborn_stream
elif os.getenv("FAKE_MODE") != "real":
    llm.stream_message = slow_stream  # run_loop 调用时才取 llm.stream_message，替换模块属性即可

if __name__ == "__main__":
    port = int(sys.argv[1])
    os.environ["INSTANCE_PORT"] = str(port)
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, app_dir=str(Path(__file__).parent.parent))
