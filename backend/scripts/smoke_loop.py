"""真实模型冒烟测试：uv run python scripts/smoke_loop.py "你的问题"

读 backend/.env 里的模型配置，打印循环产生的每个事件。
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from app.agent_loop import run_loop  # noqa: E402  需在加载 .env 之后导入


async def main(question: str) -> None:
    async for ev in run_loop([{"role": "user", "content": question}]):
        if ev["type"] == "text_delta":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "message":
            kinds = [b.get("type") for b in ev["message"]["content"]]
            print(f"\n  [message 存库] role={ev['message']['role']} blocks={kinds}")
        else:
            print(f"\n  [{ev['type']}] " + json.dumps({k: v for k, v in ev.items() if k != "type"}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "北京和上海今天天气怎么样？两地温差是多少度？"))
