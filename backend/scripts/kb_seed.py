"""M6 知识库种子：把 kb_samples/<用户名>/*.md 灌进对应用户的库。.venv\\Scripts\\python scripts/kb_seed.py

- 用户按 token 从库里查（先启动过一次服务，lifespan 才会建 demo / guest 用户）。
- DATABASE_URL、KB_DIR 和服务读同一套环境变量：e2e 用独立库时，两边要设成一样，否则 user_id 对不上。
- 第一行「# 标题」当 title，其余是正文；doc_id 用文件名。ingest 是增量的，重复跑只改动过的块会重新 embedding。
"""

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND))
load_dotenv(BACKEND / ".env", override=False)  # 不覆盖命令行设的 DATABASE_URL / KB_DIR

from app import db  # noqa: E402
from app.kb import KnowledgeBase, SiliconFlowEmbedder, open_collection  # noqa: E402

TOKENS = {"demo": os.getenv("DEMO_TOKEN", "demo-token"), "guest": os.getenv("GUEST_TOKEN", "guest-token")}


def parse(path: Path) -> tuple[str, str]:
    first, _, body = path.read_text(encoding="utf-8").partition("\n")
    return (first.removeprefix("#").strip(), body.strip()) if first.startswith("#") else (path.stem, first + body)


async def main() -> None:
    await db.connect()
    try:
        kb = KnowledgeBase(open_collection(), SiliconFlowEmbedder())
        for name, token in TOKENS.items():
            uid = await db.user_id_by_token(token)
            if uid is None:  # 跳过而不是退出：已有的用户照常灌
                print(f"跳过 {name}：库里没有这个用户，先启动一次服务（lifespan 会建 demo / guest）再跑")
                continue
            for path in sorted((BACKEND / "kb_samples" / name).glob("*.md")):
                title, text = parse(path)
                stats = await kb.ingest(str(uid), path.stem, text, title)
                print(f"{name:5} {path.stem:12} {title}  {stats}")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
