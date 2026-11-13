"""M8 浏览器验收用：端口 8003，连评测库 oasis_eval 和评测的 Chroma 目录，看 /eval 结果页并点进 trace。
评测脚本跑完后再起（脚本每次开跑会重建这个库）。"""

import os
from pathlib import Path

import uvicorn

BACKEND = Path(__file__).parent.parent
os.environ["DATABASE_URL"] = "postgresql://oasis:oasis@127.0.0.1:5433/oasis_eval"
os.environ["KB_DIR"] = str(BACKEND / "data" / "chroma_eval")
uvicorn.run("app.main:app", host="127.0.0.1", port=8003, app_dir=str(BACKEND))
