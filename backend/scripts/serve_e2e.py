"""M2 浏览器实测用：端口 8001、独立库 oasis_e2e，不碰 8000 上的服务和它的库（启动时的 recover 会改 running 的 run）。"""

import os
from pathlib import Path

import uvicorn

os.environ["DATABASE_URL"] = "postgresql://oasis:oasis@127.0.0.1:5433/oasis_e2e"
uvicorn.run("app.main:app", host="127.0.0.1", port=8001, app_dir=str(Path(__file__).parent.parent))
