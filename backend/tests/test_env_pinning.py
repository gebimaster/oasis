"""评测脚本按类别给服务端传 MAX_TOKENS 等变量；服务端加载 .env（override=True）时不能把它们盖掉（O7 真实验收时发现）。"""

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).parent.parent


def _max_tokens_after_import(env: dict) -> str:
    code = "import os, app.main; print(os.environ.get('MAX_TOKENS'))"
    out = subprocess.run([sys.executable, "-c", code], cwd=BACKEND, env={**os.environ, **env},
                         capture_output=True, text=True, timeout=60)
    return out.stdout.strip().splitlines()[-1]


def test_pinned_variables_survive_dotenv_override():
    assert _max_tokens_after_import({"MAX_TOKENS": "123", "ENV_PINNED": "MAX_TOKENS"}) == "123"
