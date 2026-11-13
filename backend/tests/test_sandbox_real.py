"""M9：连真实 Cube 的集成测试。默认跳过；虚拟机开着时 CUBE=1 pytest tests/test_sandbox_real.py。"""

import asyncio
import os
import time

import pytest
from dotenv import load_dotenv

from app import db, sandbox, sandbox_tools, tools
from test_db import with_fresh_db

pytestmark = pytest.mark.skipif(os.getenv("CUBE") != "1", reason="需要本机 Cube：CUBE=1")
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


async def _setup():
    from app.sandbox_e2b import E2BBackend
    sandbox.pool = sandbox.Pool(E2BBackend())
    uid = await db.create_user("alice", "token-alice")
    cid = await db.create_conversation(uid)
    run_id = await db.start_run(cid, [{"type": "text", "text": "算一下"}])
    tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, run_id=run_id,
                                        permissions=frozenset({sandbox_tools.PERMISSION})))
    return uid, cid


def test_real_cube_end_to_end():
    async def body():
        try:
            uid, cid = await _setup()
            content, err = await tools.execute("run_code", {"code": "x = sum(i * i for i in range(10))\nx"})
            assert not err and "[结果]\n285" in content
            content, err = await tools.execute("run_code", {"code": "x + 1"})  # 同一会话：变量还在
            assert "286" in content

            t = time.monotonic()  # 死循环：超时 → 结果未知 → 内核重启，下一次调用不被堵
            content, err = await tools.execute("run_code", {"code": "while True: pass", "timeout": 2})
            assert err and "OUTCOME_UNKNOWN" in content
            content, err = await tools.execute("run_code", {"code": "1 + 1"})
            assert "[结果]\n2" in content and time.monotonic() - t < 25
            content, err = await tools.execute("run_code", {"code": "x"})  # 重启过，变量没了
            assert err and "NameError" in content

            content, err = await tools.execute("run_code", {"code": "import urllib.request\n"
                                                            "urllib.request.urlopen('http://example.com', timeout=5)"})
            assert err and "URLError" in content  # 断网

            content, err = await tools.execute("run_shell", {"command": "echo out; exit 3"})
            assert err and "[退出码] 3" in content and "out" in content

            # 取消 shell：进程要真的停（实测前台命令被取消后还在跑，所以用了后台句柄 + kill）
            task = asyncio.create_task(tools.execute(
                "run_shell", {"command": "for i in $(seq 1 30); do echo $i >> /tmp/tick; sleep 1; done", "timeout": 60}))
            await asyncio.sleep(4)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            a, _ = await tools.execute("run_shell", {"command": "wc -l < /tmp/tick"})
            await asyncio.sleep(3)
            b, _ = await tools.execute("run_shell", {"command": "wc -l < /tmp/tick"})
            assert a.splitlines()[-1] == b.splitlines()[-1]

            await tools.execute("write_file", {"path": "data/a.txt", "content": "你好" * 3000})
            content, err = await tools.execute("read_file", {"path": "data/a.txt", "offset": 5000})
            assert not err and "第 5000–6000 字，共 6000 字" in content
            content, err = await tools.execute("run_code", {"code": "len(open('data/a.txt').read())"})
            assert "[结果]\n6000" in content  # 相对路径：文件工具和代码看到的是同一个目录

            # 对账：库里的行删掉 → 这个沙箱变成孤儿，启动对账会把它杀掉
            sid = await db.pool().fetchval("SELECT sandbox_id FROM sandboxes WHERE user_id = $1", uid)
            await db.pool().execute("DELETE FROM sandboxes")
            killed, _ = await sandbox.pool.reconcile()
            assert killed >= 1 and not await sandbox.pool.backend.alive(sid)
        finally:
            for sid in await sandbox.pool.backend.list_ours() if sandbox.pool else []:
                await sandbox.pool.backend.kill(sid)  # 不管成败，测试建的沙箱都收掉
            sandbox.pool = None
    with_fresh_db(body)
