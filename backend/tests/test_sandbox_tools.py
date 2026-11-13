"""M9 第 2 段：沙箱工具的输出拼装、三类失败、超时夹紧、执行记录。用假后端，不连 Cube。"""

import asyncio

import pytest

from app import db, sandbox, sandbox_tools, tools
from app.sandbox import Output
from tool_codes import execute_with_code
from test_db import with_fresh_db
from test_sandbox import FakeBackend


@pytest.fixture(autouse=True)
def no_pool():
    yield
    sandbox.pool = None


async def _setup(b: FakeBackend):
    sandbox.pool = sandbox.Pool(b)
    uid = await db.create_user("alice", "token-alice")
    cid = await db.create_conversation(uid)
    run_id = await db.start_run(cid, [{"type": "text", "text": "算一下"}])
    tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, run_id=run_id,
                                        permissions=frozenset({sandbox_tools.PERMISSION})))
    return cid, run_id


def _returns(out: Output):
    async def on_run(code, timeout):
        return out
    return on_run


async def _statuses(run_id):
    return [r["status"] for r in await db.pool().fetch(
        "SELECT status FROM tool_executions WHERE run_id = $1 ORDER BY id", run_id)]


def test_result_is_prefixed_and_timeout_clamped():
    async def body():
        b = FakeBackend()
        await _setup(b)
        b.on_run = _returns(Output(stdout="hi\n", result="42"))
        content, is_error = await tools.execute("run_code", {"code": "print('hi'); 41 + 1", "timeout": 9999})
        assert not is_error and content.startswith(sandbox_tools.UNTRUSTED)
        assert "[stdout]\nhi" in content and "[结果]\n42" in content
        assert b.last_timeout == sandbox_tools.MAX_TIMEOUT  # 模型要 9999 秒，按 120 算
    with_fresh_db(body)


def test_long_output_keeps_tail_where_the_error_is():
    async def body():
        b = FakeBackend()
        await _setup(b)
        long = "".join(f"{i}\n" for i in range(100_000))  # 反例：先打 10 万行，再 1/0
        tb = "Traceback\n" + "".join(f"\x1b[31mframe {i}\x1b[0m\n" for i in range(30))
        b.on_run = _returns(Output(stdout=long, error=f"ZeroDivisionError: division by zero\n{tb}"))
        content, is_error = await tools.execute("run_code", {"code": "..."})
        assert is_error and "ZeroDivisionError: division by zero" in content
        assert "99999" in content and "中间省略" in content  # 尾巴还在
        assert "frame 29" in content and "frame 19" not in content and "\x1b" not in content  # traceback 留最后 10 行、去颜色
        saved = next(v for (_, path), v in b.files.items() if path.endswith("-stdout.txt"))
        assert saved == long  # 全文存进了沙箱
    with_fresh_db(body)


def test_code_error_does_not_restart_kernel():
    async def body():
        b = FakeBackend()
        cid, run_id = await _setup(b)
        b.on_run = _returns(Output(error="NameError: name 'x' is not defined\n"))
        content, is_error = await tools.execute("run_code", {"code": "x"})
        assert is_error and "NameError" in content
        assert b.restarts == []  # 代码报错是执行完了，变量都还在
        assert await db.pool().fetchval("SELECT needs_reset FROM sandbox_contexts WHERE conversation_id = $1", cid) is False
        assert await _statuses(run_id) == ["failed"]
    with_fresh_db(body)


def test_timeout_is_outcome_unknown_and_restarts_kernel(monkeypatch):
    monkeypatch.setattr(sandbox_tools, "GRACE", 0.1)

    async def body():
        b = FakeBackend()
        _, run_id = await _setup(b)

        async def hang(code, timeout):
            await asyncio.sleep(3600)  # 死循环；SDK 自己的超时也没生效
        b.on_run = hang
        async with asyncio.timeout(5):  # 必须靠工具自己那层超时在 1.1 秒收住，等不到 150 秒的总闸
            content, is_error = await tools.execute("run_code", {"code": "while True: pass", "timeout": 1})
        assert is_error and "OUTCOME_UNKNOWN" in content and "不要原样重跑" in content
        assert len(b.restarts) == 1  # 内核被重启，后面的调用不会排在死循环后面
        assert await _statuses(run_id) == ["outcome_unknown"]  # OutcomeUnknown 继承 TimeoutError → 记成结果未知
    with_fresh_db(body)


def test_not_started_when_sandbox_unavailable():
    async def body():
        b = FakeBackend()
        _, run_id = await _setup(b)

        async def down(user_id):
            raise ConnectionError("cube unreachable")
        b.create = down
        content, is_error, code = await execute_with_code("run_code", {"code": "1"})
        assert is_error and "NOT_STARTED" in content and "可以稍后重试" in content
        assert code == tools.UNAVAILABLE  # O6：沙箱起不来算依赖不可用
        assert await _statuses(run_id) == ["failed"]  # 没发出去：确定没执行，不是结果未知
    with_fresh_db(body)


def test_model_cannot_pass_sandbox_id():
    async def body():
        b = FakeBackend()
        await _setup(b)
        content, is_error = await tools.execute("run_code", {"code": "1", "sandbox_id": "sb-of-bob"})
        assert is_error and "INVALID_INPUT" in content and b.created == []
    with_fresh_db(body)


def test_read_file_pages_and_missing_file():
    async def body():
        b = FakeBackend()
        await _setup(b)
        await tools.execute("write_file", {"path": "/workspace/a.txt", "content": "x" * 5000})
        content, is_error = await tools.execute("read_file", {"path": "/workspace/a.txt"})
        assert not is_error and "第 0–4000 字，共 5000 字" in content and "offset=4000" in content
        content, _ = await tools.execute("read_file", {"path": "/workspace/a.txt", "offset": 4000})
        assert "第 4000–5000 字" in content and "后面还有" not in content
        content, is_error = await tools.execute("read_file", {"path": "/nope"})
        assert is_error and "文件不存在" in content and b.restarts == []  # 预期内的错误不重启内核
    with_fresh_db(body)


def test_disabled_sandbox_is_unavailable():
    """O6：沙箱没启用（pool 为 None）也是 NOT_STARTED，算依赖不可用。"""
    sandbox.pool = None
    tools.current.set(tools.ToolContext(permissions=frozenset({sandbox_tools.PERMISSION})))
    content, is_error, code = asyncio.run(execute_with_code("run_code", {"code": "1"}))
    assert is_error and "NOT_STARTED" in content and code == tools.UNAVAILABLE
