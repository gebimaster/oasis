"""测试公共设置。"""

import pytest

from app import tools

ALL_PERMISSIONS = frozenset({"weather:read", "stock:read"})


@pytest.fixture(autouse=True)
def tool_context():
    """M4 起工具按权限过滤：默认给每个测试一个「全部权限」的 ToolContext（相当于 demo 用户）。
    asyncio.run 会复制当前上下文，所以测试里起的事件循环也看得到；测权限的用例自己再 set 覆盖。"""
    token = tools.current.set(tools.ToolContext(permissions=ALL_PERMISSIONS))
    yield
    tools.current.reset(token)


@pytest.fixture(autouse=True)
def no_kb_rewrite(monkeypatch):
    """O8 复核⑭：.env 里开了 KB_REWRITE=1 的话，知识库测试里弱的假检索会去调真模型。测试一律关掉，要测的自己再开。"""
    monkeypatch.delenv("KB_REWRITE", raising=False)


@pytest.fixture(autouse=True)
def no_memory_extract(monkeypatch):
    """P2：run 正常结束会在后台调真模型抽记忆。测试一律关掉，测记忆的用例自己再开。"""
    monkeypatch.setenv("MEMORY", "0")


@pytest.fixture(autouse=True)
def no_lessons(monkeypatch):
    """P4：经验的读取每个 run 都会查库、可能算一次向量。测试一律关掉，测经验的用例自己再开。"""
    monkeypatch.setenv("LESSONS", "0")
