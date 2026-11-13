"""O10 第一步：能否并发由工具声明，保序分组，组内并发有上限。不用 sleep 计时断言，靠事件记进出顺序和计数器。"""

import asyncio

import pytest

from app import tools

LOG: list[str] = []
STATE = {"cur": 0, "max": 0}


def make(name: str, safe: bool):
    async def handler(args):
        LOG.append(f"+{name}")
        STATE["cur"] += 1
        STATE["max"] = max(STATE["max"], STATE["cur"])
        for _ in range(5):  # 让出几次：同一组里的其他调用此时都已经开始了
            await asyncio.sleep(0)
        STATE["cur"] -= 1
        LOG.append(f"-{name}")
        return name
    return tools.Tool(name=name, description="测试用", input_schema={"type": "object"}, handler=handler,
                      permission=tools.PUBLIC, risk=tools.LOW, concurrency_safe=safe)


@pytest.fixture
def fake_tools():
    LOG.clear()
    STATE.update(cur=0, max=0)
    names = [("mkdir", False), ("write", False)] + [(f"read{i}", True) for i in range(8)]
    for name, safe in names:
        tools.register(make(name, safe))
    yield
    for name, _ in names:
        tools.unregister(name)


def calls(*names):
    return [tools.prepare({"id": f"c{i}", "name": n, "input": {}}, check_rules=False) for i, n in enumerate(names)]


def run(cs):
    return asyncio.run(tools.execute_all(cs))


def test_groups_keep_model_order(fake_tools):
    g = tools.groups(calls("read0", "read1", "mkdir", "write", "read2", "read3"))
    assert [[c.name for c in x] for x in g] == [["read0", "read1"], ["mkdir"], ["write"], ["read2", "read3"]]


def test_unsafe_calls_never_overlap_and_keep_order(fake_tools):
    """「建目录」和「往目录里写文件」：前一个完全结束后一个才开始，顺序和模型给的一致。"""
    out = run(calls("mkdir", "write"))
    assert LOG == ["+mkdir", "-mkdir", "+write", "-write"]
    assert [r["content"] for r in out] == ["mkdir", "write"]


def test_reads_after_a_write_wait_for_it(fake_tools):
    """不把可并发的挪到前面：先写后读，读一定看到写完之后的状态。"""
    run(calls("write", "read0", "read1"))
    assert LOG.index("-write") < LOG.index("+read0") and LOG.index("-write") < LOG.index("+read1")


def test_safe_calls_run_concurrently_up_to_the_cap(fake_tools):
    out = run(calls(*[f"read{i}" for i in range(8)]))
    assert STATE["max"] == tools.MAX_PARALLEL           # 确实并发了，而且没超过上限
    assert [r["content"] for r in out] == [f"read{i}" for i in range(8)]  # 结果顺序和入参一致


def test_calls_with_known_results_do_not_break_groups(fake_tools):
    """检查没过的调用（不存在的工具）不执行，算可并发，不会把两边的读拆成两组。"""
    g = tools.groups(calls("read0", "nope", "read1"))
    assert len(g) == 1


def test_declared_tools():
    """声明在工具上，不拿 risk 推断：沙箱 write_file / run_code 是 LOW 却会改状态，必须串行。"""
    from app import kb_tool, sandbox_tools, subagent_tool, transfer, web_tools  # noqa: F401  import 即注册（复核⑨：
    # 少了 transfer、subagent_tool 的话单独跑这个文件时它们不在注册表里，下面那条断言等于没测）
    reg = tools._REGISTRY
    assert all(reg[n].concurrency_safe for n in ("get_weather", "kb_search", "web_search", "read_file"))
    assert not any(reg[n].concurrency_safe for n in ("write_file", "run_code", "spawn_subagents", "transfer"))
    assert reg["write_file"].risk == tools.LOW  # 正是「不能拿 risk 判断」的反例
