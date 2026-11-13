"""M7 第 2 步：子智能体的工具裁剪。只能从父能用的里缩小；不能再派子智能体；要人确认的当拒绝。"""

import asyncio

from app import recall, rules, subagent_tool, tool_search, tools, transfer  # noqa: F401  import 即注册工具


PARENT = frozenset({"weather:read", "transfer:write"})  # 没有 stock:read


def in_child(parent_perms, requested, fn, parent_rules=()):
    """先在父的上下文里算出子上下文（和 runner 里的顺序一样），再在子上下文里跑 fn。"""
    async def main():
        tools.current.set(tools.ToolContext(permissions=parent_perms, rules=list(parent_rules)))
        tools.current.set(tools.child_context(None, requested))
        return await fn()
    return asyncio.run(main())


def names():
    return asyncio.sleep(0, sorted(t["name"] for t in tools.list_tools()))


def test_only_narrows_never_widens():
    got = in_child(PARENT, ["get_weather", "get_stock", "calculate", "spawn_subagents", "no_such"], names)
    assert got == ["calculate", "get_weather"]  # 没权限的 get_stock、不存在的、防递归的都没有


def test_default_is_parent_minus_root_only():
    got = in_child(PARENT, None, names)
    assert "spawn_subagents" not in got and "tool_search" not in got and "recall" not in got
    assert {"calculate", "get_weather", "transfer"} <= set(got)
    assert "call_tool" not in got  # 延迟工具直接给完整定义，不需要代理
    assert in_child(PARENT, None, lambda: asyncio.sleep(0, tools.deferred_catalog())) == ""


def test_execution_also_blocked():
    unknown = in_child(PARENT, ["calculate"], lambda: tools.execute("no_such", {}))[0]
    for name, args in [("get_weather", {"city": "北京"}),                                    # 父能用，子没点名
                       ("call_tool", {"tool_name": "get_weather", "arguments": {"city": "北京"}}),
                       ("spawn_subagents", {"tasks": [{"title": "x", "task": "y"}]})]:
        out, is_error = in_child(PARENT, ["calculate"], lambda: tools.execute(name, args))
        assert is_error and out == unknown.replace("no_such", name if name != "call_tool" else "get_weather")


def test_root_only_wins_even_if_whitelist_is_wrong():
    ctx = tools.ToolContext(permissions=PARENT, only=frozenset({"spawn_subagents", "calculate"}))
    assert not tools.allowed(ctx, tools._REGISTRY["spawn_subagents"])
    assert tools.allowed(ctx, tools._REGISTRY["calculate"])


def test_ask_becomes_refusal_but_allow_rule_still_runs():
    use = {"id": "t1", "name": "transfer", "input": {"to": "张三", "amount": 300}}
    call = in_child(PARENT, None, lambda: asyncio.sleep(0, tools.prepare(use)))
    assert call.decision == rules.ASK and "NEEDS_USER_CONFIRMATION" in call.error  # 有 error → 不会执行
    small = [{"tool_pattern": "transfer", "conditions": [{"field": "amount", "op": "lte", "value": 500}],
              "effect": "allow"}]
    call = in_child(PARENT, None, lambda: asyncio.sleep(0, tools.prepare(use)), parent_rules=small)
    assert call.decision == rules.ALLOW and call.error is None  # 规则放行的照常执行


def test_spawn_rejects_tool_list_that_leaves_child_with_nothing():
    """真实模型实测：父填了 ["call_tool", "tool_search"]，子任务一个工具都没有，只能在文字里假装调用。现在直接报错让父改。"""
    args = {"tasks": [{"title": "慢调用", "task": "调一次 slow_task", "tools": ["call_tool", "tool_search"]}]}
    out, is_error = as_parent(lambda: tools.execute("spawn_subagents", args))
    assert is_error and "没有可用的工具" in out


def as_parent(fn):
    async def main():
        tools.current.set(tools.ToolContext(permissions=PARENT))
        return await fn()
    return asyncio.run(main())
