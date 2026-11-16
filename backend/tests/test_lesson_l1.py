"""P4 第 6 步：L1 的挑题和判定（纯函数），以及两个演示工具的隐含规矩。"""

import asyncio

import pytest

from app import lesson_demo, lesson_l1 as l1

LESSON = {"precondition": "提交报销时", "behavior": "submit_expense 的金额按分填", "tools": ["submit_expense"]}
BANK = [
    {"id": "e1", "input": "报销 45 元午餐", "tools": ["submit_expense"], "base": False},
    {"id": "e2", "input": "报销 128 元打车", "tools": ["submit_expense"], "base": False},
    {"id": "e3", "input": "报销 9 元", "tools": ["submit_expense"], "base": False},     # 目标题只取 2 道
    {"id": "e4", "input": "金额还没定的报销", "tools": ["submit_expense"], "base": True},
    {"id": "d1", "input": "45 元打 8 折是多少金额", "tools": ["calculate"], "base": True},
    {"id": "w1", "input": "北京天气", "tools": ["get_weather"], "base": True},
    {"id": "x1", "input": "算一下报销", "tools": ["calculate"], "base": None},          # 基线平票：不用
    {"id": "x2", "input": "报销金额怎么算", "tools": ["calculate"], "base": False},     # 基线不会：不当证伪题
]


def test_pick_roles():
    got = l1.pick(LESSON, BANK, {"d1": 0.6, "w1": 0.1})
    assert [p["id"] for p in got["target"]] == ["e1", "e2"]
    assert [p["id"] for p in got["regress"]] == ["e4"]
    assert [p["id"] for p in got["falsify"]] == ["d1", "w1"]                 # 最像的诱饵排前面
    assert l1.pick(LESSON, [b for b in BANK if b["id"] != "d1" and b["id"] != "w1"], {})["falsify"] == []


def row(role, base, *w, probe="p"):
    return {"probe": probe, "role": role, "base": base, "with": list(w)}


@pytest.mark.parametrize("rows,commit,why", [
    ([row("source", False, True, True, False), row("falsify", True, True, True, True)], True, None),
    ([row("source", False, False, False, True), row("falsify", True, True, True, True)], False, "没有一道题"),
    ([row("target", False, True, True, True), row("falsify", True, True, False, False, probe="d1")], False, "falsify:d1"),
    ([row("target", False, True, True, True), row("regress", True, True, False, None, probe="e4"),
      row("falsify", True, True, True, True)], False, "regress:e4"),               # 平票算没守住
    ([row("target", False, True, True, True)], False, "挑不出证伪题"),
    ([row("target", True, True, True, True), row("falsify", True, True, True, True)], False, "没有一道题"),  # 本来就会
])
def test_decide(rows, commit, why):
    d = l1.decide(rows)
    assert d["commit"] is commit and (why is None or any(why in w for w in d["why"])), d


def test_decide_without_falsify_is_the_ablation():
    rows = [row("target", False, True, True, True), row("falsify", True, False, False, True, probe="d1")]
    assert l1.decide(rows)["commit"] is False
    abl = l1.decide(rows, use_falsify=False)
    assert abl["commit"] is True and "不看" in abl["summary"]
    assert l1.decide([row("target", False, True, True, True)], use_falsify=False)["commit"] is True


def test_demo_tools_hidden_rules():
    async def body():
        lesson_demo.register()
        from app import tools
        sub, fl = tools._REGISTRY["submit_expense"].handler, tools._REGISTRY["search_flights"].handler
        assert "0.32 元" in await sub({"amount": 32, "category": "打车"})     # 静默错误：传元就少 100 倍
        assert "32.00 元" in await sub({"amount": 3200, "category": "打车"})
        with pytest.raises(ValueError, match="INVALID_CITY: 北京"):
            await fl({"origin": "北京", "destination": "CTU", "date": "2026-11-20"})
        a = await fl({"origin": "pek", "destination": "CTU", "date": "2026-11-20"})
        assert a == await fl({"origin": "PEK", "destination": "CTU", "date": "2026-11-20"}) and "成都(CTU)" in a
        tools.unregister("submit_expense")
        tools.unregister("search_flights")
    asyncio.run(body())


def test_new_grade_checks():
    from app.eval_grade import Trial, tool_calls_max, tool_errors_max
    t = Trial(calls=[{"name": "f", "is_error": True}, {"name": "f", "is_error": False},
                     {"name": "f", "is_error": False}, {"name": "g", "is_error": True}])
    assert tool_errors_max(t, "f", 0)[0] is False and tool_errors_max(t, "f", 1)[0] is True
    assert tool_calls_max(t, "f", 1)[0] is False and tool_calls_max(t, "f", 2)[0] is True
    assert tool_errors_max(t, "h")[0] is True
