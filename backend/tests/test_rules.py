"""M5 第 2 步：规则判定 deny > ask > allow，按字段匹配参数，没命中按风险等级。"""

import pytest

from app import db, rules
from test_db import with_fresh_db

# 用户给的例子：transfer 金额 ≤100 放行，>100 要确认，给 blocked_user 一律拒绝
TRANSFER_RULES = [
    {"tool_pattern": "transfer", "conditions": [{"field": "amount", "op": "lte", "value": 100}], "effect": "allow"},
    {"tool_pattern": "transfer", "conditions": [{"field": "amount", "op": "gte", "value": 100.01}], "effect": "ask"},
    {"tool_pattern": "transfer", "conditions": [{"field": "to", "op": "eq", "value": "blocked_user"}], "effect": "deny"},
]


def d(args, tool="transfer", risk="high", rs=TRANSFER_RULES):
    return rules.decide(tool, risk, args, rs)


def test_transfer_example():
    assert d({"to": "alice", "amount": 50}) == "allow"
    assert d({"to": "alice", "amount": 100}) == "allow"      # 边界：≤100
    assert d({"to": "alice", "amount": 500}) == "ask"


def test_deny_wins_when_two_rules_hit():
    """反例：给 blocked_user 转 50，同时命中 allow（≤100）和 deny 两条，不是一条。
    如果按「第一条命中就返回」实现，规则顺序就决定了结果，allow 排前面就放行了。"""
    args = {"to": "blocked_user", "amount": 50}
    assert [r["effect"] for r in TRANSFER_RULES if rules.matches(r, "transfer", args)] == ["allow", "deny"]
    assert d(args) == "deny"
    assert d(args, rs=list(reversed(TRANSFER_RULES))) == "deny"  # 顺序无关


def test_default_by_risk_when_nothing_hits():
    assert d({"city": "北京"}, tool="get_weather", risk="low") == "allow"
    assert d({"x": 1}, tool="delete_all", risk="high") == "ask"   # 没有规则的写工具：要确认，不是放行


def test_types_must_match():
    assert d({"to": "alice", "amount": True}) == "ask"           # True 是 int 的子类，不能当成 1 元
    assert d({"to": "alice", "amount": "50"}) == "ask"           # 字符串不参与大小比较


def test_regex_is_fullmatch_on_one_field():
    r = [{"tool_pattern": "mcp__*", "conditions": [{"field": "path", "op": "regex", "value": r"/tmp/.*"}],
          "effect": "allow"}]
    assert rules.decide("mcp__fs__write", "high", {"path": "/tmp/a.txt"}, r) == "allow"
    assert rules.decide("mcp__fs__write", "high", {"path": "/etc/x;/tmp/a"}, r) == "ask"  # search 会误放行
    assert rules.decide("MCP__fs__write", "high", {"path": "/tmp/a"}, r) == "ask"         # 大小写敏感


def test_validate_rule_rejects_bad_rules():
    with pytest.raises(ValueError):
        rules.validate_rule({"tool_pattern": "x", "conditions": [{"field": "a", "op": "like", "value": 1}],
                             "effect": "allow"})
    with pytest.raises(ValueError):
        rules.validate_rule({"tool_pattern": "x", "conditions": [], "effect": "maybe"})


def test_rules_from_db_user_plus_global():
    async def body():
        alice = await db.create_user("alice", "t-alice")
        bob = await db.create_user("bob", "t-bob")
        await db.add_rule(None, "transfer", [{"field": "to", "op": "eq", "value": "blocked_user"}], "deny")
        await db.add_rule(alice, "transfer", [{"field": "amount", "op": "lte", "value": 100}], "allow")
        a, b = await db.get_rules(alice), await db.get_rules(bob)
        assert len(a) == 2 and len(b) == 1                       # 全局规则人人都有，个人规则互不可见
        assert rules.decide("transfer", "high", {"to": "x", "amount": 50}, a) == "allow"
        assert rules.decide("transfer", "high", {"to": "x", "amount": 50}, b) == "ask"
    with_fresh_db(body)
