"""M5 规则判定：一次工具调用是直接执行（allow）、等人确认（ask），还是拒绝（deny）。

两层分工：tools.allowed() 管「能不能看见/调用这个工具」；这里的 decide() 只在执行入口、
allowed 和 jsonschema 校验都通过之后调用，管「这次带着这些参数，能不能直接执行」。
优先级 deny > ask > allow；规则可以按参数字段比较（不只看工具名）。
"""

import fnmatch
import re

DENY, ASK, ALLOW = "deny", "ask", "allow"
SEVERITY = {DENY: 0, ASK: 1, ALLOW: 2}  # 数字越小越严；命中多条取最严的
OPS = {"eq", "lte", "gte", "in", "regex"}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)  # True 也是 int，不能当金额比


def _match_cond(args: dict, c: dict) -> bool:
    """单个条件：{"field": "amount", "op": "lte", "value": 100}。字段不存在或类型不对一律不命中。"""
    if c["field"] not in args:
        return False
    v, target = args[c["field"]], c["value"]
    match c["op"]:
        case "eq":
            return v == target
        case "lte":
            return _is_number(v) and v <= target
        case "gte":
            return _is_number(v) and v >= target
        case "in":
            return v in target
        case "regex":  # 只对单个字符串字段、整串匹配；不对整段 JSON 跑正则（空格、键序都能绕过）
            return isinstance(v, str) and re.fullmatch(target, v) is not None
    return False


def matches(rule: dict, tool_name: str, args: dict) -> bool:
    # fnmatchcase：fnmatch.fnmatch 在 Windows 上不区分大小写
    return (fnmatch.fnmatchcase(tool_name, rule["tool_pattern"])
            and all(_match_cond(args, c) for c in rule["conditions"]))


def decide(tool_name: str, risk: str, args: dict, rules: list[dict]) -> str:
    """命中的规则里取最严的；一条都没命中按风险等级：low 直接执行，high 要确认。"""
    hits = [r for r in rules if matches(r, tool_name, args)]
    if hits:
        return min(hits, key=lambda r: SEVERITY[r["effect"]])["effect"]
    return ALLOW if risk == "low" else ASK


def validate_rule(rule: dict) -> None:
    """写库前校验，免得一条坏规则在判定时抛异常。"""
    if rule["effect"] not in SEVERITY:
        raise ValueError(f"effect 必须是 {set(SEVERITY)}")
    for c in rule["conditions"]:
        if c.get("op") not in OPS or "field" not in c or "value" not in c:
            raise ValueError(f"坏条件：{c}")
        if c["op"] == "regex":
            re.compile(c["value"])
