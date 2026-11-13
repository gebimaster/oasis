"""O6 第一步：动作指纹的规范化（纯函数，不碰循环）。

每条断言比的是两份独立写出来的入参，不拿同一个来源的两个值互相比。
"""

import random

from app.loop_guard import fingerprint


def same(a, b, name="t"):
    return fingerprint(name, a) == fingerprint(name, b)


def test_key_order_does_not_matter():
    assert same({"a": 1, "b": {"x": 1, "y": 2}}, {"b": {"y": 2, "x": 1}, "a": 1})


def test_numbers():
    assert same({"n": 1}, {"n": 1.0})
    assert same({"n": [0.5, 2]}, {"n": [0.50, 2.0]})
    assert same({"n": 0}, {"n": -0.0})
    assert not same({"n": True}, {"n": 1})          # Python 里 True == 1，规范化不能把它们合成一个
    assert not same({"n": False}, {"n": 0})
    assert not same({"n": 2**53}, {"n": 2**53 + 1})  # int 转 float 会丢精度撞在一起；方向必须是 float → int
    assert not same({"n": "1"}, {"n": 1})            # 字符串和数字不合并
    assert not same({"n": None}, {"n": 0})


def test_strings_l2():
    assert same({"city": "北京"}, {"city": "  北京　"})        # 首尾空白，含全角空格
    assert same({"q": "ＡＢＣ１２３"}, {"q": "abc123"})              # NFKC：全角转半角
    assert same({"q": "Hello World"}, {"q": "hello world"})        # 忽略大小写
    assert same({"q": "STRASSE"}, {"q": "straße"})                  # casefold 比 lower 多做的一步
    assert not same({"q": "a b"}, {"q": "ab"})                      # 中间的空白不动
    assert not same({"q": "北京"}, {"q": "南京"})


def test_known_false_positive_case_sensitive_values():
    """接受的误判（你选的 L2）：区分大小写的值会被当成同一个动作。写成测试，免得以后以为没有这回事。"""
    assert same({"path": "/workspace/A.txt"}, {"path": "/workspace/a.txt"})


def test_keys_are_not_normalized():
    assert not same({"A": 1}, {"a": 1})
    assert not same({"a": 1, "A": 2}, {"a": 2})  # 键规范化的话两个键会合成一个、丢信息


def test_lone_surrogate_does_not_crash():
    """O6 复核甲：JSON 里合法的单个代理字符（\\ud800）解析出来是孤立代理，utf-8 严格编码会抛 UnicodeEncodeError，
    冒出 run_loop。用 surrogatepass 编码：不崩，不同的代理字符也不撞。"""
    import json
    a, b = json.loads('{"q": "\\ud800"}'), json.loads('{"q": "\\ud801"}')
    assert fingerprint("t", a) == fingerprint("t", json.loads('{"q": " \\ud800 "}'))
    assert fingerprint("t", a) != fingerprint("t", b)


def test_list_order_and_tool_name_matter():
    assert not same({"x": [1, 2]}, {"x": [2, 1]})
    assert fingerprint("get_weather", {"city": "北京"}) != fingerprint("get_stock", {"city": "北京"})
    assert fingerprint("Lookup", {}) != fingerprint("lookup", {})  # 工具名原样比，不规范化


# ---------------- 性质测试 ----------------
# 生成器产出边界类型：布尔、None、2**53 附近的整数、整数值的浮点、嵌套；_variant 再改出全角、首尾空白、大小写、
# 打乱的键顺序、整数换等值浮点。-0.0 生成器产不出来，由 test_numbers 单测

_ASCII = "abcXYZ019_-./ "
_CJK = "北京天气查询"


def _gen(rng: random.Random, depth: int = 0):
    kind = rng.choice(["bool", "none", "int", "bigint", "float", "intfloat", "str", "str", "list", "dict", "dict"]
                      if depth < 3 else ["bool", "int", "float", "str"])
    if kind == "bool":
        return rng.choice([True, False])
    if kind == "none":
        return None
    if kind == "int":
        return rng.choice([0, 1, -1, rng.randint(-1000, 1000)])
    if kind == "bigint":
        return 2**53 + rng.randint(-2, 2)
    if kind == "float":
        return rng.choice([0.5, -2.25, 1e-3, rng.random()])
    if kind == "intfloat":
        return float(rng.randint(-5, 5))
    if kind == "str":
        return "".join(rng.choice(_ASCII + _CJK) for _ in range(rng.randint(0, 8)))
    if kind == "list":
        return [_gen(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return {f"k{i}": _gen(rng, depth + 1) for i in range(rng.randint(0, 4))}


def _variant(rng: random.Random, v):
    """写法不同、意思相同的一份：打乱键顺序、整数换成等值浮点、字符串加空白 / 换大小写 / 转全角。"""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int):
        return float(v) if abs(v) < 2**53 and rng.random() < 0.5 else v
    if isinstance(v, float):
        return int(v) if v.is_integer() and rng.random() < 0.5 else v
    if isinstance(v, str):
        s = "".join(c.swapcase() if rng.random() < 0.5 else c for c in v)
        s = "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" and rng.random() < 0.5 else c for c in s)
        return rng.choice(["", " ", "　", "\t"]) + s + rng.choice(["", " ", "　", "\n"])
    if isinstance(v, list):
        return [_variant(rng, x) for x in v]
    items = list(v.items())
    rng.shuffle(items)
    return {k: _variant(rng, x) for k, x in items}


def _leaves(v, path=()):
    if isinstance(v, dict):
        for k, x in v.items():
            yield from _leaves(x, path + (k,))
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from _leaves(x, path + (i,))
    else:
        yield path, v


def _replace(v, path, new):
    if not path:
        return new
    head, rest = path[0], path[1:]
    if isinstance(v, dict):
        return {**v, head: _replace(v[head], rest, new)}
    return [_replace(x, rest, new) if i == head else x for i, x in enumerate(v)]


def _different(v):
    """规范化之后也一定不同的值。"""
    if isinstance(v, bool):
        return not v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v + 1
    return v.strip() + "z" if v.strip() else "z"


def test_property_variants_collide_and_leaf_changes_dont():
    rng = random.Random(20261109)
    kinds = set()
    for _ in range(500):
        args = {"root": _gen(rng)}
        assert fingerprint("t", args) == fingerprint("t", _variant(rng, args))
        for path, leaf in _leaves(args):
            kinds.add(type(leaf).__name__ if not (isinstance(leaf, int) and abs(leaf) >= 2**53) else "bigint")
            changed = _replace(args, path, _different(leaf))
            assert fingerprint("t", args) != fingerprint("t", changed), (args, path)
    assert {"bool", "NoneType", "int", "bigint", "float", "str"} <= kinds  # 生成器真的产出过这些边界类型
