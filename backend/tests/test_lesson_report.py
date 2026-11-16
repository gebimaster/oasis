"""P4 验收分析的纯函数：收益 / 回归 / 噪声底 / 有害更新率的口径。"""

from app import lesson_report as lr


def recs(probe, config, flags, injected=()):
    return [{"probe": probe, "config": config, "trial": i, "passed": f, "injected": list(injected)}
            for i, f in enumerate(flags)]


def test_compare_gain_regression_noise_and_blame():
    tests = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    r = (recs("a", "base", [False] * 6) + recs("a", "gated", [True, True, False], [1])          # 收益
         + recs("b", "base", [True] * 6) + recs("b", "gated", [False, False, True], [2])         # 回归，归因到 2
         + recs("b", "all", [False, True, False], [2, 3])                                         # 失败的两次都注入了 2、3
         + recs("a", "all", [True, True, True]) + recs("c", "all", [True] * 3)
         + recs("c", "base", [True, True, True, False, False, True]) + recs("c", "gated", [True] * 3))
    cmp = lr.compare(tests, r, ["gated", "all"])
    assert cmp["totals"]["gated"] == {"gains": 1, "regressions": 1}
    assert cmp["totals"]["all"] == {"gains": 1, "regressions": 1}
    assert cmp["noise_n"] == 3 and cmp["noise_flips"] == 1                          # c：前 3 次过、后 3 次多数没过
    b = next(x for x in cmp["rows"] if x["probe"] == "b")
    assert b["gated"]["blamed"] == [2] and b["all"]["blamed"] == [2, 3]
    assert next(x for x in cmp["rows"] if x["probe"] == "a")["gated"]["blamed"] == [1]  # 失败的那次也记，但不是回归
    h = lr.harmful(cmp, "gated", [1, 2])
    assert h == {"committed": 2, "harmful": [2], "rate": 0.5}
    assert lr.harmful(cmp, "all", [1, 2, 3, 4])["harmful"] == [2, 3]                    # 只算提交了的
    assert lr.harmful(cmp, "gated", [])["rate"] == 0.0


def test_gate_table():
    cands = [{"id": 1, "label": "real", "expect": None, "l0_ok": True, "commit": True, "commit_noF": True},
             {"id": 2, "label": "url", "expect": "L0", "l0_ok": False, "commit": False, "commit_noF": False},
             {"id": 3, "label": "overbroad", "expect": "L1", "l0_ok": True, "commit": False, "commit_noF": True},
             {"id": 4, "label": "slipped", "expect": "L1", "l0_ok": True, "commit": True, "commit_noF": True}]
    t = lr.gate_table(cands)
    assert t["known_bad"] == 3 and t["blocked"] == 2 and round(t["blocked_rate"], 3) == 0.667
    assert t["at_expected_level"] == 2 and t["leaked_without_falsify"] == ["overbroad", "slipped"]
    assert [c["id"] for c in t["real"]] == [1]


def test_blame_only_failed_trials_and_committed():
    tests = [{"id": "b"}]
    r = recs("b", "base", [True] * 6) + [
        {"probe": "b", "config": "all", "trial": 0, "passed": False, "injected": [2]},
        {"probe": "b", "config": "all", "trial": 1, "passed": False, "injected": [2, 3]},
        {"probe": "b", "config": "all", "trial": 2, "passed": True, "injected": [5]}]   # 过了的那次注入的不算
    cmp = lr.compare(tests, r, ["all"])
    assert cmp["rows"][0]["all"]["blamed"] == [2, 3]
    assert lr.harmful(cmp, "all", [2, 5])["harmful"] == [2]                  # 3 没提交，不算进有害
