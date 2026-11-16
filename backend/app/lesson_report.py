"""P4 验收的分析（纯函数，scripts/lesson_eval.py 调用）：收益 / 回归 / 噪声底 / 有害更新率 / 门禁拦截。

协议（WikiTrace）：每道测试题无经验跑 6 次，前 3 次多数定基准，后 3 次多数和前 3 次比，翻转的比例就是噪声底。
每种配置（不过门禁全部提交 / 过门禁 / 去掉证伪题的消融）跑 3 次取多数：基准没过、它过了 = 收益；基准过了、它没过 = 回归。
有害更新（FGSE）：提交上去的经验里，在测试集上造成过回归的比例。归因靠注入记录：哪道回归题的失败 trial 里注入了它。
"""

from app.lesson_l1 import majority


def compare(tests: list[dict], recs: list[dict], configs: list[str]) -> dict:
    """recs = [{config, probe, trial, passed, injected: [原经验 id]}]。返回每种配置的逐题结果和合计。"""
    out = {"rows": [], "noise_flips": 0, "noise_n": 0}
    for t in tests:
        r = [x for x in recs if x["probe"] == t["id"]]
        base = sorted((x for x in r if x["config"] == "base"), key=lambda x: x["trial"])
        b1, b2 = majority([x["passed"] for x in base[:3]]), majority([x["passed"] for x in base[3:]])
        row = {"probe": t["id"], "base1": b1, "base2": b2, "base_pass": [x["passed"] for x in base]}
        if None not in (b1, b2):
            out["noise_n"] += 1
            out["noise_flips"] += b1 != b2
        for c in configs:
            mine = [x for x in r if x["config"] == c]
            m = majority([x["passed"] for x in mine])
            row[c] = {"maj": m, "pass": [x["passed"] for x in mine], "gain": b1 is False and m is True,
                      "regression": b1 is True and m is False,
                      # 失败的那几次 trial 里注入过的经验：回归归因用
                      "blamed": sorted({i for x in mine if x["passed"] is False for i in x["injected"]})}
        out["rows"].append(row)
    out["totals"] = {c: {"gains": sum(x[c]["gain"] for x in out["rows"]),
                         "regressions": sum(x[c]["regression"] for x in out["rows"])} for c in configs}
    return out


def harmful(cmp: dict, config: str, committed: list[int]) -> dict:
    """提交的经验里，在本配置的某道回归题上被归因的个数 / 提交总数。"""
    bad = sorted({i for x in cmp["rows"] if x[config]["regression"] for i in x[config]["blamed"]} & set(committed))
    return {"committed": len(committed), "harmful": bad, "rate": len(bad) / len(committed) if committed else 0.0}


def gate_table(cands: list[dict]) -> dict:
    """cands = [{id, label, expect: L0/L1/None（None = 真实提炼出的）, status, l0_ok, commit, commit_noF}]。
    已知坏经验：被哪一级拦下；去掉证伪题时会不会放进来。"""
    known = [c for c in cands if c["expect"]]
    blocked = [c for c in known if not c["commit"]]
    leak_nof = [c for c in known if c["commit_noF"]]
    return {"known_bad": len(known), "blocked": len(blocked), "blocked_rate": len(blocked) / len(known) if known else 0.0,
            "at_expected_level": sum((c["expect"] == "L0") == (not c["l0_ok"]) for c in blocked),
            "leaked_without_falsify": [c["label"] for c in leak_nof],
            "real": [c for c in cands if not c["expect"]]}
