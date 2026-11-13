"""M8 第 3 步：把 trial 记录汇总成对比表。纯函数，scripts/agent_eval.py 写文件时调用。

口径：
- pass@1：先算每道题 k 次的通过比例，再对题目求平均（每题权重相同，不让跑得多的题占便宜）
- pass^k：k 次全过的题占比；不稳定题 = 0 < pass@1 < 1 的题数（看方差）
- 评委出错、环境故障的 trial 不进分母，单独计数
- 机制达标率只算有机制断言、且本配置没关掉该功能的 trial
- O8 引用：正确率 = 全部 trial 的 ok 之和 /（ok + wrong）之和（按引用数加权，不是各题正确率的平均）；
  覆盖率是各 trial 覆盖率的平均。忠实度 × 引用的四格表把「检索对了回答不对」拆开：
  不忠实但引用都对 = 编了原文没有的内容（生成问题）；忠实但引错 = 内容有依据、出处标错了（归因问题）
"""

import statistics
from collections import defaultdict

from app.budget import hit_rate
from app.eval_grade import pass_metrics


def citation_summary(records: list[dict]) -> dict:
    """O8：从 faithful 那条判定里取 citations（eval_grade.grade 写进去的）。没有知识库题时全是 None。"""
    # 评委出错（passed=None）的 trial 两处都不算（复核⑤：原来正确率算了、四格表没算）
    pairs = [(g.get("passed"), g["citations"]) for r in records for g in r.get("grades", [])
             if g.get("check") == "faithful" and g.get("citations") and g.get("passed") is not None]
    ok = sum(c["ok"] for _, c in pairs)
    wrong = sum(c["wrong"] for _, c in pairs)
    cov = [c["coverage"] for _, c in pairs if c["coverage"] is not None]
    # 「引用无错」只算真有能判的引用、且都判对的；一条能判的都没有（没引、或全是 unchecked）单列，
    # 否则「不忠实·引用无错 = 生成编造」会把根本没引用的回答也算进去（复核⑤）
    grid = {f"{a}·{b}": 0 for a in ("忠实", "不忠实") for b in ("引用无错", "有引错", "无可判引用")}
    for passed, c in pairs:
        kind = "有引错" if c["wrong"] else ("引用无错" if c["ok"] else "无可判引用")
        grid[("忠实" if passed else "不忠实") + "·" + kind] += 1
    return {"citation_correctness": round(ok / (ok + wrong), 3) if ok + wrong else None,
            "citation_coverage": round(sum(cov) / len(cov), 3) if cov else None,
            "citations_total": f'{sum(c["n"] for _, c in pairs)}/{sum(c["unchecked"] for _, c in pairs)}',
            "faithful_x_citation": grid if pairs else None}


def summarize(records: list[dict]) -> dict:
    by_case = defaultdict(list)
    for r in records:
        by_case[(r["category"], r["case"])].append(r)
    per_case = {cid: pass_metrics([r.get("passed") for r in rs]) | {"category": cat}
                for (cat, cid), rs in by_case.items()}
    rated = [m for m in per_case.values() if m["pass@1"] is not None]
    cats = defaultdict(list)
    for m in rated:
        cats[m["category"]].append(m["pass@1"])
    ok = [r for r in records if not r.get("env_error")]
    mech = [r["mechanism"] for r in ok if r.get("mechanism") is not None]
    lat = sorted(r["latency_s"] for r in ok)

    def avg(xs):
        return round(sum(xs) / len(xs), 3) if xs else None
    return {
        "pass@1": avg([m["pass@1"] for m in rated]),
        "pass^k": avg([float(m["pass^k"]) for m in rated]),
        "unstable_cases": sum(1 for m in rated if 0 < m["pass@1"] < 1),
        "by_category": {c: avg(v) for c, v in sorted(cats.items())},
        "mechanism_rate": avg([float(x) for x in mech]),
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_p90_s": round(lat[int(0.9 * (len(lat) - 1))], 2) if lat else None,
        "tokens_avg": avg([sum(r["usage"].values()) for r in ok]),
        "cost_usd_avg": avg([r["cost_usd"] for r in ok]),
        "cost_usd_total": round(sum(r["cost_usd"] for r in ok), 4),
        "hit_rate": hit_rate([r["usage"] for r in ok]),  # O3：按 token 加权，不是各题命中率的平均
        "tool_calls_avg": avg([r["tool_calls"] for r in ok]),
        "judge_errors": sum(m["judge_errors"] for m in per_case.values()) - sum(1 for r in records if r.get("env_error")),
        "env_errors": len(records) - len(ok),
        "n_trials": len(records),
        "per_case": per_case,
    } | citation_summary(ok)


ROWS = [("pass@1", "pass@1"), ("pass^k（k 次全过的题占比）", "pass^k"), ("不稳定题数", "unstable_cases"),
        ("机制达标率", "mechanism_rate"), ("延迟 p50 (s)", "latency_p50_s"), ("延迟 p90 (s)", "latency_p90_s"),
        ("平均 token / 题", "tokens_avg"), ("平均费用 / 题 (USD)", "cost_usd_avg"), ("总费用 (USD)", "cost_usd_total"), ("缓存命中率", "hit_rate"),
        ("平均工具调用 / 题", "tool_calls_avg"), ("评委出错", "judge_errors"), ("环境故障", "env_errors"),
        ("trial 数", "n_trials"), ("引用正确率（O8，按引用数）", "citation_correctness"),
        ("引用覆盖率（O8）", "citation_coverage"), ("引用总数 / 其中判不了", "citations_total"),
        ("忠实度 × 引用（O8）", "faithful_x_citation")]


def report(snapshot: dict, records: list[dict]) -> str:
    cfgs = list(snapshot["configs"])
    s = {c: summarize([r for r in records if r["config"] == c]) for c in cfgs}
    head = "| 指标 | " + " | ".join(f"{c} {snapshot['configs'][c]['name']}" for c in cfgs) + " |"
    sep = "|---" * (len(cfgs) + 1) + "|"
    lines = [f"# 端到端评测 {snapshot['time']}", "",
             f"被测模型 {snapshot['model']}，评委 {snapshot['judge_model']}，每题 {snapshot['trials']} 次，"
             f"数据集 sha {snapshot['dataset_sha']}（{snapshot['n_cases']} 题）。{snapshot['latency_note']}。", "",
             head, sep]
    lines += [f"| {label} | " + " | ".join(str(s[c][key]) for c in cfgs) + " |" for label, key in ROWS]
    cats = sorted({k for c in cfgs for k in s[c]["by_category"]})
    lines += ["", "## 分类 pass@1", "", "| 类别 | " + " | ".join(cfgs) + " |", "|---" * (len(cfgs) + 1) + "|"]
    lines += [f"| {k} | " + " | ".join(str(s[c]["by_category"].get(k, "不适用")) for c in cfgs) + " |" for k in cats]
    cases = sorted({k for c in cfgs for k in s[c]["per_case"]})
    lines += ["", "## 每题 pass@1", "", "| 题 | " + " | ".join(cfgs) + " |", "|---" * (len(cfgs) + 1) + "|"]
    lines += [f"| {k} | " + " | ".join(str(s[c]["per_case"].get(k, {}).get("pass@1", "不适用")) for c in cfgs) + " |"
              for k in cases]
    # O3：每个 run 一行。审批题一道有两个 run（挂起 + 续跑）；子 run 的花费算在派它的父 run 上
    lines += ["", "## 每个 run 的成本和命中率", "", "| 配置 | 题 | trial | run | stop_reason | 成本 (USD) | 命中率 |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['config']} | {r['case']} | {r.get('trial')} | {u['run_id'][:8]} | {u['stop_reason']} | "
              f"{u['cost_usd']} | {'无输入' if u['hit_rate'] is None else u['hit_rate']} |"
              for r in sorted(records, key=lambda r: (r["config"], r["case"], r.get("trial", 0)))
              for u in r.get("runs", [])]
    return "\n".join(lines) + "\n"
