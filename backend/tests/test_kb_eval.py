"""M6 第 7 步：评测指标。钉住「截到 k」「文档级去重」「库外题不进 hit」这三个容易算错的地方。"""

from app.kb import Hit
from app.kb_eval import doc_ranking, hit_at, mrr_at, summarize, threshold_table
from app.kb_search import Fused


def f(doc_id, i=0):
    return Fused(Hit(f"{doc_id}-{i}", "", doc_id, "t", i, 0.0))


def test_doc_ranking_dedupes_chunks_of_same_doc():
    assert doc_ranking([f("a"), f("a", 1), f("b"), f("a", 2), f("c")]) == ["a", "b", "c"]


def test_metrics_are_truncated_at_k():
    docs = ["x", "y", "z", "w", "v", "rel"]  # 相关文档排第 6
    assert hit_at(docs, ["rel"], 3) == 0 and mrr_at(docs, ["rel"], 5) == 0  # 不截 k 的话会记 1/6
    assert mrr_at(["x", "rel"], ["rel"], 5) == 0.5 and hit_at(["x", "rel"], ["rel"], 1) == 0


def test_out_of_kb_queries_do_not_count_in_hit():
    rows = [{"relevant": ["a"], "docs": ["a"]}, {"relevant": [], "docs": ["b"]}]
    assert summarize(rows) == {"n": 1, "hit@1": 1.0, "hit@3": 1.0, "mrr@5": 1.0}


def test_threshold_table_counts_missing_relevant_as_not_answered():
    t = threshold_table([0.9, 0.05, None], [0.001, 0.2], [0.1])
    assert t == [{"threshold": 0.1, "pos_answered": "1/3", "neg_refused": "1/2"}]


# ---- O5 第 1 步：片段级指标（答案片段 span_hit + 按字的 context_precision） ----

import json
from pathlib import Path

from app.kb_chunk import split_text
from app.kb_eval import context_precision, norm, production_cut, span_hit, span_summary

BACKEND = Path(__file__).parent.parent
QUERIES = json.loads((BACKEND / "eval" / "kb_queries.json").read_text(encoding="utf-8"))


def _corpus() -> dict[str, str]:
    """和 scripts/kb_seed.py 的 parse 一样剥掉第一行「# 标题」，只留正文（评测和种子入库的都是正文）。"""
    out = {}
    for p in [*(BACKEND / "kb_samples" / "demo").glob("*.md"), *(BACKEND / "kb_samples" / "eval_extra").glob("*.md")]:
        first, _, body = p.read_text(encoding="utf-8").partition("\n")
        out[p.stem] = body.strip()
    return out


def t(doc_id, text, i=0, rerank=None):
    return Fused(Hit(f"{doc_id}-{i}", text, doc_id, "t", i, 0.0), rerank=rerank)


E1024 = "E1024：设备离线。先检查电源。"
E1025 = "E1025：认证失败。重置设备密钥。"
E2001 = "E2001：固件过低。升级到 3.2。"


def test_span_hit_any_span_within_k_and_normalized():
    hits = [t("x", "无关内容"), t("ec", f"{E1024}\n{E1025}")]
    assert span_hit(hits, ["重置设备密钥"], 1) == 0.0  # 片段在第 2 块，截到 k=1 看不到
    assert span_hit(hits, ["不存在的话", "重置设备密钥"], 2) == 1.0  # 用户选「任一」
    # 用户选 NFKC + 去空白：全角数字 / 全角空格 / 换行都不影响命中
    assert span_hit([t("ec", "升级到 ３．２\n或以上")], ["升级到3.2 或以上"], 1) == 1.0


def test_context_precision_counts_characters_not_chunks():
    merged = f"{E1024}\n{E1025}\n{E2001}"  # 按累计长度合出来的块
    # 按块算会是 1/1；按字算，只有 E1025 那段算相关
    assert context_precision([t("ec", merged)], [E1025], ["ec"]) == len(norm(E1025)) / len(norm(merged))
    assert context_precision([t("ec", E1025)], [E1025], ["ec"]) == 1.0
    two = context_precision([t("ec", E1025), t("ec", E1024, 1)], [E1025], ["ec"])
    assert two == len(norm(E1025)) / len(norm(E1025 + E1024))


def test_context_precision_long_entry_split_and_overlap():
    entry = "第一句很长的规定。第二句很长的规定。第三句很长的规定。"
    a, b = "第一句很长的规定。第二句", "第二句很长的规定。第三句很长的规定。"  # 长条目被切成两块，b 带重叠
    assert context_precision([t("d", a), t("d", b)], [entry], ["d"]) == 1.0  # 块整个在条目里
    tail = "第三句很长的规定。下一条无关。"  # 首尾相接：前 9 字属于条目
    assert context_precision([t("d", tail)], [entry], ["d"]) == len("第三句很长的规定。") / len(norm(tail))


def test_context_precision_other_doc_and_empty():
    assert context_precision([t("x2", E1025)], [E1025], ["ec"]) == 0.0  # 别的文档里的同样文字不算
    assert context_precision([], [E1025], ["ec"]) is None  # 一块都没返回：没有分母


def test_production_cut_and_summary():
    hits = [t("a", "甲", rerank=0.9), t("b", "乙", rerank=0.5), t("c", "丙", rerank=0.01)]
    assert [f.hit.doc_id for f in production_cut(hits, 0.02, 5)] == ["a", "b"]
    assert [f.hit.doc_id for f in production_cut(hits, 0.02, 1)] == ["a"]
    rows = [{"spans": ["x"], "span_hit@3": 1.0, "cp_prod": 0.5}, {"spans": ["y"], "span_hit@3": 0.0, "cp_prod": None},
            {"spans": [], "span_hit@3": 0.0, "cp_prod": 0.0}]  # 库外题不进统计
    assert span_summary(rows) == {"n": 2, "span_hit@3": 0.5, "cp_prod": 0.5, "cp_prod_none": 1}


def test_annotations_are_valid_substrings_of_the_corpus():
    """用户定的两条约束：片段在全语料里只出现一次；片段是相关条目的子串。另外条目必须是相关文档正文的子串。"""
    docs = {k: norm(v) for k, v in _corpus().items()}
    for q in QUERIES:
        assert bool(q["spans"]) == bool(q["relevant"]) == bool(q["entries"]), q["q"]
        for e in q["entries"]:
            assert any(norm(e) in docs[d] for d in q["relevant"]), (q["q"], e[:10])
        for s in q["spans"]:
            assert sum(d.count(norm(s)) for d in docs.values()) == 1, (q["q"], s)
            assert any(norm(s) in norm(e) for e in q["entries"]), (q["q"], s)


def test_annotations_survive_any_chunking():
    """与切法无关：同一道题、同一份标注，在三种切法下都能命中片段、算出 precision。
    按块 id 标注的话，一换切法就失效。"""
    docs = _corpus()
    chunkers = {
        "当前 split_text": lambda body: split_text(body),
        "小块 60/10": lambda body: split_text(body, size=60, overlap=10),
        "一段一块": lambda body: [p.strip() for p in body.split("\n\n") if p.strip()],
    }
    for name, chunk in chunkers.items():
        for q in (q for q in QUERIES if q["relevant"]):
            hits = [t(d, c, i) for d in q["relevant"] for i, c in enumerate(chunk(docs[d]))]
            assert span_hit(hits, q["spans"], len(hits)) == 1.0, (name, q["q"])
            with_span = [f for f in hits if span_hit([f], q["spans"], 1)]
            cp = context_precision(with_span, q["entries"], q["relevant"])
            assert cp is not None and cp > 0, (name, q["q"])
            paras = {norm(p) for d in q["relevant"] for p in docs[d].split("\n\n")}
            if name == "一段一块" and all(norm(e) in paras for e in q["entries"]):
                assert cp == 1.0, q["q"]  # 条目就是整段：只含片段的那块应该全是相关内容（列表项条目除外）


# ---------------- O8 第三步：划分、置信区间、库外题去重 ----------------

def _queries():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parent.parent / "eval" / "kb_queries.json").read_text(encoding="utf-8"))


def test_split_in_file_matches_the_function():
    """划分写进了 kb_queries.json 的 split 字段；和函数算出来的不一致，说明有人手改了文件或改了算法。"""
    from app.kb_eval import assign_splits
    q = _queries()
    assert [x["split"] for x in q] == assign_splits(q)
    for key in {(bool(x["relevant"]), x["style"]) for x in q}:  # 每层 calib 和 test 最多差 1
        layer = [x["split"] for x in q if (bool(x["relevant"]), x["style"]) == key]
        assert abs(layer.count("calib") - layer.count("test")) <= 1


def test_split_ignores_file_order():
    from app.kb_eval import assign_splits
    q = _queries()
    assert assign_splits(list(reversed(q))) == list(reversed(assign_splits(q)))


def test_wilson_ci_does_not_collapse_and_is_wide_for_small_samples():
    """复核⑦：百分位 bootstrap 在全对时塌成 (1.0, 1.0)；Wilson 给出约 0.72–1.0。"""
    from app.kb_eval import wilson_ci
    assert wilson_ci([1.0] * 10) == (0.722, 1.0)
    lo, hi = wilson_ci([1.0] * 7 + [0.0] * 3)
    assert lo < 0.45 and hi > 0.85 and wilson_ci([]) is None


def test_paired_diff_ci_uses_per_question_pairs():
    from app.kb_eval import paired_diff_ci
    a, b = [0.0, 1.0] * 10, [1.0, 1.0] * 10  # 同一批题：b 在一半的题上更好、其余一样
    lo, hi = paired_diff_ci(a, b)
    assert 0 < lo <= 0.5 <= hi and paired_diff_ci(a, b) == (lo, hi)
    assert paired_diff_ci(a, b[:-1]) is None
    spread_a, spread_b = [0.0] * 12, [1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0]
    assert paired_diff_ci(spread_a, spread_b, n=40) == paired_diff_ci(spread_a, spread_b, n=40)  # 固定 seed：少量重抽也可复现


def test_terms_hit_matches_whole_numbers():
    from app.kb import Hit
    from app.kb_eval import terms_hit
    from app.kb_search import Fused
    hits = [Fused(Hit("a", "病假 150 天以内全额发放", "d", "t", 0, 0.0))]
    assert not terms_hit(hits, ["15"]) and terms_hit(hits, ["150", ["全额", "全部"]])


def test_agent_kb_out_cases_are_not_calibration_questions():
    """阈值是在 kb_queries 的库外题上调的；agent_cases 的 kb_out 题如果是同一批问题，端到端的拒答率就偏乐观。
    O8 之前三道题完全重合（股票期权、食堂、停车），已换成新问题。"""
    import json
    from pathlib import Path
    from app.kb_eval import norm
    cases = [json.loads(line) for line in
             (Path(__file__).parent.parent / "eval" / "agent_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    kb_out = {norm(c["input"]) for c in cases if c["category"] == "kb_out"}
    assert kb_out and not kb_out & {norm(x["q"]) for x in _queries()}



# 已知的近似重复：kb_in 端到端题和阈值校准用的标注集是同一批问题（O8 复核⑥）。阈值 0.04 是在全部 23 题上选的，
# 所以端到端 kb_in 的答对率偏乐观。这里把它们列出来：新出现的重复（尤其是库外题）会让测试变红
KNOWN_KB_IN_OVERLAP = {"kb-taxi", "kb-hotel", "kb-meal", "kb-e1025", "kb-sick", "kb-paternity"}


def test_near_duplicates_between_agent_cases_and_calibration_set():
    import json
    from pathlib import Path
    from app.kb_eval import bigram_jaccard
    cases = [json.loads(line) for line in
             (Path(__file__).parent.parent / "eval" / "agent_cases.jsonl").read_text(encoding="utf-8").splitlines()]
    qs = [x["q"] for x in _queries()]
    dup = {c["id"] for c in cases if c["category"] in ("kb_in", "kb_out")
           if max(bigram_jaccard(c["input"], q) for q in qs) >= 0.3}
    assert dup == KNOWN_KB_IN_OVERLAP
