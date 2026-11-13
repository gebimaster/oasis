"""M6 第 7 步：三种配置的检索评测 + 阈值校准；O5 第 1 步加片段级指标和块统计。.venv\\Scripts\\python scripts/kb_eval.py

语料 = kb_samples/demo + kb_samples/eval_extra（干扰文档），在内存里的独立集合中现建，不碰主库。
标注集 eval/kb_queries.json；结果连同配置快照写到 eval/results/kb-eval-<时间>.json。
真实调用 SiliconFlow：入库每篇文档 1 次 embedding（一篇 ≤ 32 块），每题每配置 1 次 embedding，混合+rerank 每题 1 次 rerank。
片段级指标（span_hit、按字的 context_precision、返回字数）在两种口径上算：送进 rerank 的整池前 3 块，
和生产口径（按 kb_tool 的参数调 search 返回的 hits，即真正送进模型的内容）。两者出自同一次调用、同一个池。

O8：
- 默认模式另外按 kb_queries.json 的 split 分别汇总，hit@1 / span_hit@1 带 Wilson 95% 区间；阈值表分 calib / test。
- --grid：BM25 那一路的 RRF 权重网格（0 = 纯向量）。不开 rerank：开了 rerank 权重只影响谁进前 20，网格基本是平的（复核⑮）；
  看融合排序本身的 span_hit@1 和候选池（前 20）召回，只在 calib 上选，test 上报。
- --rewrite：开查询改写（LLMRewriter，结果缓存在 eval/results/rewrite-cache.json，重跑不重复花钱），
  按几种触发线逐题对比，库内题看答案块进没进生产返回、库外题看拒没拒，分开数改好、改坏，并给配对差的区间。
- --replay <agent-eval 结果名>：回放 agent 评测（配置 A）里模型实际发出的 query，和用户原句对比答案关键词能不能检索回来。
"""

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import chromadb
from dotenv import load_dotenv

BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(BACKEND / ".env", override=True)

from app import kb_chunk, kb_rewrite, kb_search  # noqa: E402
from app.kb import EMBED_MODEL, KnowledgeBase, SiliconFlowEmbedder, open_collection  # noqa: E402
from app.kb_eval import (context_precision, doc_ranking, norm, paired_diff_ci, production_cut,  # noqa: E402
                         returned_chars, span_hit, span_summary, summarize, terms_hit, threshold_table, wilson_ci)
from kb_seed import parse  # noqa: E402

CORPUS = sorted([*(BACKEND / "kb_samples" / "demo").glob("*.md"), *(BACKEND / "kb_samples" / "eval_extra").glob("*.md")])
QUERIES = BACKEND / "eval" / "kb_queries.json"
THRESHOLDS = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5]
USER = "eval"
PROD_TOP_K = 5  # kb_tool 的默认 top_k


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def chunk_stats(col) -> dict:
    """块长分布和带重叠的块数，只看库里存了什么，不依赖切法的实现。
    带重叠 = 同一篇里，这块开头至少 10 字和上一块结尾相同（重叠 carry 的痕迹）。"""
    r = col.get(include=["documents", "metadatas"])
    by_doc: dict[str, list[tuple[int, str]]] = {}
    for d, m in zip(r["documents"], r["metadatas"]):
        by_doc.setdefault(m["doc_id"], []).append((m["chunk_index"], norm(d)))
    lens = sorted(len(norm(d)) for d in r["documents"])
    overlapped = 0
    for chunks in by_doc.values():
        chunks.sort()
        for (_, a), (_, b) in zip(chunks, chunks[1:]):
            overlapped += any(a.endswith(b[:n]) for n in range(10, len(b)))
    return {"n_chunks": len(lens), "len_min": lens[0], "len_median": lens[len(lens) // 2], "len_max": lens[-1],
            "overlapped_chunks": overlapped, "per_doc": {k: len(v) for k, v in sorted(by_doc.items())}}


def snapshot(stats: dict) -> dict:
    """结果离开配置就没法比：模型、切块、融合、阈值、语料和标注集的 hash 都记下来。"""
    return {"embed_model": EMBED_MODEL, "rerank_model": kb_search.RERANK_MODEL,
            "chunker": (kb_chunk.__doc__ or "").strip().splitlines()[0],
            "chunk_size": kb_chunk.SIZE, "chunk_overlap": kb_chunk.OVERLAP, "contextual_embedding": "kb_chunk.embed_text：标题 > 小节\\n块（无小节时 标题\\n块）",
            "bm25": "rank_bm25.BM25L，入库时 jieba.lcut_for_search 分词 + 停用词", "rrf_k": kb_search.RRF_K,
            "rerank_pool": kb_search.RERANK_POOL, "candidates": "max(4*top_k, 20)",
            "production_threshold": kb_search.THRESHOLD, "eval_top_k": PROD_TOP_K, "keep_pool": True,
            "corpus": {p.stem: sha(p) for p in CORPUS}, "chunks": stats, "queries_sha": sha(QUERIES),
            "kb_chunk_sha": sha(BACKEND / "app" / "kb_chunk.py"),
            "chromadb": version("chromadb"), "rank_bm25": version("rank-bm25")}


async def build_kb() -> tuple[KnowledgeBase, dict]:
    kb = KnowledgeBase(open_collection(chromadb.EphemeralClient(), "kb_eval"), SiliconFlowEmbedder())
    for p in CORPUS:
        title, text = parse(p)
        await kb.ingest(USER, p.stem, text, title)
    return kb, chunk_stats(kb.col)


def split_report(rows: list[dict]) -> dict:
    """O8：每个 split 的 hit@1、span_hit@1 和 Wilson 95% 区间（只算库内题）。"""
    out = {}
    for sp in ("calib", "test", "all"):
        pos = [r for r in rows if r["relevant"] and (sp == "all" or r["split"] == sp)]
        hit1 = [1.0 if r["docs"][:1] and r["docs"][0] in r["relevant"] else 0.0 for r in pos]
        sh1 = [r["span_hit@1"] for r in pos]
        out[sp] = {"n": len(pos), "hit@1": round(sum(hit1) / len(hit1), 3) if hit1 else None,
                   "hit@1_ci95": wilson_ci(hit1), "span_hit@1": round(sum(sh1) / len(sh1), 3) if sh1 else None,
                   "span_hit@1_ci95": wilson_ci(sh1)}
    return out


async def grid(kb, queries) -> dict:
    """O8 第四步：BM25 一路的权重网格。不开 rerank，直接看融合排序：span_hit@1 和池子（前 20）里有没有答案块。
    只在 calib 上选（平手保留现在的等权），test 上报。只调 embedding，不调 rerank。"""
    table = {}
    for w in (0.0, 0.25, 0.5, 1.0, 2.0):
        per = {"calib": {"top1": [], "pool": []}, "test": {"top1": [], "pool": []}}
        for q in queries:
            if not q["relevant"]:
                continue
            res = await kb_search.search(kb, None, USER, q["q"], top_k=PROD_TOP_K, keep_pool=True, weights={"bm25": w})
            if [f for f in res.flags if f != "rerank_disabled"]:
                sys.exit(f"评测无效：{q['q']} 出现降级 {res.flags}")
            per[q["split"]]["top1"].append(span_hit(res.pool, q["spans"], 1))
            per[q["split"]]["pool"].append(span_hit(res.pool, q["spans"], kb_search.RERANK_POOL))
        table[w] = {sp: {k: round(sum(v) / len(v), 3) for k, v in d.items()} | {"top1_ci95": wilson_ci(d["top1"])}
                    for sp, d in per.items()}
        table[w]["_raw"] = per
        print(f"bm25 权重 {w}: calib {table[w]['calib']}  test {table[w]['test']}", flush=True)
    best = max(table, key=lambda w: (table[w]["calib"]["top1"], table[w]["calib"]["pool"], w == 1.0))
    diff = paired_diff_ci(table[1.0]["_raw"]["test"]["top1"], table[best]["_raw"]["test"]["top1"])
    return {"table": {w: {k: v for k, v in t.items() if k != "_raw"} for w, t in table.items()},
            "chosen_on_calib": best, "test_at_chosen": table[best]["test"], "test_diff_vs_equal_ci95": diff}


async def rewrite_compare(kb, queries) -> dict:
    """O8 第二步的真实验收：几种触发线（WEAK、每次都改）下逐题对比有无改写。库内题看答案块进没进生产返回，
    库外题看拒没拒（not_found），分开数改好、改坏（复核③：现有数据上 WEAK 只会作用于库外题）。
    同一题没改写时直接沿用基线的结果，不重新检索：rerank 打分有 0.001 的抖动，挨着阈值的块会凭空「改好 / 改坏」（复核⑫）。"""
    rr = kb_search.SiliconFlowReranker()
    rw = kb_rewrite.DiskCachedRewriter(kb_rewrite.LLMRewriter(), BACKEND / "eval" / "results" / "rewrite-cache.json")
    score = lambda q, res: (span_hit(res.hits, q["spans"], PROD_TOP_K) if q["relevant"]
                            else float(res.status == "not_found"))
    base = {q["q"]: score(q, await kb_search.search(kb, rr, USER, q["q"], top_k=PROD_TOP_K)) for q in queries}
    out = {}
    for name, weak in (("weak", kb_search.WEAK), ("always", 1.0)):
        rows = []
        for q in queries:
            res = await kb_search.search(kb, rr, USER, q["q"], top_k=PROD_TOP_K, rewriter=rw, weak=weak)
            n = score(q, res) if res.rewrites else base[q["q"]]
            rows.append({"q": q["q"], "split": q["split"], "in_kb": bool(q["relevant"]), "base": base[q["q"]],
                         "rewrite": n, "rewrites": res.rewrites, "notes": res.notes})
            print(f"  [{name}] {base[q['q']]} → {n}  {q['q'][:24]}  {res.rewrites or res.notes}", flush=True)
        summary = {}
        for part, keep in (("in_kb", True), ("out_of_kb", False)):
            rs = [r for r in rows if r["in_kb"] == keep]
            summary[part] = {"n": len(rs), "rewritten": sum(1 for r in rs if r["rewrites"]),
                             "better": [r["q"] for r in rs if r["rewrite"] > r["base"]],
                             "worse": [r["q"] for r in rs if r["rewrite"] < r["base"]],
                             "diff_ci95": paired_diff_ci([r["base"] for r in rs], [r["rewrite"] for r in rs])}
        out[name] = {"rows": rows, "summary": summary}
    return out


async def replay(kb, name: str) -> dict:
    """O8 第四步：agent 评测里模型实际发出的 query，和用户原句各检索一次，看答案关键词能不能回来。"""
    data = json.loads((BACKEND / "eval" / "results" / f"{name}.json").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in map(json.loads, (BACKEND / "eval" / "agent_cases.jsonl").read_text(encoding="utf-8").splitlines())}
    # 复核⑧：只回放配置 A（B 是纯向量，拿去走混合 + rerank 回放没意义）；一条记录的几次 query 合起来算（模型把两问拆成
    # 两次检索时，单次都拿不全关键词）；用户原句每道题只检索一次；出错的、没有 query 的调用跳过
    rr = kb_search.SiliconFlowReranker()
    orig_cache, rows = {}, []
    for r in data["records"]:
        case = cases.get(r["case"])
        calls = [c for c in r.get("kb_calls") or [] if c.get("query") and not c.get("is_error")]
        if r.get("config") != "A" or not case or case["category"] != "kb_in" or not calls:
            continue
        terms = next((c["args"]["all_of"] for c in case["checks"] if c["name"] == "answer_has"), [])
        if case["id"] not in orig_cache:
            orig = await kb_search.search(kb, rr, USER, case["input"], top_k=PROD_TOP_K)
            orig_cache[case["id"]] = terms_hit(orig.hits, terms)
        hits = []
        for call in calls:
            hits += (await kb_search.search(kb, rr, USER, call["query"], top_k=PROD_TOP_K)).hits
        rows.append({"case": r["case"], "trial": r["trial"], "user_input": case["input"],
                     "model_queries": [c["query"] for c in calls], "model_hit": terms_hit(hits, terms)})
    per_case_orig = list(orig_cache.values())
    return {"rows": rows, "n_records": len(rows), "n_cases": len(per_case_orig),
            "orig_hit": sum(per_case_orig) / len(per_case_orig) if per_case_orig else None,
            "model_hit": sum(r["model_hit"] for r in rows) / len(rows) if rows else None}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--rewrite", action="store_true")
    ap.add_argument("--replay", metavar="AGENT_EVAL_NAME")
    args = ap.parse_args()
    kb, stats = await build_kb()
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    extra = {"grid": grid, "rewrite": rewrite_compare}
    for flag, fn in extra.items():
        if getattr(args, flag):
            res = await fn(kb, queries)
            out = BACKEND / "eval" / "results" / f"kb-{flag}-{datetime.now():%Y%m%d-%H%M%S}.json"
            out.write_text(json.dumps({"config": snapshot(stats), flag: res}, ensure_ascii=False, indent=1, default=str),
                           encoding="utf-8")
            print(f"结果：{out.relative_to(BACKEND)}")
            return
    if args.replay:
        res = await replay(kb, args.replay)
        out = BACKEND / "eval" / "results" / f"kb-replay-{datetime.now():%Y%m%d-%H%M%S}.json"
        out.write_text(json.dumps({"config": snapshot(stats), "source": args.replay, "replay": res},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"回放 {res['n_records']} 条记录（{res['n_cases']} 道题）：用户原句关键词命中 {res['orig_hit']}，"
              f"模型 query 命中 {res['model_hit']}")
        print(f"结果：{out.relative_to(BACKEND)}")
        return
    n_chunks = stats["n_chunks"]
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    rr = kb_search.SiliconFlowReranker()
    # O5 复核（用户选 A）：完全按生产参数调用（top_k=PROD_TOP_K、生产阈值），候选数和 kb_tool 一致；
    # keep_pool=True 带回送进 rerank 的整池。hits = 真正送进模型的内容；排序指标和阈值校准都用整池。
    # 以前用 top_k=20、threshold=-1，候选数变成 max(4*20, 20)=80，语料 48 块时选进 rerank 的 20 块和生产不同
    configs = {"vector": dict(reranker=None, use_bm25=False), "hybrid": dict(reranker=None),
               "hybrid+rerank": dict(reranker=rr)}
    rows = {name: [] for name in configs}
    for q in queries:
        for name, kw in configs.items():
            res = await kb_search.search(kb, user_id=USER, query=q["q"], top_k=PROD_TOP_K, keep_pool=True, **kw)
            bad = [fl for fl in res.flags if fl != "rerank_disabled"]
            if bad:  # 有一路挂了，这次的数字不代表这个配置
                sys.exit(f"评测无效：{name} / {q['q']} 出现降级 {bad}")
            pool = res.pool
            row = {"q": q["q"], "style": q["style"], "split": q["split"], "relevant": q["relevant"],
                   "spans": q["spans"], "docs": doc_ranking(pool)}
            top3 = pool[:3]
            row.update({"span_hit@1": span_hit(pool, q["spans"], 1), "span_hit@3": span_hit(pool, q["spans"], 3),
                        "cp@3": context_precision(top3, q["entries"], q["relevant"]), "chars@3": returned_chars(top3)})
            if name == "hybrid+rerank":
                prod = res.hits
                # 自检：整池按阈值截出来的必须就是生产返回的，否则离线重算的口径就不对
                if [f.hit.id for f in production_cut(pool, kb_search.THRESHOLD, PROD_TOP_K)] != [f.hit.id for f in prod]:
                    sys.exit(f"评测无效：{q['q']} 整池截出来的和生产返回的不一致")
                rel = [f.rerank for f in pool if f.hit.doc_id in q["relevant"]]
                row["best_relevant"] = max(rel) if rel else None
                row["max_score"] = max((f.rerank for f in pool), default=0.0)
                row.update({"cp_prod": context_precision(prod, q["entries"], q["relevant"]),
                            "chars_prod": returned_chars(prod), "n_prod": len(prod),
                            # 含答案片段的块的最高 rerank 分（没进 rerank 池记 None）
                            "span_rerank": max((f.rerank for f in pool if span_hit([f], q["spans"], 1)), default=None),
                            "top3": [{"doc": f.hit.doc_id, "idx": f.hit.chunk_index, "chars": len(norm(f.hit.text)),
                                      "rerank": round(f.rerank, 4)} for f in top3],
                            # 整池（按 rerank 降序）：换阈值时离线重算生产口径，不用再调 API。
                            # rel = 这块里属于相关条目的字数（库外题为 0）
                            "pool": [{"doc": f.hit.doc_id, "idx": f.hit.chunk_index, "rerank": round(f.rerank, 6),
                                      "chars": len(norm(f.hit.text)), "span": bool(span_hit([f], q["spans"], 1)),
                                      "rel": round((context_precision([f], q["entries"], q["relevant"]) or 0)
                                                   * len(norm(f.hit.text)))} for f in pool]})
            rows[name].append(row)

    summary = {name: {"all": summarize(r), **{s: summarize([x for x in r if x["style"] == s]) for s in ("原句", "口语")},
                      "span": span_summary(r), "splits": split_report(r)}
               for name, r in rows.items()}
    rr_rows = rows["hybrid+rerank"]
    pos_best = [r["best_relevant"] for r in rr_rows if r["relevant"]]
    neg_max = [r["max_score"] for r in rr_rows if not r["relevant"]]
    calib = threshold_table(pos_best, neg_max, THRESHOLDS)
    # O5：片段级校准。文档级「相关文档有块过线」会把「过线的是同篇别的条目、答案块没过线」算成答对
    # （周末加班调休题：调休流程块过线，答案块在 0.05 下被挡；004306 里是 0.127 / 0.0437，按生产候选池重跑的
    # 171346 里是 0.124 / 0.045），阈值按这一张定
    span_best = [r["span_rerank"] for r in rr_rows if r["relevant"]]
    calib_span = threshold_table(span_best, neg_max, THRESHOLDS)
    # O8：阈值只在 calib 上选；test 那张只用来看选出来的阈值在没见过的题上掉多少
    by_split = {sp: threshold_table([r["span_rerank"] for r in rr_rows if r["relevant"] and r["split"] == sp],
                                    [r["max_score"] for r in rr_rows if not r["relevant"] and r["split"] == sp], THRESHOLDS)
                for sp in ("calib", "test")}

    out = BACKEND / "eval" / "results" / f"kb-eval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({"config": snapshot(stats), "summary": summary, "threshold": calib,
                               "threshold_span": calib_span, "threshold_span_by_split": by_split,
                               "pos_best": pos_best, "neg_max": neg_max, "rows": rows},
                              ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"语料 {len(CORPUS)} 篇 / {n_chunks} 块，题目 {len(queries)} 条（库外 {len(neg_max)}）")
    print(f"块统计：{stats}")
    print("\n| 配置 | hit@1 | hit@3 | MRR@5 | 原句 hit@1 | 口语 hit@1 | 库外拒答 |\n|---|---|---|---|---|---|---|")
    for name, s in summary.items():
        refuse = "0/4（无阈值，永远返回结果）" if name != "hybrid+rerank" else "见阈值表"
        print(f"| {name} | {s['all']['hit@1']} | {s['all']['hit@3']} | {s['all']['mrr@5']} | "
              f"{s['原句']['hit@1']} | {s['口语']['hit@1']} | {refuse} |")
    print("\n| 配置 | span_hit@1 | span_hit@3 | cp@3 | 字数@3 | cp 生产口径 | 字数 生产口径 |\n|---|---|---|---|---|---|---|")
    for name, s in summary.items():
        sp = s["span"]
        print(f"| {name} | {sp['span_hit@1']} | {sp['span_hit@3']} | {sp['cp@3']} | {sp['chars@3']} | "
              f"{sp.get('cp_prod', '-')} | {sp.get('chars_prod', '-')} |")
    print(f"\n库外题生产口径送进模型的字数：{[r['chars_prod'] for r in rr_rows if not r['relevant']]}")
    print("\n各题（hybrid+rerank）：含片段块的 rerank 分 / cp 生产口径 / 生产口径块数 / top3 (文档, 第几块, 字数)")
    for r in rr_rows:
        if r["relevant"]:
            sr = None if r["span_rerank"] is None else round(r["span_rerank"], 4)
            cp = None if r["cp_prod"] is None else round(r["cp_prod"], 3)
            print(f"  {r['q'][:20]:20} {sr} / {cp} / {r['n_prod']} / {[(x['doc'], x['idx'], x['chars']) for x in r['top3']]}")
    print("\n| 阈值 | 库内答对（文档级） | 库内答对（片段级） | 库外拒对 |\n|---|---|---|---|")
    for c, cs in zip(calib, calib_span):
        print(f"| {c['threshold']} | {c['pos_answered']} | {cs['pos_answered']} | {c['neg_refused']} |")
    print("\n库内题相关文档最高分：", sorted(round(s, 4) if s is not None else None for s in pos_best
                                   if s is not None), "缺失", pos_best.count(None))
    print("库外题最高分：", sorted(round(s, 4) for s in neg_max))
    print("\n各配置 top1 不对的题：")
    for name, r in rows.items():
        for x in r:
            if x["relevant"] and x["docs"][:1] != x["relevant"][:1]:
                print(f"  {name:14} {x['q']} → {x['docs'][:3]}（应为 {x['relevant']}）")
    print("\n| 配置 | split | n | hit@1 (95% CI) | span_hit@1 (95% CI) |\n|---|---|---|---|---|")
    for name, s in summary.items():
        for sp, v in s["splits"].items():
            print(f"| {name} | {sp} | {v['n']} | {v['hit@1']} {v['hit@1_ci95']} | {v['span_hit@1']} {v['span_hit@1_ci95']} |")
    print(f"\n结果：{out.relative_to(BACKEND)}")


if __name__ == "__main__":
    asyncio.run(main())
