"""知识库检索管线（M6 第 3 步）：两路召回 → 应用层 RRF → 一次 rerank → 阈值卡在 rerank 分数上。

常见的坑：阈值卡在 RRF 融合后的分数上（归一化后第一名恒为 1.0，或者分数全挤在一个窄区间里），
rerank 后又不设阈值，无关查询也一定返回 top_k 条。
我们的 RRF 只负责排序、不当「相关度」；是否相关只看 rerank（cross-encoder 把查询和块放在一起读）。
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from app.kb import Hit, KnowledgeBase
from app.kb_chunk import embed_text

log = logging.getLogger(__name__)
RRF_K = 60
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_POOL = 20     # 融合后取前 20 个交给 rerank：宽召回、窄精排，一次 API 调用
# 第 7 步标注集校准（eval/results/kb-eval-20261030-022854.json）：库内相关文档最高分最低 0.0321，
# 库外题最高 0.0093；原来的 0.1 会把「周末加班调休」错拒成「知识库里没有」。用户选 0.02（两者之间）。
# 代价：E1025 这类 0.036 的弱相关块也会过线；样本只有 4 道库外题，换语料要重新校准
# O5 按条目切块后重新校准（改按片段级：含答案片段的块过线才算答对）。初次校准用的 004306 候选池和生产不一致
# （评测用 top_k=20，候选数 80），复核后按生产参数重跑 eval/results/kb-eval-20261105-171346.json：
# 库内含片段块最低 0.045（周末加班调休），库外最高 0.0464，两头已经交叉，没有能全对的阈值。0.02 时库外只拒 2/4；
# 用户先选 0.05，片段级复核发现它会挡掉周末题的答案块（文档级校准表看不出来），改选 0.04：库内 19/19、库外 3/4。
# 重跑结论不变，但 0.04 离最弱的答案块只有 0.005，同一块两次打分能差 0.001
# O8 复核⑥：原来是在全部 23 题上选的。按 split 只用 calib 重选（同一份 171346）：0.0243 < t ≤ 0.045 都能 calib 全对，
# 0.04 在里面，不改；留出的 test 上库内 9/9、库外只拒对 1/2（0.0464 那道过线）。这才是没见过的题上的数
THRESHOLD = 0.04
# O8 第二步：查询改写（只在第一轮弱的时候做，见 search）。WEAK 是拍的，O8 复核③用存下来的生产口径评测
# （kb-eval-20261105-171346）核对过：库内 19 题的最高分都 ≥ 0.1（连改写的动机题「周末干活换假」也是 0.124，
# 它的答案块只有 0.045），一题都不会触发；库外 4 题（0.001–0.046）全会触发。也就是说在现有数据上改写只会作用于
# 库外题，而多补候选会提高「最高分过线」的概率，库外误答可能变多。所以默认关着，--rewrite 分库内 / 库外数改好改坏
WEAK = 0.1
MAX_REWRITES = 2        # 每次最多加几条改写；每条都跑向量 + BM25 两路，检索调用最多 2 + 2×2 = 6 次
# 改写最多往池子里补几个新候选。不和原句抢 RERANK_POOL 的名额：初版把改写各路以 0.5 权重和原句一起 RRF 再截前 20，
# 原句召回满 20 条时改写找回的块一个都进不了池（测试先红在这里）。现在第一轮的整池原样保留，改写只往上加
REWRITE_POOL = 10
REWRITE_TIMEOUT = 8.0   # 改写器超时（秒）：超时 / 出错都退回不改写的结果，带 rewrite_failed 标记


class Rewriter(Protocol):
    name: str
    async def rewrite(self, query: str) -> list[str]: ...


class Reranker(Protocol):
    name: str
    async def rerank(self, query: str, docs: list[str]) -> list[float]: ...


class SiliconFlowReranker:
    name = RERANK_MODEL

    def __init__(self, base_url: str | None = None, api_key: str | None = None, timeout: float = 15):
        self.base_url = base_url or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        self.api_key = api_key or os.environ["SILICONFLOW_API_KEY"]
        self.timeout = timeout

    async def rerank(self, query: str, docs: list[str]) -> list[float]:
        """返回与 docs 同序的分数。接口按相关度降序返回 results[].index，这里按 index 放回原位。"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/rerank",
                                  headers={"Authorization": f"Bearer {self.api_key}"},
                                  json={"model": self.name, "query": query, "documents": docs,
                                        "top_n": len(docs), "return_documents": False})
            r.raise_for_status()
        scores = [0.0] * len(docs)
        for x in r.json()["results"]:
            scores[x["index"]] = x["relevance_score"]
        return scores


@dataclass
class Fused:
    hit: Hit
    rrf: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # 各路名次，从 1 开始；没出现的路不在字典里
    rerank: float | None = None


def rrf(paths: dict[str, list[Hit]], k: int = RRF_K, weights: dict[str, float] | None = None) -> list[Fused]:
    """score = Σ w/(k + rank)。只出现在一路的块，另一路那项记 0（不出现就不加）。
    k=60 让第 1 名和第 2 名差得很少（1/61 vs 1/62），两路都靠前的块才胜出。
    O8：weights 按路名给权重，没给的记 1；权重为 0 的路整路不参与（只出现在那一路的块也不进结果），
    所以 bm25 权重 0 就退化成纯向量。"""
    fused: dict[str, Fused] = {}
    for name, hits in paths.items():
        w = (weights or {}).get(name, 1.0)
        if w == 0:
            continue
        for rank, h in enumerate(hits, start=1):
            f = fused.setdefault(h.id, Fused(h))
            f.rrf += w / (k + rank)
            f.ranks[name] = rank
    return sorted(fused.values(), key=lambda f: -f.rrf)  # sorted 稳定：同分时先出现的在前


@dataclass
class SearchResult:
    """status 四种（用户第 4 步选定）：
    ok         rerank 校验过，hits 都过了阈值
    unverified 有结果但没经过 rerank（rerank 挂了，或评测配置里关掉了），没有阈值可用，不能说「没找到」
    not_found  rerank 正常工作、全部低于阈值，或者根本没有候选；唯一能说「库里没有」的状态
    error      两路召回都挂了
    flags 记录每一处降级（vector_failed / bm25_failed / rerank_failed / rerank_disabled）。
    not_found 带着 vector_failed 等标记时，只能说「这次没搜全」，不能说「没有」。"""
    status: str
    hits: list[Fused] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    # O5 复核（用户选 A）：keep_pool=True 时带回送进 rerank 的整池（有 rerank 分时按分降序，否则 RRF 顺序）。
    # 评测用它：hits 就是生产真正送进模型的内容，排序指标和阈值校准用同一个池，候选数和生产一致
    pool: list[Fused] = field(default_factory=list)
    # O8：查询改写的痕迹。不放进 flags：flags 表示「这次没搜全」，会让 not_found 不能说「没有」；
    # 改写失败时原句那一路照常完整跑过，结果和不改写一样，不影响「能不能说没有」
    rewrites: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # rewritten / rewrite_failed / rewrite_empty / rewrite_<路>_failed


async def _recall(kb: KnowledgeBase, user_id: str, queries: dict[str, str], cand: int,
                  use_bm25: bool) -> tuple[dict[str, list[Hit]], list[str]]:
    """queries：后缀 → 查询（原句的后缀是空串）。每条查询跑向量和 BM25 两路，路名 vector{后缀} / bm25{后缀}。"""
    calls = {}
    for suffix, q in queries.items():
        calls[f"vector{suffix}"] = kb.vector_search(user_id, q, cand)
        if use_bm25:
            calls[f"bm25{suffix}"] = kb.keyword_search(user_id, q, cand)
    # return_exceptions：一路挂了不拖垮另一路（只记日志不够，这里还带降级标记）
    results = await asyncio.gather(*calls.values(), return_exceptions=True)
    paths, flags = {}, []
    for name, r in zip(calls, results):
        if isinstance(r, Exception):
            flags.append(f"{name}_failed")
            log.warning("知识库 %s 一路失败：%s", name, type(r).__name__)  # 不记异常原文，防止回显凭据
        else:
            paths[name] = r
    return paths, flags


async def search(kb: KnowledgeBase, reranker: Reranker | None, user_id: str, query: str,
                 top_k: int = 5, use_bm25: bool = True, threshold: float = THRESHOLD,
                 keep_pool: bool = False, rewriter: Rewriter | None = None,
                 weights: dict[str, float] | None = None, weak: float | None = None) -> SearchResult:
    """三种配置（第 7 步对比用）：只向量 = use_bm25=False, reranker=None；混合 = reranker=None；
    混合+rerank = 默认。O8：rewriter 非空时，第一轮最高 rerank 分低于 weak 才改写（weak=1.0 就是每次都改），
    见 _with_rewrites；没有候选时直接 not_found、不改写；没有 reranker 就没有「弱」的信号，也不改写。
    weights 是第四步的加权 RRF，按路名前缀（vector / bm25）同样作用于改写那几路。"""
    cand = max(4 * top_k, 20)
    paths, flags = await _recall(kb, user_id, {"": query}, cand, use_bm25)
    if not paths:  # 用户选①(b)：所有路都挂了才是故障；只挂一路算部分降级
        return SearchResult("error", flags=flags)
    pool = rrf(paths, weights=weights)[:RERANK_POOL]
    if not pool:
        return SearchResult("not_found", flags=flags)
    kept_pool = pool if keep_pool else []
    if reranker is None:
        return SearchResult("unverified", pool[:top_k], flags + ["rerank_disabled"], kept_pool)
    try:
        # rerank 看到的和 embedding 用的是同一份文本（O5 用户选 Q7=B：小节标题也进）
        scores = await reranker.rerank(query, [embed_text(f.hit.title, f.hit.section, f.hit.text) for f in pool])
    except Exception as e:  # 不吞成空列表（那样和「全都不相关」分不清），带标记返回粗排结果（用户选 A）
        log.warning("rerank 失败，回退 RRF 排序：%s", type(e).__name__)
        return SearchResult("unverified", pool[:top_k], flags + ["rerank_failed"], kept_pool)
    for f, s in zip(pool, scores):
        f.rerank = s
    ranked = sorted(pool, key=lambda f: -f.rerank)
    kept = [f for f in ranked if f.rerank >= threshold][:top_k]
    # 全部低于阈值 → not_found，不硬凑 top_k
    res = SearchResult("ok" if kept else "not_found", kept, flags, ranked if keep_pool else [])
    if rewriter is not None and ranked[0].rerank < (WEAK if weak is None else weak):
        return await _with_rewrites(kb, reranker, rewriter, user_id, query, top_k, use_bm25, threshold,
                                    keep_pool, res, ranked, weights)
    return res


async def _with_rewrites(kb, reranker, rewriter, user_id, query, top_k, use_bm25, threshold, keep_pool,
                         first: SearchResult, first_ranked: list[Fused], weights=None) -> SearchResult:
    """O8 第二步：改写的各路先自己 RRF，取不在第一轮池里的前 REWRITE_POOL 个，补进第一轮的池；rerank 永远用原句打分、阈值照旧。
    - 第一轮的整池原样保留（约束：原句那一路永远保留），改写只补召回
    - rerank 用原句：改写只负责把候选找回来，相不相关仍按用户的原话判，改写跑偏也不会让无关块过线
    - 第一轮已经打过分的块不再送 rerank（同一对 query/块，分数不变），只给新进池的块打分
    - 改写器超时、出错、没给出新查询，或改写后的 rerank 出错：原样返回第一轮的结果，只加一条 note"""
    try:
        got = await asyncio.wait_for(rewriter.rewrite(query), REWRITE_TIMEOUT)
    except Exception as e:  # 超时、限流、解析失败：不改写
        log.warning("查询改写失败，按原句检索：%s", type(e).__name__)
        first.notes.append("rewrite_failed")
        return first
    extra = list(dict.fromkeys(q.strip() for q in got if q and q.strip() and q.strip() != query))[:MAX_REWRITES]
    if not extra:
        first.notes.append("rewrite_empty")
        return first
    more, more_flags = await _recall(kb, user_id, {f"@{i}": q for i, q in enumerate(extra, 1)},
                                     max(4 * top_k, 20), use_bm25)
    known = {f.hit.id: f.rerank for f in first_ranked}
    w = {name: (weights or {}).get(name.split("@")[0], 1.0) for name in more}  # 复核⑭：权重 0 的路改写里也不参与
    new = [f for f in rrf(more, weights=w) if f.hit.id not in known][:REWRITE_POOL]
    if not new:  # 改写没找回任何新候选：结果和第一轮一样
        first.rewrites, first.notes = extra, ["rewrite_no_new"] + [f"rewrite_{x}" for x in more_flags]
        return first
    pool = first_ranked + new
    try:
        texts = [embed_text(f.hit.title, f.hit.section, f.hit.text) for f in new]
        scores = await reranker.rerank(query, texts)
    except Exception as e:
        log.warning("改写后的 rerank 失败，用第一轮结果：%s", type(e).__name__)
        first.notes.append("rewrite_failed")
        return first
    for f, s in zip(new, scores):
        f.rerank = s
    ranked = sorted(pool, key=lambda f: -f.rerank)
    kept = [f for f in ranked if f.rerank >= threshold][:top_k]
    # 改写那几路挂了只记 note：原句那几路是完整的，不让结果变成「没搜全」
    return SearchResult("ok" if kept else "not_found", kept, first.flags, ranked if keep_pool else [],
                        rewrites=extra, notes=["rewritten"] + [f"rewrite_{x}" for x in more_flags])
