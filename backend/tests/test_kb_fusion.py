"""M6 第 3 步：RRF 公式、rerank 只调一次、阈值卡在 rerank 分数上、全低于阈值返回空。"""

import asyncio

import pytest

from app.kb import Hit
from app.kb_search import RERANK_POOL, rrf, search
from kb_fakes import FakeReranker, make_kb


def h(id_):
    return Hit(id_, id_, "doc", "t", 0, 0.0)


def test_rrf_formula_and_missing_path():
    fused = {f.hit.id: f for f in rrf({"vector": [h("x"), h("y")], "bm25": [h("z"), h("y")]})}
    assert fused["y"].rrf == pytest.approx(1 / 62 + 1 / 62)  # 括号要对：1/(60+r)，不是 1/r+60
    assert fused["x"].rrf == pytest.approx(1 / 61)           # 只在一路出现：另一路记 0
    assert fused["x"].ranks == {"vector": 1} and fused["y"].ranks == {"vector": 2, "bm25": 2}


def test_rrf_prefers_consensus_over_single_first_place():
    order = [f.hit.id for f in rrf({"vector": [h("x"), h("y")], "bm25": [h("z"), h("y")]})]
    assert order[0] == "y"  # 两路都第 2 名 > 某一路第 1 名（k=60 时）
    assert rrf({"vector": [h("x"), h("y")]}, k=0)[0].rrf == 1  # 对照：k=0 时第 1 名独大


DOCS = {
    "e1024": "错误码 E1024 表示设备离线，请重启网关。",
    "e1025": "错误码 E1025 表示密码错误，请重置密码。",
    "travel": "打车需提供发票，单次上限 500 元。",
}
# 0.91 取自 M6 一次真实 bge-reranker-v2-m3 打分（「E1024 怎么处理」对单独一句 E1024，0.9096）；
# 同一次 E1025 那句是 0.0362，O5 把阈值改成 0.04 后改成 0.045，让它刚过默认阈值、仍被 0.1 挡掉——这个值是为测试凑的
RERANK = FakeReranker({"E1024": 0.91, "E1025": 0.045, "发票": 0.0001})


async def _kb():
    kb, _ = make_kb()
    for doc_id, text in DOCS.items():
        await kb.ingest("A", doc_id, text)
    return kb


def test_threshold_is_applied_to_rerank_score():
    async def go():
        kb = await _kb()
        rr = FakeReranker(RERANK.scores)
        res = await search(kb, rr, "A", "E1024 怎么处理")
        out = res.hits
        assert res.status == "ok" and res.flags == []
        # 阈值 0.04（O5 重新校准，原为 0.02）：E1025 的 0.045 过线排第二，打车的 0.0001 被挡掉；按 rerank 分排序
        assert [f.hit.doc_id for f in out] == ["e1024", "e1025"]
        assert out[0].rerank == 0.91 and "bm25" in out[0].ranks
        strict = await search(kb, FakeReranker(RERANK.scores), "A", "E1024 怎么处理", threshold=0.1)
        assert [f.hit.doc_id for f in strict.hits] == ["e1024"]  # 原来的 0.1 会挡掉 E1025
        assert len(rr.calls) == 1 and len(rr.calls[0][1]) <= RERANK_POOL  # 一次调用，宽召回窄精排
    asyncio.run(go())


def test_all_below_threshold_returns_empty_not_top_k():
    async def go():
        kb = await _kb()
        res = await search(kb, FakeReranker({}), "A", "公司股票期权怎么行权")
        assert (res.status, res.hits) == ("not_found", [])
    asyncio.run(go())


def test_without_reranker_returns_fused_order_without_threshold():
    async def go():
        kb = await _kb()
        res = await search(kb, None, "A", "E1024 怎么处理", top_k=2)
        assert (res.status, res.flags) == ("unverified", ["rerank_disabled"])
        hybrid = res.hits
        assert len(hybrid) == 2 and all(f.rerank is None for f in hybrid)
        vec_only = (await search(kb, None, "A", "E1024 怎么处理", use_bm25=False)).hits
        assert all(set(f.ranks) == {"vector"} for f in vec_only)
    asyncio.run(go())


def test_keep_pool_returns_exactly_what_production_reranked():
    """O5 复核（用户选 A）：评测按生产参数调用，keep_pool 带回的整池就是生产送进 rerank 的那批块，
    按 rerank 分降序；hits 和不带 keep_pool 时完全一样，也等于整池按阈值截出来的结果。
    第二轮复核：两次调用各用一个 FakeReranker，比 rerank 实际收到的批次。原来比的 res.pool 和 rr.calls
    出自同一次 keep_pool=True 调用，keep_pool 时偷偷换候选数（cand=80）也测不出来；60 块的语料下
    候选数 20 和 80 选进 rerank 的块不同，这里必须逐项相等。"""
    from app.kb_chunk import embed_text
    from app.kb_eval import production_cut
    from app.kb_search import THRESHOLD

    async def go():
        kb, _ = make_kb()
        for i in range(30):  # 60 块，超过 RERANK_POOL：候选数决定哪些块进 rerank
            await kb.ingest("A", f"d{i}", f"错误码 E{1000 + i} 的处理说明，第一条内容足够长。\n\n第二条：E1024 相关的补充说明文字。")
        scores = {"E1024": 0.5, "E1003": 0.03}
        prod_rr, rr = FakeReranker(scores), FakeReranker(scores)
        plain = await search(kb, prod_rr, "A", "E1024 怎么处理", top_k=5)
        res = await search(kb, rr, "A", "E1024 怎么处理", top_k=5, keep_pool=True)
        assert plain.pool == [] and len(res.pool) == RERANK_POOL
        assert rr.calls == prod_rr.calls  # 送进 rerank 的批次（内容和顺序）和生产调用完全一样
        assert sorted(embed_text(f.hit.title, f.hit.section, f.hit.text) for f in res.pool) == sorted(rr.calls[0][1])
        assert [f.rerank for f in res.pool] == sorted((f.rerank for f in res.pool), reverse=True)
        ids = lambda fs: [f.hit.id for f in fs]
        assert ids(res.hits) == ids(plain.hits) == ids(production_cut(res.pool, THRESHOLD, 5))
        empty = await search(kb, FakeReranker({}), "A", "股票期权", top_k=5, keep_pool=True)
        assert empty.status == "not_found" and len(empty.pool) == RERANK_POOL  # 库外题也带回分数，校准要用
    asyncio.run(go())


def test_hits_follow_rerank_order_not_rrf_order():
    """rerank 分高的排前面，不管它在 RRF 里排第几（复核修复里截断改成先排序再过阈值，钉住顺序）。"""
    async def go():
        kb = await _kb()
        res = await search(kb, None, "A", "E1024 怎么处理")
        assert res.hits[0].hit.doc_id == "e1024"  # RRF 里 e1024 在前
        flipped = await search(kb, FakeReranker({"E1025": 0.9, "E1024": 0.5}), "A", "E1024 怎么处理")
        assert [f.hit.doc_id for f in flipped.hits] == ["e1025", "e1024"]
    asyncio.run(go())
