"""M6 第 4 步：显式降级。每种故障都要在 status / flags 里看得出来，不能和「真的没有」混在一起。"""

import asyncio

from app.kb_search import search
from kb_fakes import FakeReranker, make_kb

DOCS = {
    "e1024": "错误码 E1024 表示设备离线，请重启网关。",
    "travel": "打车需提供发票，单次上限 500 元。",
}


async def _kb():
    kb, emb = make_kb()
    for doc_id, text in DOCS.items():
        await kb.ingest("A", doc_id, text)
    return kb, emb


def test_rerank_failure_is_unverified_not_empty():
    """反例：rerank 失败吞成 []，和「全都不相关」分不清。我们回退 RRF 结果并标记。"""
    async def go():
        kb, _ = await _kb()
        rr = FakeReranker({"E1024": 0.91})
        rr.fail = True
        res = await search(kb, rr, "A", "E1024 怎么处理")
        assert res.status == "unverified" and res.flags == ["rerank_failed"]
        assert res.hits and res.hits[0].hit.doc_id == "e1024" and res.hits[0].rerank is None
    asyncio.run(go())


def test_vector_failure_degrades_to_bm25_with_flag():
    async def go():
        kb, emb = await _kb()
        emb.fail = True  # 查询时 embedding 接口挂了
        res = await search(kb, FakeReranker({"E1024": 0.91}), "A", "E1024 怎么处理")
        assert res.status == "ok" and res.flags == ["vector_failed"]
        assert [f.hit.doc_id for f in res.hits] == ["e1024"] and set(res.hits[0].ranks) == {"bm25"}
    asyncio.run(go())


def test_not_found_with_vector_failed_keeps_flag():
    """用户选④(b)：向量一路挂了又没找到，只能说「没搜全」。flags 必须带到结果里，不能只进日志。"""
    async def go():
        kb, emb = await _kb()
        emb.fail = True
        res = await search(kb, FakeReranker({}), "A", "发票")
        assert (res.status, res.flags) == ("not_found", ["vector_failed"])
    asyncio.run(go())


def test_all_paths_failed_is_error():
    async def go():
        kb, emb = await _kb()
        emb.fail = True
        res = await search(kb, FakeReranker({}), "A", "E1024", use_bm25=False)  # 只剩向量一路，也挂了
        assert (res.status, res.hits, res.flags) == ("error", [], ["vector_failed"])
    asyncio.run(go())


def test_user_without_documents_is_plain_not_found():
    async def go():
        kb, _ = await _kb()
        res = await search(kb, FakeReranker({}), "nobody", "E1024")
        assert (res.status, res.flags) == ("not_found", [])
    asyncio.run(go())
