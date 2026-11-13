"""M6 第 2 步：分词 + BM25 关键词一路 + 向量一路，两路都先按用户过滤。"""

import asyncio

from app.kb_chunk import tokenize
from kb_fakes import FakeEmbedder, make_kb

DOCS = {
    "errors": "错误码 E1024 表示设备离线，请重启网关。",
    "errors2": "错误码 E1025 表示密码错误，请重置密码。",
    "travel": "打车需提供发票，单次上限 500 元。",
}


async def _ingest_all(kb, user="A"):
    for doc_id, text in DOCS.items():
        await kb.ingest(user, doc_id, text)


def test_tokenize_drops_punctuation_and_stopwords():
    toks = tokenize("请问 E1024 的错误码怎么处理？")
    assert "e1024" in toks and "错误码" in toks
    assert not {"的", "怎么", "请问", "？", " "} & set(toks)


def test_keyword_search_hits_exact_code_only():
    async def go():
        kb, _ = make_kb()
        await _ingest_all(kb)
        hits = await kb.keyword_search("A", "E1024 怎么处理", k=5)
        assert [h.doc_id for h in hits] == ["errors"]  # E1025 那块一个词都没命中，不进排名
        assert await kb.keyword_search("A", "天气", k=5) == []
    asyncio.run(go())


def test_bm25_small_corpus_still_scores():
    """2 个块、查询词只在其中 1 个里：Okapi 的 idf=log(1.5)-log(1.5)=0，会一分不得；BM25L 要能命中。"""
    async def go():
        kb, _ = make_kb()
        await kb.ingest("A", "travel", DOCS["travel"])
        await kb.ingest("A", "errors", DOCS["errors"])
        hits = await kb.keyword_search("A", "发票", k=5)
        assert [h.doc_id for h in hits] == ["travel"] and hits[0].score > 0
    asyncio.run(go())


def test_keyword_search_is_filtered_before_scoring():
    """B 的文档既不能出现在 A 的结果里，也不能影响 A 的分数（idf 只用 A 自己的语料算）。"""
    async def go():
        kb, _ = make_kb()
        await _ingest_all(kb, "A")
        before = await kb.keyword_search("A", "发票", k=5)
        for i in range(20):  # B 塞一堆含「发票」的文档，全库建的话「发票」的 idf 会被拉低
            await kb.ingest("B", f"b{i}", f"第{i}张发票的报销说明。")
        after = await kb.keyword_search("A", "发票", k=5)
        assert [h.id for h in after] == [h.id for h in before]
        assert after[0].score == before[0].score
    asyncio.run(go())


def test_vector_search_is_filtered_by_user():
    async def go():
        emb = FakeEmbedder()
        kb, _ = make_kb(emb)
        await kb.ingest("A", "travel", DOCS["travel"])
        await kb.ingest("B", "travel", DOCS["travel"])  # 和 A 完全相同的内容和向量
        emb.vectors["打车报销"] = (await emb.embed(["travel\n" + DOCS["travel"]]))[0]
        hits = await kb.vector_search("A", "打车报销", k=10)  # k 大于 A 的块数也不出错
        assert len(hits) == 1
        assert abs(hits[0].score - 1.0) < 1e-6  # 同一向量：cos = 1
        assert kb.col.get(ids=[hits[0].id])["metadatas"][0]["user_id"] == "A"
        assert await kb.vector_search("C", "打车报销", k=10) == []  # 没有文档的用户
    asyncio.run(go())
