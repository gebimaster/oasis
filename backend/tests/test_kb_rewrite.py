"""O8 第二步：查询改写（多路查询进同一个 RRF，rerank 用原句打分）+ 第四步的加权 RRF。全部用假实现，不连网。"""

import asyncio

import pytest

from app import kb_rewrite, kb_search
from app.kb import Hit
from app.kb_search import rrf, search
from kb_fakes import FakeEmbedder, FakeReranker, make_kb

Q = "周末来公司干活了能换几天假"          # 用户原句：口语，和文档用词对不上
REWRITE = "休息日加班 调休"               # 改写后的书面说法
OVERTIME = "法定休息日加班的，可以申请调休，按一比一折算。"
LEAVE = "员工因私事请假需提前一天在系统里提交。"


def vec(e1=0.0, e2=0.0, e3=0.0):
    return [e1, e2, e3, 0.0, 0.0, 0.0, 0.0, 0.1]


class FakeRewriter:
    name = "fake"

    def __init__(self, out=(REWRITE,), exc=None, delay=0.0):
        self.out, self.exc, self.delay, self.calls = list(out), exc, delay, []

    async def rewrite(self, query):
        self.calls.append(query)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.out


class CountingKB:
    """包一层，数检索调用次数。"""

    def __init__(self, kb):
        self.kb, self.calls = kb, []

    async def vector_search(self, user_id, query, k):
        self.calls.append(("vector", query))
        return await self.kb.vector_search(user_id, query, k)

    async def keyword_search(self, user_id, query, k):
        self.calls.append(("bm25", query))
        return await self.kb.keyword_search(user_id, query, k)


async def corpus():
    """32 块：调休那块的向量离原句最远（原句一路的前 20 名里没有它），离改写最近；
    请假那块离原句最近、离改写最远（改写一路的前 20 名里没有它）。30 块噪声都靠近改写。"""
    vectors = {Q: vec(e1=1), REWRITE: vec(e1=-0.3, e2=1),
               f"overtime\n{OVERTIME}": vec(e1=-0.3, e2=1), f"leave\n{LEAVE}": vec(e1=1, e2=-0.3)}
    noise = {f"n{i}": f"第{i}号设备的保养记录，按季度更换滤网。" for i in range(30)}
    vectors |= {f"{k}\n{t}": vec(e2=1, e3=0.01 * i) for i, (k, t) in enumerate(noise.items())}
    kb, _ = make_kb(FakeEmbedder(vectors))
    for doc_id, text in {"overtime": OVERTIME, "leave": LEAVE, **noise}.items():
        await kb.ingest("A", doc_id, text)
    return kb


def rerank():  # 不看 query 的假 rerank：调休 0.5；请假 0.06（过阈值 0.04、但低于 WEAK 0.1，算弱）
    return FakeReranker({"调休": 0.5, "请假": 0.06})


def test_rewrite_brings_back_the_answer_and_keeps_the_original_path():
    async def go():
        kb, rr, rw = CountingKB(await corpus()), rerank(), FakeRewriter()
        base = await search(kb.kb, rerank(), "A", Q)
        assert [f.hit.doc_id for f in base.hits] == ["leave"]  # 不改写：调休那块根本没进候选池
        res = await search(kb, rr, "A", Q, rewriter=rw)
        assert [f.hit.doc_id for f in res.hits] == ["overtime", "leave"]  # 调休成了第一；原句找到的请假还在
        assert res.rewrites == [REWRITE] and res.notes == ["rewritten"] and res.flags == []
        assert all(q == Q for q, _ in rr.calls)          # rerank 永远用原句打分
        assert len(rr.calls) == 2 and not any("请假" in d for d in rr.calls[1][1])  # 打过分的不再送 rerank
        assert not set(rr.calls[1][1]) & set(rr.calls[0][1])  # 噪声块两路都召回了，也只打一次分
        assert len(kb.calls) <= 2 + 2 * kb_search.MAX_REWRITES  # 检索调用有上限
        assert rw.calls == [Q]
    asyncio.run(go())


def test_strong_first_pass_does_not_rewrite():
    async def go():
        kb, rw = await corpus(), FakeRewriter()
        res = await search(kb, FakeReranker({"调休": 0.5, "请假": 0.9}), "A", Q, rewriter=rw)
        assert rw.calls == [] and res.notes == []  # 最高分 0.9 ≥ WEAK：不花这次调用
    asyncio.run(go())


@pytest.mark.parametrize("rw, note", [
    (FakeRewriter(exc=RuntimeError("429")), "rewrite_failed"),
    (FakeRewriter(delay=1.0), "rewrite_failed"),            # 超时（下面把超时改成 0.05 秒）
    (FakeRewriter(out=[Q, "  "]), "rewrite_empty"),          # 只给回原句 / 空白：等于没改写
], ids=["error", "timeout", "empty"])
def test_rewriter_failure_degrades_to_the_original_result(rw, note, monkeypatch):
    monkeypatch.setattr(kb_search, "REWRITE_TIMEOUT", 0.05)

    async def go():
        kb = await corpus()
        base = await search(kb, rerank(), "A", Q)
        res = await search(kb, rerank(), "A", Q, rewriter=rw)
        assert [f.hit.id for f in res.hits] == [f.hit.id for f in base.hits] and res.status == base.status
        assert res.notes == [note] and res.flags == []  # 不进 flags：不影响「能不能说没有」
    asyncio.run(go())


def test_rerank_failure_after_rewrite_keeps_first_pass():
    async def go():
        kb, rr = await corpus(), rerank()
        orig = rr.rerank
        n = {"i": 0}

        async def flaky(q, docs):
            n["i"] += 1
            if n["i"] == 2:
                raise TimeoutError("rerank down")
            return await orig(q, docs)
        rr.rerank = flaky
        res = await search(kb, rr, "A", Q, rewriter=FakeRewriter())
        assert [f.hit.doc_id for f in res.hits] == ["leave"] and res.notes == ["rewrite_failed"]
    asyncio.run(go())


def test_no_reranker_means_no_rewrite():
    async def go():
        rw = FakeRewriter()
        res = await search(await corpus(), None, "A", Q, rewriter=rw)
        assert rw.calls == [] and res.status == "unverified"
    asyncio.run(go())


def test_weighted_rrf_zero_weight_is_pure_vector():
    h = lambda i: Hit(i, i, "doc", "t", 0, 0.0)
    paths = {"vector": [h("x"), h("y")], "bm25": [h("z"), h("y")]}
    assert [f.hit.id for f in rrf(paths, weights={"bm25": 0})] == [f.hit.id for f in rrf({"vector": paths["vector"]})]
    assert "z" not in {f.hit.id for f in rrf(paths, weights={"bm25": 0})}  # 只在 bm25 里的块也不进
    heavy = rrf(paths, weights={"bm25": 3})
    assert heavy[0].hit.id == "y" and heavy[1].hit.id == "z"  # bm25 加权后 z 超过只在向量第 1 的 x


def test_llm_rewriter_parses_and_caches():
    calls = []

    async def complete(*, system, prompt, model=None, max_tokens=None):
        calls.append(prompt)
        return "1. 休息日加班 调休\n- 「加班换调休」\n\n解释不应出现在第三行"
    rw = kb_rewrite.LLMRewriter(n=2, complete=complete)
    assert asyncio.run(rw.rewrite(Q)) == ["休息日加班 调休", "加班换调休"]
    asyncio.run(rw.rewrite(Q))
    assert calls == [Q]  # 第二次走缓存，不再花钱


def test_disk_cache_survives_restart(tmp_path):
    inner = FakeRewriter(out=["a", "b"])
    p = tmp_path / "rw.json"
    assert asyncio.run(kb_rewrite.DiskCachedRewriter(inner, p).rewrite(Q)) == ["a", "b"]
    again = FakeRewriter(out=["不该被调用"])
    assert asyncio.run(kb_rewrite.DiskCachedRewriter(again, p).rewrite(Q)) == ["a", "b"] and again.calls == []


def test_rewriter_off_by_default(monkeypatch):
    monkeypatch.delenv("KB_REWRITE", raising=False)
    assert kb_rewrite.default() is None
    monkeypatch.setenv("KB_REWRITE", "1")
    assert isinstance(kb_rewrite.default(), kb_rewrite.LLMRewriter)


def test_rewrites_are_capped():
    async def go():
        kb, rw = CountingKB(await corpus()), FakeRewriter(out=[REWRITE, "加班调休", "调休怎么算", "休息日上班"])
        res = await search(kb, rerank(), "A", Q, rewriter=rw)
        assert len(res.rewrites) == kb_search.MAX_REWRITES
        assert len(kb.calls) == 2 + 2 * kb_search.MAX_REWRITES
    asyncio.run(go())


def test_parse_keeps_numbers_in_queries_and_drops_preambles():
    """复核⑨：初版 lstrip 掉了「10 天」的 10、「32G」的 32，还把「以下是改写：」当成一条查询。"""
    raw = "好的，以下是改写：\n1. 10 天以上病假工资怎么算\n2) 32G 内存电脑申请\n- 加班调休"
    assert kb_rewrite.parse(raw, 3) == ["10 天以上病假工资怎么算", "32G 内存电脑申请", "加班调休"]


def test_rewrite_finding_nothing_new_returns_first_pass():
    """复核⑭：改写只找回第一轮已有的块：结果和第一轮一样，note 写清楚，不写 rewritten。"""
    async def go():
        kb = await corpus()
        res = await search(kb, rerank(), "A", Q, rewriter=FakeRewriter(out=[LEAVE]))  # 只会找回请假那块和噪声
        base = await search(kb, rerank(), "A", Q)
        assert [f.hit.id for f in res.hits] == [f.hit.id for f in base.hits]
    asyncio.run(go())


def test_weights_apply_to_rewrite_paths_too():
    """复核⑭：bm25 权重 0 时，改写那几路的 bm25 也不参与。"""
    async def go():
        kb = CountingKB(await corpus())
        await search(kb, rerank(), "A", Q, rewriter=FakeRewriter(), weights={"bm25": 0})
        assert ("bm25", REWRITE) in kb.calls  # 检索照跑（召回不变）
        res = await search(kb.kb, rerank(), "A", Q, rewriter=FakeRewriter(), weights={"bm25": 0}, keep_pool=True)
        assert all("bm25" not in n for f in res.pool for n in f.ranks)
    asyncio.run(go())
