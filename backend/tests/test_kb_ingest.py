"""M6 第 1 步：切块 + 幂等入库 + 按用户隔离的块 id；O5 第 2 步：按条目、按小节切，边界只由内容决定。"""

import asyncio
import random

from app.kb_chunk import MIN_ENTRY, OVERLAP, SIZE, split_chunks, split_text
from kb_fakes import make_kb


DOC = "报销制度\n\n打车需提供发票，单次上限 500 元。\n\n年假按工龄计算，满一年 5 天。"


def test_short_paragraphs_merge_and_long_paragraph_overlaps():
    # O5 起含义变了：不是「攒到 200 字」，而是三段都短于 MIN_ENTRY，一路并进下一条，到末尾成一块
    assert all(len(p) < MIN_ENTRY for p in DOC.split("\n\n"))
    assert split_text(DOC) == [DOC.replace("\n\n", "\n")]
    long = "".join(f"第{i}句规定了某项很具体的报销细节内容。" for i in range(30))
    chunks = split_text(long)
    assert len(chunks) > 1
    assert all(len(c) <= SIZE + OVERLAP for c in chunks)
    for a, b in zip(chunks, chunks[1:]):  # 后一块以前一块末尾 OVERLAP 字开头
        assert b.startswith(a[-OVERLAP:])
    for i in range(30):  # 没有句子丢失
        assert f"第{i}句" in "".join(chunks)


def test_editing_one_paragraph_keeps_other_chunks():
    """按段落切：只改中间一段，前后的块 hash 不变。"""
    paras = [f"第{i}段。" + "内容" * 90 for i in range(4)]  # 每段约 185 字，各成一块
    before = split_text("\n\n".join(paras))
    paras[1] = paras[1].replace("内容", "修改", 1)
    after = split_text("\n\n".join(paras))
    assert len(before) == len(after) == 4
    assert [b == a for b, a in zip(before, after)] == [True, False, True, True]


def test_reingest_is_idempotent_and_incremental():
    async def go():
        kb, emb = make_kb()
        text = "\n\n".join(f"第{i}段。" + "内容" * 90 for i in range(4))
        assert await kb.ingest("A", "manual", text) == {"added": 4, "deleted": 0, "kept": 0}
        emb.calls.clear()
        assert await kb.ingest("A", "manual", text) == {"added": 0, "deleted": 0, "kept": 4}
        assert emb.calls == []  # 内容没变：一次 embedding 都不调
        changed = text.replace("第1段。内容", "第1段。修改")
        assert await kb.ingest("A", "manual", changed) == {"added": 1, "deleted": 1, "kept": 3}
        assert len(emb.calls) == 1 and emb.calls[0].startswith("manual\n")  # 向量带标题算
        stored = kb.col.get(where={"user_id": "A"})["documents"]
        assert len(stored) == 4 and not any("第1段。内容" in d for d in stored)  # 旧块已删
    asyncio.run(go())


def test_same_content_for_two_users_is_isolated():
    """反例钉住：id 若只用内容 hash，B 会被跳过；A 删文档后 B 也跟着没了。"""
    async def go():
        kb, _ = make_kb()
        assert (await kb.ingest("A", "policy", DOC))["added"] == 1
        assert (await kb.ingest("B", "policy", DOC))["added"] == 1  # B 不会因为 A 有同内容而跳过
        await kb.delete_document("A", "policy")
        assert kb.col.get(where={"user_id": "A"})["ids"] == []
        assert len(kb.col.get(where={"user_id": "B"})["ids"]) == 1  # A 删除不影响 B
    asyncio.run(go())


def test_title_change_reembeds_every_chunk():
    """标题进了向量和词表（contextual embedding）：改标题后旧向量全部过期，不能当 kept 留着。"""
    async def go():
        kb, emb = make_kb()
        await kb.ingest("A", "d", DOC, title="旧标题")
        emb.calls.clear()
        stats = await kb.ingest("A", "d", DOC, title="新标题")
        assert stats["kept"] == 0 and stats["added"] == stats["deleted"] > 0
        assert all(t.startswith("新标题\n") for t in emb.calls)
        hits = await kb.keyword_search("A", "新标题", 5)
        assert hits and all(h.title == "新标题" for h in hits)
    asyncio.run(go())


# ---- O5 第 2 步：按条目、按标题切，边界只由内容决定 ----

import re
from pathlib import Path

SAMPLES = Path(__file__).parent.parent / "kb_samples"
_ENTRY = re.compile(r"^E\d{4}：", re.M)


def _body(path: Path) -> str:
    """和 scripts/kb_seed.py 的 parse 一样先剥掉「# 标题」。不剥的话标题行也占长度，断言的块数就是错的。"""
    return path.read_text(encoding="utf-8").partition("\n")[2].strip()


def test_error_codes_one_entry_per_chunk():
    """修前红：旧切法按累计长度合并，剥掉标题后第 0 块是 E1024 + E1025 + E2001（70+1+66+1+59=197 ≤ 200）。"""
    chunks = split_text(_body(SAMPLES / "demo" / "error-codes.md"))
    assert [len(_ENTRY.findall(c)) for c in chunks if _ENTRY.search(c)] == [1, 1, 1, 1]


SECTIONED = (
    "开头没有小节标题的一段说明，这段内容超过二十个字。\n\n"
    "## 申请\n\n"
    "步骤如下：\n\n"
    "- 第一项：在系统里提交申请单，写明申请理由。\n"
    "- 第二项：直属主管在两个工作日内完成审批。\n\n"
    "## 注意\n"
    "这一段紧跟在标题下面没有空行，也要归到注意小节。\n\n"
    "以上。"
)


def test_sections_list_items_and_short_entries():
    """块不跨标题、metadata 有小节标题；列表按项切；短引导句并进下一条；小节末尾的短条目自成一块。"""
    chunks = split_chunks(SECTIONED)
    assert [(c.section, c.para_index, c.text[:6]) for c in chunks] == [
        ("", 0, "开头没有小节"),
        ("申请", 1, "步骤如下：\n"),  # 引导句（5 字）并进第一项，para_index 记引导句那段
        ("申请", 2, "- 第二项："),
        ("注意", 3, "这一段紧跟在"),
        ("注意", 4, "以上。"),
    ]
    assert chunks[1].text == "步骤如下：\n- 第一项：在系统里提交申请单，写明申请理由。"
    assert not any("#" in c.text for c in chunks)  # 标题行不进块正文


def test_backup_doc_stays_in_sections_and_overlap_happens():
    """新文档：长段落被切成带重叠的两块，列表项各自成块，没有块跨标题。"""
    chunks = split_chunks(_body(SAMPLES / "eval_extra" / "backup.md"))
    assert {c.section for c in chunks} == {"备份策略", "恢复申请", "保留与销毁"}
    assert not any("#" in c.text for c in chunks)
    assert all(len(c.text) <= SIZE + OVERLAP for c in chunks)
    pairs = [(a, b) for a, b in zip(chunks, chunks[1:]) if a.para_index == b.para_index and a.section == b.section]
    assert any(b.text.startswith(a.text[-OVERLAP:]) for a, b in pairs)  # 重叠真正生效了一次
    assert sum(c.text.startswith(("2.", "3.", "4.")) for c in chunks) == 3


_POOL = "规定员工申请审批流程系统提交材料部门负责人确认完成之后可以按照标准执行相关事项记录保存期限通知"


def _para(rng: random.Random, lo: int, hi: int) -> str:
    """一段由汉字组成、以句号结尾的段落，长度在 [lo, hi]。不含 #、列表标记，只测条目和合并规则。"""
    n = rng.randint(lo, hi)
    return "".join(rng.choice(_POOL) for _ in range(n - 1)) + "。"


def test_insert_paragraph_changes_at_most_one_chunk():
    """性质测试（固定种子）：原有段落都 ≥ MIN_ENTRY 时，在位置 i 插一段（可能短于 MIN_ENTRY），
    i 之前的块全不变，新增、消失的块各不超过 1 个，内容不丢。按累计长度合并时第一条就会挂。
    复核指出这只是常见情况：插入点前面是短条目时见下一条测试。"""
    rng = random.Random(20261105)
    for _ in range(300):
        paras = [_para(rng, MIN_ENTRY, 90) for _ in range(rng.randint(2, 12))]
        i = rng.randint(0, len(paras))
        inserted = _para(rng, 3, 90)
        before = split_text("\n\n".join(paras))
        after = split_text("\n\n".join(paras[:i] + [inserted] + paras[i:]))
        assert after[:i] == before[:i], (len(paras), i, len(inserted))
        assert len(set(after) - set(before)) <= 1 and len(set(before) - set(after)) <= 1
        assert all(p in "".join(after) for p in [*paras, inserted])


def _window(after_paras: list[str], before_paras: list[str], i: int) -> tuple[int, int]:
    """插入第 i 段后受影响的段落区间（按插入后的序号）：往前连着的短条目，到插入前后各自所在合并组的末尾。"""
    s = i
    while s > 0 and len(after_paras[s - 1]) < MIN_ENTRY:
        s -= 1
    e = i
    while e < len(after_paras) - 1 and len(after_paras[e]) < MIN_ENTRY:
        e += 1
    eb = s  # 插入前，从 s 开始的那个合并组的末尾
    while eb < len(before_paras) - 1 and len(before_paras[eb]) < MIN_ENTRY:
        eb += 1
    return s, max(e, eb + 1 if eb >= i else eb)


def test_insert_changes_only_its_merge_group():
    """复核修复（用户选 A：缩小保证）：原有段落里也有短条目和超过 SIZE 的长段落时，插入一段，
    改动只限于它所在的合并组（前面连着的短条目 + 新段落 + 它们并进的那一条 + 那串短条目原来并进的那一条），
    组外的块一个不变。
    插入点前面不是短条目、合并后不超过 SIZE 时，就是上一条测试的严格情况。"""
    rng = random.Random(20261106)

    def para() -> str:
        r = rng.random()
        if r < 0.3:
            return _para(rng, 3, MIN_ENTRY - 1)  # 引导句、残片
        if r < 0.45:
            return "".join(_para(rng, 10, 20) for _ in range(rng.randint(10, 16)))  # 超过 SIZE 的长段落
        return _para(rng, MIN_ENTRY, 190)

    seen_two = 0
    for _ in range(500):
        paras = [para() for _ in range(rng.randint(2, 12))]
        i = rng.randint(0, len(paras))
        after_paras = paras[:i] + [para()] + paras[i:]
        before = split_chunks("\n\n".join(paras))
        after = split_chunks("\n\n".join(after_paras))
        s, e = _window(after_paras, paras, i)
        shift = lambda c: (c.text, c.para_index + (c.para_index >= i))  # 插入前的段号换算到插入后
        outside_before = [shift(c) for c in before if not s <= shift(c)[1] <= e]
        outside_after = [(c.text, c.para_index) for c in after if not s <= c.para_index <= e]
        assert outside_before == outside_after, (i, s, e)
        joined = "".join(c.text for c in after)  # 长段落带重叠，整段不再是连续子串，按句检查不丢
        assert all(sent in joined for p in after_paras for sent in p.split("。") if sent)
        seen_two += len({c.text for c in after} - {c.text for c in before}) >= 2
    assert seen_two > 0  # 确实生成到了「一次插入重算 2 块」的情况，不是只测了常见情况


def test_insert_after_lead_in_reembeds_two_chunks():
    """复核复现：backup.md 在引导句「恢复步骤如下：」后面插一段，新段落把引导句抢走，
    「引导句 + 第 1 项」那块被删，新增「引导句 + 新段落」和「第 1 项」两块，算 2 次 embedding。"""
    async def go():
        kb, emb = make_kb()
        body = _body(SAMPLES / "eval_extra" / "backup.md")
        await kb.ingest("A", "backup", body, title="备份")
        emb.calls.clear()
        changed = body.replace("恢复步骤如下：\n\n", "恢复步骤如下：\n\n注意：恢复前先确认备份点，避免覆盖现有数据。\n\n")
        assert await kb.ingest("A", "backup", changed, title="备份") == {"added": 2, "deleted": 1, "kept": 7}
        assert len(emb.calls) == 2
    asyncio.run(go())


def test_short_lead_in_before_long_entry_resplits_whole_entry():
    """第二轮复核（用户选「改说法 + 补测试」）：短引导句并进一条超过 SIZE 的长条目，整条要重新按句子切，
    这个例子里每句 20 字、第一块装不下引导句，贪心断句的边界一路挪位，长条目的块全部重算；但合并组外的块不动。
    钉住代价，不是钉住「只变 1-2 块」。边界不一定全挪：每句 19 字时第一块还装得下，只有第一块变（第三轮复核）。"""
    async def go():
        kb, emb = make_kb()
        head = "开头一段普通说明文字，长度超过二十个字没有问题。"
        long_entry = "".join(f"第{i:02d}句规定员工提交申请材料的具体要求。" for i in range(50))  # 每句 20 字，共 1000 字
        tail = "最后一段也是普通说明文字，长度超过二十个字没有问题。"
        await kb.ingest("A", "d", f"{head}\n\n{long_entry}\n\n{tail}", title="t")
        before = split_text(f"{head}\n\n{long_entry}\n\n{tail}")
        emb.calls.clear()
        lead = "步骤如下："
        assert len(lead) < MIN_ENTRY < len(head) and len(long_entry) > SIZE
        stats = await kb.ingest("A", "d", f"{head}\n\n{lead}\n\n{long_entry}\n\n{tail}", title="t")
        after = split_text(f"{head}\n\n{lead}\n\n{long_entry}\n\n{tail}")
        assert after[0] == before[0] == head and after[-1] == before[-1] == tail  # 组外不变
        assert after[1].startswith(lead + "\n第00句")
        group_before, group_after = set(before[1:-1]), set(after[1:-1])
        assert not group_before & group_after  # 这个例子里长条目的块一块都没留下：边界全挪了
        assert stats == {"added": len(group_after), "deleted": len(group_before), "kept": 2}
        assert len(emb.calls) == len(group_after) >= 5
    asyncio.run(go())


def test_list_paragraph_edit_can_flip_whole_paragraph():
    """第二轮复核（用户选「改说法、列进不足」）：列表判定看「超过一半的行是列表项」。给第 2 项加一行续行，
    比例跌到一半，整段变成一条，第 1 项的块也被删。这是已知限制，这条测试记录它；改规则时它会变红提醒改文档。"""
    async def go():
        kb, emb = make_kb()
        item1 = "1. 第一项：在系统里提交申请单，写明申请理由和期限。"
        item2 = "2. 第二项：直属主管在两个工作日内完成审批并通知。"
        before = f"{item1}\n{item2}\n补充说明一行。"
        after = before + "\n再补充一行。"
        assert split_text(before) == [item1, item2 + "\n补充说明一行。"]
        assert split_text(after) == [after]
        await kb.ingest("A", "d", before, title="t")
        emb.calls.clear()
        assert await kb.ingest("A", "d", after, title="t") == {"added": 1, "deleted": 2, "kept": 0}
    asyncio.run(go())


def test_insert_paragraph_reembeds_one_chunk():
    """同一性质落到入库上：用 FakeEmbedder.calls 断言重灌只算 1 次 embedding。"""
    async def go():
        rng = random.Random(42)
        for _ in range(20):
            kb, emb = make_kb()
            paras = [_para(rng, MIN_ENTRY, 90) for _ in range(rng.randint(2, 10))]
            await kb.ingest("A", "d", "\n\n".join(paras), title="t")
            emb.calls.clear()
            i = rng.randint(0, len(paras))
            text = "\n\n".join(paras[:i] + [_para(rng, 3, 90)] + paras[i:])
            stats = await kb.ingest("A", "d", text, title="t")
            assert len(emb.calls) == 1 and stats["added"] == 1, stats
            # 没重算的块（kept）段落序号也要跟着挪：引用「第N段」读的就是它
            stored = kb.col.get(where={"user_id": "A"}, include=["documents", "metadatas"])
            assert {d: m["para_index"] for d, m in zip(stored["documents"], stored["metadatas"])} == \
                {c.text: c.para_index for c in split_chunks(text)}
    asyncio.run(go())


def test_edit_inside_long_paragraph_stays_in_that_paragraph():
    """长段落内部按句子拼块、带重叠，carry 只在本段里传：改它中间一句，别的段落的块一个都不变。"""
    rng = random.Random(7)
    long_a = "".join(_para(rng, 15, 30) for _ in range(20))
    long_b = "".join(_para(rng, 15, 30) for _ in range(20))
    paras = [_para(rng, 30, 80), long_a, _para(rng, 30, 80), long_b, _para(rng, 30, 80)]
    before = split_chunks("\n\n".join(paras))
    sents = long_a.split("。")
    sents[10] = "这一句被改过了"
    paras[1] = "。".join(sents)
    after = split_chunks("\n\n".join(paras))
    in_a = [c.text for c in before if c.para_index == 1]
    assert len(in_a) > 2 and all(b.startswith(a[-OVERLAP:]) for a, b in zip(in_a, in_a[1:]))
    # 长段落的第一块从本段开头开始，不带上一段的尾巴：重叠不跨条目
    assert in_a[0].startswith(long_a[:10]) and [c.text for c in before if c.para_index == 3][0].startswith(long_b[:10])
    others = lambda cs: [c.text for c in cs if c.para_index != 1]
    assert others(before) == others(after)
    assert [c.text for c in before if c.para_index == 1] != [c.text for c in after if c.para_index == 1]


def test_section_title_in_embedding_and_rename_reembeds_that_section_only():
    """用户选 Q7=B：小节标题拼进 embedding 文本（也就进了块 id）。改一个小节标题，只重算这一节的块。"""
    async def go():
        kb, emb = make_kb()
        await kb.ingest("A", "d", SECTIONED, title="手册")
        assert "手册 > 申请\n- 第二项：直属主管在两个工作日内完成审批。" in emb.calls
        assert "手册\n开头没有小节标题的一段说明，这段内容超过二十个字。" in emb.calls  # 没小节时和 M6 一样
        metas = kb.col.get(where={"user_id": "A"})["metadatas"]
        assert {m["section_title"] for m in metas} == {"", "申请", "注意"}
        emb.calls.clear()
        stats = await kb.ingest("A", "d", SECTIONED.replace("## 注意", "## 注意事项"), title="手册")
        assert stats == {"added": 2, "deleted": 2, "kept": 3}
        assert all(c.startswith("手册 > 注意事项\n") for c in emb.calls)
    asyncio.run(go())


def test_rerank_sees_section_title():
    """用户选 Q7=B：rerank 看到的文本和 embedding 一样带小节标题，不只是「文档标题\n块」。"""
    from app.kb_search import search
    from kb_fakes import FakeReranker

    async def go():
        kb, _ = make_kb()
        await kb.ingest("A", "d", SECTIONED, title="手册")
        rr = FakeReranker({"第二项": 0.9})
        res = await search(kb, rr, "A", "第二项 审批", top_k=3)
        assert res.status == "ok" and res.hits[0].hit.section == "申请"
        assert "手册 > 申请\n- 第二项：直属主管在两个工作日内完成审批。" in rr.calls[0][1]
    asyncio.run(go())


def test_chunks_ingested_before_o5_still_readable():
    """O5 之前入库的块没有 section_title / para_index：检索不报错，段落序号退回 chunk_index，重灌一次就补上。"""
    async def go():
        kb, _ = make_kb()
        kb.col.add(ids=["old"], embeddings=[[0.1] * 8], documents=["旧块：E1024 设备离线。"],
                   metadatas=[{"user_id": "A", "doc_id": "d", "title": "t", "chunk_index": 3, "tokens": "e1024 设备 离线"}])
        [h] = await kb.keyword_search("A", "E1024", 5)
        assert (h.section, h.para_index, h.chunk_index) == ("", 3, 3)
    asyncio.run(go())
