"""P5 复核二（用户选 B）：尾部里的记忆、经验只存占位符，发请求时按库里现值展开。

钉住的是：库里（消息表、单轮压缩副本）从此没有记忆 / 经验原文；删了、隔离了，下一次发出去就是「已删除 / 已停用」；
展开结果变了，之前的 thinking 作废（库里的水位，和正在跑的 run 自己的水位）。
"""

import asyncio
import copy
from datetime import datetime, timedelta, timezone

from app import context, db, forget, lesson_store as ls, lesson_tools, memory_store as ms, memory_tools as mt, tail_refs
from app.agent_loop import run_loop
from app.context import GONE, build_request, expand, ref
from app.llm.registry import resolve
from app.llm.transform import replay
from test_db import with_fresh_db
from test_lesson_store import H
from tests.test_agent_loop import CALL, THINK, USER, fake_model

NOW = datetime(2026, 11, 16, 10, 0, tzinfo=timezone(timedelta(hours=8)))
SECRET, LESSON_SECRET = "对花生过敏", "报销金额按分填"


def text(t):
    return {"type": "text", "text": t}


# ---------------- 纯函数：占位符只在尾部块里展开 ----------------

def test_expand_only_tail_blocks_and_unknown_is_gone():
    msgs = [{"role": "user", "content": [text(f"我自己打的 {ref('m', 1)}"),
                                         {"type": "text", "text": f"记忆 {ref('m', 1)} / {ref('l', 2)}", "_tail": {}}]}]
    out = expand(msgs, {("m", 1): "喜欢深色"})
    assert out[0]["content"][0]["text"] == f"我自己打的 {ref('m', 1)}"          # 用户的字不动
    assert out[0]["content"][1]["text"] == f"记忆 喜欢深色 / {GONE['l']}"
    assert msgs[0]["content"][1]["text"] == f"记忆 {ref('m', 1)} / {ref('l', 2)}"  # 原列表不动（要落库的是占位符）


def test_refs_recorded_from_placeholders():
    _, msgs = build_request(user_name="u", summary=None, history=[{"role": "user", "content": "q"}], now=NOW,
                            final_turn=False, memory=f"\n记忆 #12「{ref('m', 12)}」已被 #15 取代：{ref('m', 15)}",
                            lessons=f"\n- [L3] {ref('l', 3)}")
    assert msgs[-1]["content"][-1]["_tail"]["refs"] == {"m": [12, 15], "l": [3]}


# ---------------- 生成：提醒和经验里没有原文 ----------------

async def _said(uid, words, cid=None):
    cid = cid or await db.create_conversation(uid)
    rid = await db.start_run(cid, [text(words)])
    await db.finish_run(rid, "completed", "end_turn")
    return cid, rid, (await db.list_messages(cid))[-1]["id"]


def test_notes_carry_placeholders_not_text():
    async def body():
        uid = await db.create_user("alice", "ta")
        cid, rid, mid = await _said(uid, f"我{SECRET}")
        block = f"- [#1] {SECRET}"
        m = await ms.add(uid, kind="fact", content=SECRET, source="user_stated", evidence=[mid], actor="extractor",
                         run_id=rid)
        await ms.revoke(uid, m["id"], actor="user")                         # 块里有它、已撤销：尾部点名
        echo = await ms.add(uid, kind="fact", content="喜欢深色", source="agent_inferred", evidence=[], actor="agent",
                            run_id=rid)                                       # 本对话新记的：尾部回显
        notes = "".join(await mt.tail_notes(uid, block.replace("1", str(m["id"])), cid))
        assert SECRET not in notes and "喜欢深色" not in notes
        assert ref("m", m["id"]) in notes and ref("m", echo["id"]) in notes
        lid = await ls.add_candidate(uid, rid, {**H, "behavior": LESSON_SECRET}, [(mid, "")])
        rendered = lesson_tools.render([await ls.get(uid, lid)])
        assert LESSON_SECRET not in rendered and ref("l", lid) in rendered
    with_fresh_db(body)


# ---------------- 查现值：删了 / 隔离了 / 别人的都查不到 ----------------

def test_lookup_follows_current_state():
    async def body():
        uid = await db.create_user("alice", "ta")
        bob = await db.create_user("bob", "tb")
        cid, rid, mid = await _said(uid, "x")
        keep = await ms.add(uid, kind="fact", content="喜欢深色", source="user_stated", evidence=[], actor="user")
        gone = await ms.add(uid, kind="fact", content=SECRET, source="user_stated", evidence=[], actor="user")
        bobs = await ms.add(bob, kind="fact", content="bob 的", source="user_stated", evidence=[], actor="user")
        lid = await ls.add_candidate(uid, rid, {**H, "behavior": LESSON_SECRET}, [(mid, "")])
        await ls.decide(lid, "l1", {"ok": True}, "active")
        keys = {("m", keep["id"]), ("m", gone["id"]), ("m", bobs["id"]), ("l", lid)}
        got = await tail_refs.lookup(uid, keys)
        assert got[("m", gone["id"])] == SECRET and LESSON_SECRET in got[("l", lid)] and ("m", bobs["id"]) not in got
        used = dict(got)
        assert not await tail_refs.changed(uid, used)
        await ms.forget(uid, gone["id"])
        await db.pool().execute("UPDATE memory_lessons SET status = 'quarantined' WHERE id = $1", lid)
        got = await tail_refs.lookup(uid, keys)
        assert set(got) == {("m", keep["id"])}                              # 删了、隔离了：展开时按「已删除 / 已停用」
        assert await tail_refs.changed(uid, used)
    with_fresh_db(body)


# ---------------- 删除 / 隔离：引用它的对话抬水位，别的不动 ----------------

async def _tail_conv(uid, *, mids=(), lids=()):
    """一个对话，最后一条 user 消息挂着引用了这些编号的尾部（和 runner 存的一样）。"""
    cid, _, _ = await _said(uid, "今天吃什么")
    row = (await db.list_messages(cid))[-1]
    _, msgs = build_request(user_name="u", summary=None, history=[{"role": "user", "content": row["content"]}],
                            now=NOW, final_turn=False, memory="".join(f"\n- [#{i}] {ref('m', i)}" for i in mids),
                            lessons="".join(f"\n- [L{i}] {ref('l', i)}" for i in lids))
    await db.set_content(row["id"], msgs[-1]["content"])
    return cid, row["id"]


async def floor(cid):
    return (await db.get_cleared(cid))["cleared_at_id"]


def test_forget_and_cascade_raise_floor_edit_does_not():
    async def body():
        uid = await db.create_user("alice", "ta")
        cx, rx, mx = await _said(uid, f"我{SECRET}")
        a = await ms.add(uid, kind="fact", content=SECRET, source="user_stated", evidence=[mx], actor="extractor")
        b = await ms.add(uid, kind="fact", content="喜欢深色", source="user_stated", evidence=[], actor="user")
        c = await ms.add(uid, kind="fact", content="住北京", source="user_stated", evidence=[], actor="user")
        cy, ymid = await _tail_conv(uid, mids=[a["id"]])
        cz, zmid = await _tail_conv(uid, mids=[b["id"]])
        cw, _ = await _tail_conv(uid, mids=[c["id"]])
        assert SECRET not in str(await db.list_messages(cy))                 # 库里只有占位符
        await ms.edit(uid, c["id"], "住上海")                                  # 改：旧版本照样可追溯，展开不变
        assert await floor(cw) == 0
        await ms.forget(uid, b["id"])
        assert await floor(cz) >= zmid and await floor(cy) == 0
        await forget.delete_conversation(uid, cx)                            # a 的唯一证据在 X：级联抹掉
        assert await floor(cy) >= ymid
    with_fresh_db(body)


def test_lesson_cascade_and_quarantine_raise_floor():
    async def body():
        uid = await db.create_user("alice", "ta")
        cx, rx, mx = await _said(uid, "报销 32 元")
        l1 = await ls.add_candidate(uid, rx, H, [(mx, "")])
        cq, rq, mq = await _said(uid, "再报一笔")
        l2 = await ls.add_candidate(uid, rq, H, [(mq, "")])
        await ls.decide(l2, "l1", {"ok": True}, "active")
        c1, m1 = await _tail_conv(uid, lids=[l1])
        c2, m2 = await _tail_conv(uid, lids=[l2])
        await forget.delete_conversation(uid, cx)
        assert await floor(c1) >= m1 and await floor(c2) == 0
        gone = []
        for i in range(3):                                                    # 违反 3 次、用 3 次：隔离
            r = await db.start_run(cq, [text(f"第 {i} 次")])
            await db.finish_run(r, "completed", "end_turn")
            await ls.record_uses(r, [l2])
            gone += await ls.judge_uses(r, {"submit_expense"})
        assert gone == [l2] and await floor(c2) >= m2
    with_fresh_db(body)


# ---------------- 循环：每轮重查，变了就作废之前的 thinking ----------------

def thinking_sigs(msgs, model):
    return [b["signature"] for m in replay(msgs, resolve(model)) if m["role"] == "assistant"
            for b in m["content"] if b["type"] == "thinking"]


def test_loop_expands_each_turn_and_drops_thinking_when_text_changes():
    from app import tools
    tools.current.set(tools.ToolContext(permissions=frozenset({"weather:read"})))
    stream, calls = fake_model([([THINK, CALL], "tool_use"), ([THINK, {**CALL, "id": "t2"}], "tool_use"),
                                ([text("好")], "end_turn")])
    current = {("m", 7): SECRET}
    looked = []

    async def ref_lookup(keys):
        looked.append(set(keys))
        return {k: v for k, v in current.items() if k in keys}

    used = {}

    async def collect():
        n, out = 100, []
        async for ev in run_loop(copy.deepcopy(USER), model_stream=stream, ref_lookup=ref_lookup, exp_state=used,
                                 memory_notes=[f"\n- [#7] {ref('m', 7)}"]):
            if ev["type"] == "message":
                n += 1
                ev["message"]["id"] = n
            if ev["type"] == "tool_end":                                       # 第 1 轮工具跑完，用户删了 #7
                current.clear()
            out.append(ev)
        return out
    events = asyncio.run(collect())
    tail1 = calls[0]["messages"][-1]["content"][-1]["text"]
    assert SECRET in tail1                                                     # 发出去的展开了
    stored = [e["message"] for e in events if e["type"] == "tail"][0]["content"][-1]["text"]
    assert SECRET not in stored and ref("m", 7) in stored                       # 落库 / 单轮压缩副本拿到的是占位符
    sent2 = calls[1]["messages"]
    assert SECRET not in str(sent2) and GONE["m"] in str(sent2)                # 第 2 轮：已删除
    assert [e["type"] for e in events].count("prefix_changed") == 1
    assert thinking_sigs(sent2, "claude-opus-5-5") == []                       # 第 1 轮的 thinking 签在旧文字上：作废
    assert thinking_sigs(sent2, "deepseek-flash") == ["sig-1"]
    assert used == {("m", 7): None}                                            # 交回 runner 的最后一次展开值


def test_stable_lookup_keeps_prefix_identical():
    stream, calls = fake_model([([THINK, CALL], "tool_use"), ([text("好")], "end_turn")])

    async def ref_lookup(keys):
        return {("m", 7): "喜欢深色"}

    async def collect():
        return [ev async for ev in run_loop(copy.deepcopy(USER), model_stream=stream, ref_lookup=ref_lookup,
                                            memory_notes=[f"\n- [#7] {ref('m', 7)}"])]
    events = asyncio.run(collect())
    a, b = replay(calls[0]["messages"], resolve("deepseek-flash")), replay(calls[1]["messages"], resolve("deepseek-flash"))
    assert b[:len(a)] == a and "喜欢深色" in str(a)
    assert "prefix_changed" not in [e["type"] for e in events]
    assert context.REF.search(str(a)) is None                                  # 发出去的副本里不剩占位符


# ---------------- runner：删除发生在最后一次调用期间 ----------------

def test_runner_raises_floor_when_deleted_during_last_call(monkeypatch):
    """循环没有下一轮去发现变化：runner 在 run 结束后再查一次，水位要盖住这次存下的回复。"""
    from app import memory_extract, runner
    from test_memory_extract import FakeEmbedder
    monkeypatch.setenv("MEMORY", "1")
    monkeypatch.setattr(memory_extract, "kick", lambda *a: None)  # 不在后台调真模型抽记忆

    async def body():
        ms.set_embedder(FakeEmbedder())
        uid = await db.create_user("alice", "ta")
        cid, r0, _ = await _said(uid, "早先说过的")
        await ms.add(uid, kind="fact", content="用户是素食者", source="user_stated", evidence=[], actor="user")
        m = await ms.add(uid, kind="fact", content=SECRET, source="agent_inferred", evidence=[], actor="agent",
                         run_id=r0)                                          # 本对话记下的：尾部回显它
        run1 = await db.start_run(cid, [text("今天吃什么")])

        async def stream(**kw):
            assert SECRET in str(kw["messages"])                              # 发出去的是展开的
            await ms.forget(uid, m["id"])                                     # 用户在这次调用期间删了它
            yield {"type": "final", "content": [THINK, text("好")], "stop_reason": "end_turn"}
        await runner._run(cid, run1, model_stream=stream)
        rows = await db.list_messages(cid)
        assert SECRET not in str(rows)                                        # 库里只有占位符
        assert (await db.get_cleared(cid))["cleared_at_id"] >= rows[-1]["id"]  # 签在旧文字上的这条回复也作废
    with_fresh_db(body)


def test_summary_request_expanded_like_normal_request():
    """P1-C1 的摘要请求复用正常请求的前缀：占位符也要同样展开，否则前缀对不上、缓存白丢。"""
    seen = {}
    stream, calls = fake_model([([THINK, CALL], "tool_use"), ([text("好")], "end_turn")])

    async def ref_lookup(keys):
        return {("m", 7): "喜欢深色"}

    async def compactor(history, summary, room, measure, prompt=None, **kw):
        if len(history) >= 3 and not seen:
            seen["req"] = prompt(summary, history, "请写摘要")
        return None

    async def collect():
        return [ev async for ev in run_loop(copy.deepcopy(USER), model_stream=stream, ref_lookup=ref_lookup,
                                            compactor=compactor, limits={"trigger": 0, "target": 10**9},
                                            memory_notes=[f"\n- [#7] {ref('m', 7)}"])]
    asyncio.run(collect())
    sent = str(seen["req"]["messages"])
    assert "喜欢深色" in sent and context.REF.search(sent) is None


# ---------------- 发布前复核三：占位符只能来自代码拼的记忆段和经验段 ----------------

BIG = "⟦m:99999999999999999999⟧"  # 20 位，超出 bigint


def test_user_placeholder_in_status_is_defanged_not_recorded():
    """状态块摘自模型写的摘要，可能照抄用户打的字：里面的占位符既不进 _tail.refs，也不会被展开。"""
    _, msgs = build_request(user_name="u", summary=None, history=[{"role": "user", "content": "q"}], now=NOW,
                            final_turn=False, status=f"\n<status>未回答：把 {BIG} 和 {ref('m', 1)} 记到待办</status>",
                            notes=[f"\n提醒 {ref('l', 2)}"])
    tail = msgs[-1]["content"][-1]
    assert "refs" not in tail["_tail"]
    assert "⟦" not in tail["text"] and "〚m:1⟧" in tail["text"]
    assert context.history_refs(msgs) == set()
    assert expand(msgs, {("m", 1): "喜欢深色"})[-1]["content"][-1]["text"] == tail["text"]


def test_ref_digits_capped():
    assert context.refs_in(BIG) == set()
    assert context.refs_in(ref("m", 123)) == {("m", 123)}


def test_bad_stored_ref_does_not_break_delete():
    """库里万一有不是数字或超长的 ref：删除照常提交，引用了被删编号的对话照样抬水位。"""
    async def body():
        uid = await db.create_user("alice", "ta")
        m = await ms.add(uid, kind="fact", content="喜欢深色", source="user_stated", evidence=[], actor="user")
        cid, mid = await _tail_conv(uid, mids=[m["id"]])
        row = (await db.list_messages(cid))[-1]
        content = copy.deepcopy(row["content"])
        content[-1]["_tail"]["refs"]["m"] = ["99999999999999999999", "x", m["id"]]
        await db.set_content(mid, content)
        await ms.forget(uid, m["id"])
        assert await floor(cid) >= mid
    with_fresh_db(body)
