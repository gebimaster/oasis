"""P4 第 1 步：经验表的存储规则。只对本人生效、证据必须是本人的、门禁不翻案、违反够多才隔离、删对话级联。"""

import pytest

from app import db, forget, lesson_store as ls
from test_db import with_fresh_db

H = {"precondition": "用户要提交报销", "behavior": "submit_expense 的 amount 按分填（元 × 100）",
     "expected": "报销单上的金额和用户说的一致", "verify": [{"name": "tool_called", "args": {"name": "submit_expense"}}],
     "tools": ["submit_expense"], "rationale": "传了 32，单子上是 0.32 元，用户纠正后改传 3200", "outcome": "corrected"}


def text(t):
    return {"type": "text", "text": t}


async def _run(name):
    uid = await db.create_user(name, f"tok-{name}")
    cid = await db.create_conversation(uid)
    rid = await db.start_run(cid, [text("帮我报销 32 元打车费")])
    mid = (await db.list_messages(cid))[-1]["id"]
    return uid, cid, rid, mid


def test_candidate_isolation_and_evidence():
    async def body():
        alice, _, rid, mid = await _run("alice")
        bob, _, _, bob_mid = await _run("bob")
        lid = await ls.add_candidate(alice, rid, H, [(mid, ""), (mid, "")])  # 重复证据只记一行
        got = await ls.get(alice, lid)
        assert got["status"] == "candidate" and got["tools"] == ["submit_expense"] and got["verify"] == H["verify"]
        assert await ls.get(bob, lid) is None                                  # 别人查不到
        assert [e["message_id"] for e in await ls.evidence(alice, lid)] == [mid]
        assert await ls.evidence(bob, lid) == []
        with pytest.raises(PermissionError):                                   # 拿别人的消息当证据：整笔回滚
            await ls.add_candidate(alice, rid, H, [(bob_mid, "")])
        assert len(await ls.by_status(alice, "candidate")) == 1
        with pytest.raises(ValueError):
            await ls.add_candidate(alice, rid, {**H, "outcome": "great"}, [])
    with_fresh_db(body)


def test_decide_does_not_overturn():
    async def body():
        uid, _, rid, mid = await _run("alice")
        lid = await ls.add_candidate(uid, rid, H, [(mid, "")])
        await ls.decide(lid, "l0", {"ok": True})                               # 只记结果，状态不变
        assert (await ls.get(uid, lid))["status"] == "candidate"
        await ls.decide(lid, "l1", {"ok": False, "why": "回归 1 道"}, "rejected")
        await ls.decide(lid, "l1", {"ok": True}, "active")                     # 已判过的不翻案
        got = await ls.get(uid, lid)
        assert got["status"] == "rejected" and got["gate"]["l0"] == {"ok": True}
        assert got["gate"]["l1"]["why"] == "回归 1 道" and got["decided_at"] is not None
    with_fresh_db(body)


def test_violations_quarantine_needs_three_and_majority():
    async def body():
        uid, cid, rid, mid = await _run("alice")
        lid = await ls.add_candidate(uid, rid, H, [(mid, "")])
        await ls.decide(lid, "l1", {"ok": True}, "active")
        await db.finish_run(rid, "completed", "end_turn")
        for i in range(7):
            r = await db.start_run(cid, [text(f"第 {i} 次")])
            await db.finish_run(r, "completed", "end_turn")
            await ls.record_uses(r, [lid])
            await ls.record_uses(r, [lid])                                     # 同一个 run 只算一次
            # 前 3 次好好的（get_weather 失败不算它的违反）；之后 submit_expense 失败
            failed = {"get_weather"} if i < 3 else {"submit_expense"}
            gone = await ls.judge_uses(r, failed)
            assert await ls.judge_uses(r, {"submit_expense"}) == []            # 判过的不再判
            got = await ls.get(uid, lid)
            assert got["uses"] == i + 1
            # i=5：违反 3 次、用了 6 次，没过半 → 不隔离；i=6：违反 4 次、用了 7 次 → 隔离
            assert (gone == [lid]) == (i == 6) and got["status"] == ("quarantined" if i == 6 else "active")
    with_fresh_db(body)


def test_delete_conversation_cascades_lessons():
    async def body():
        uid, cid, rid, mid = await _run("alice")
        lid = await ls.add_candidate(uid, rid, H, [(mid, "")])
        bob, _, rid2, mid2 = await _run("bob")  # 别人的经验不受影响
        other = await ls.add_candidate(bob, rid2, H, [(mid2, "")])
        await db.finish_run(rid, "completed", "end_turn")
        await ls.record_uses(rid, [lid])
        out = await forget.delete_conversation(uid, cid)
        assert out["lessons"] == 1 and await ls.get(uid, lid) is None
        assert await db.pool().fetchval("SELECT count(*) FROM lesson_uses") == 0
        assert [r["id"] for r in await db.pool().fetch("SELECT id FROM memory_lessons")] == [other]
    with_fresh_db(body)
