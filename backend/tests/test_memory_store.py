"""P2 第 2 步：记忆表的存储规则。取代只作废不删、跨用户隔离、证据必须是本人的、事件只追加。"""

import asyncio

import pytest

from app import db, memory_store as ms
from test_db import with_fresh_db


def text(t):
    return {"type": "text", "text": t}


async def _user_msg(name, said):
    uid = await db.create_user(name, f"tok-{name}")
    cid = await db.create_conversation(uid)
    rid = await db.start_run(cid, [text(said)])
    mid = (await db.list_messages(cid))[-1]["id"]
    return uid, rid, mid


def test_add_supersede_keeps_old_row():
    async def body():
        uid, rid, m1 = await _user_msg("alice", "我前端用 Vue")
        a = await ms.add(uid, kind="fact", content="前端用 Vue", source="user_stated",
                         evidence=[m1], actor="extractor", run_id=rid)
        b = await ms.add(uid, kind="fact", content="前端改用 React", source="user_stated",
                         evidence=[m1], actor="extractor", run_id=rid, supersedes=[a["id"]])
        assert b["superseded"] == [a["id"]]
        assert [r["content"] for r in await ms.active(uid)] == ["前端改用 React"]     # 只返回 active
        old = await ms.open_item(uid, a["id"])                                       # 旧行还在
        assert old["status"] == "superseded" and old["superseded_by"] == b["id"]
        assert old["invalid_at"] is not None and old["superseded_at"] is not None
        new = await ms.open_item(uid, b["id"])
        assert [r["id"] for r in new["replaced"]] == [a["id"]]                       # 取代链双向可查
        assert new["evidence"][0]["content"] == [text("我前端用 Vue")]               # 证据是原文
        assert [e["op"] for e in await ms.events(uid, a["id"])] == ["add", "supersede"]
        # 再取代一次已作废的：跳过，不报错，也不改状态
        c = await ms.add(uid, kind="fact", content="x", source="user_stated", evidence=[],
                         actor="extractor", supersedes=[a["id"]])
        assert c["superseded"] == []
    with_fresh_db(body)


def test_isolation_between_users():
    async def body():
        alice, _, ma = await _user_msg("alice", "我喜欢深色主题")
        bob, _, mb = await _user_msg("bob", "我喜欢浅色主题")
        a = await ms.add(alice, kind="preference", content="喜欢深色主题", source="user_stated",
                         evidence=[ma], actor="extractor")
        # bob 不能取代、撤销、打开 alice 的条目
        await ms.add(bob, kind="preference", content="喜欢浅色", source="user_stated", evidence=[mb],
                     actor="agent", supersedes=[a["id"]])
        assert not await ms.revoke(bob, a["id"], actor="agent")
        assert await ms.open_item(bob, a["id"]) is None
        assert [r["content"] for r in await ms.active(alice)] == ["喜欢深色主题"]
        # 证据不能指向别人的消息：整笔回滚，不留半条记忆
        with pytest.raises(ValueError):
            await ms.add(bob, kind="fact", content="偷看", source="user_stated", evidence=[ma], actor="agent")
        assert [r["content"] for r in await ms.active(bob)] == ["喜欢浅色"]
    with_fresh_db(body)


def test_revoke_and_source_filter():
    async def body():
        uid, rid, m = await _user_msg("alice", "我每周三健身")
        a = await ms.add(uid, kind="commitment", content="每周三健身", source="user_stated",
                         evidence=[m], actor="extractor")
        await ms.add(uid, kind="note", content="可能喜欢运动", source="agent_inferred", evidence=[], actor="agent")
        assert [r["content"] for r in await ms.active(uid, source="user_stated")] == ["每周三健身"]
        assert await ms.revoke(uid, a["id"], actor="user", reason="不去了")
        assert not await ms.revoke(uid, a["id"], actor="user")                       # 第二次不重复记事件
        assert await ms.active(uid, source="user_stated") == []
        assert [e["op"] for e in await ms.events(uid, a["id"])] == ["add", "revoke"]
        with pytest.raises(ValueError):
            await ms.add(uid, kind="secret", content="x", source="user_stated", evidence=[], actor="agent")
    with_fresh_db(body)


def test_concurrent_supersede_only_once():
    async def body():
        uid, _, m = await _user_msg("alice", "我住北京")
        a = await ms.add(uid, kind="fact", content="住北京", source="user_stated", evidence=[m], actor="extractor")
        r = await asyncio.gather(*[ms.add(uid, kind="fact", content=f"搬到{c}", source="user_stated",
                                          evidence=[m], actor="extractor", supersedes=[a["id"]])
                                   for c in ("上海", "深圳")])
        assert sorted(len(x["superseded"]) for x in r) == [0, 1]                    # 只有一次取代生效
        await ms.set_embedding(r[0]["id"], [0.1] * 1024)                              # 向量按文本转 ::vector
        assert await db.pool().fetchval("SELECT vector_dims(embedding) FROM memory_items WHERE id = $1",
                                        r[0]["id"]) == 1024
    with_fresh_db(body)
