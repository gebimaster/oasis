"""P3：删除对话（硬删除）+ 记忆级联（引用计数、抹内容、恢复旧条）+ 管理页的改 / 删。需要 PG（oasis_test）。"""

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app import db, forget, main, memory_store as ms
from test_db import with_fresh_db


def text(t):
    return {"type": "text", "text": t}


async def said(uid, words, cid=None):
    """一个对话（或已有对话）里用户说一句、run 正常结束。返回 (会话, run, 消息 id)。"""
    cid = cid or await db.create_conversation(uid)
    rid = await db.start_run(cid, [text(words)])
    await db.finish_run(rid, "completed", "end_turn")
    return cid, rid, (await db.list_messages(cid))[-1]["id"]


async def count(sql, *args):
    return await db.pool().fetchval(sql, *args)


def test_delete_conversation_cascades_by_reference_count_and_restores():
    async def body():
        uid = await db.create_user("alice", "ta")
        ca, ra, ma = await said(uid, "我前端用 Vue")
        cb, rb, mb = await said(uid, "前端改用 React 了，另外我对花生过敏")
        _, _, ma2 = await said(uid, "我对花生过敏", ca)
        vue = await ms.add(uid, kind="fact", content="前端用 Vue", source="user_stated", evidence=[ma], actor="extractor")
        react = await ms.add(uid, kind="fact", content="前端改用 React", source="user_stated", evidence=[mb],
                             actor="extractor", run_id=rb, supersedes=[vue["id"]])
        nut = await ms.add(uid, kind="fact", content="对花生过敏", source="user_stated", evidence=[mb, ma2], actor="extractor")
        # 子智能体的子对话、流式事件、大结果、别的对话里冻结的摘要块
        child = await count("INSERT INTO conversations (user_id, parent_conversation_id) VALUES ($1, $2) RETURNING id", uid, cb)
        await said(uid, "子任务", child)
        await db.pool().execute("INSERT INTO run_events (run_id, seq, type, data) VALUES ($1, 999, 'text_delta', '{}')", rb)
        await db.save_tool_output("out_0000000000000001", uid, cb, "read_file", "原文")
        await db.pool().execute("UPDATE conversations SET memory_block = $2 WHERE id = $1", ca, f"- [#{react['id']}] 前端改用 React")

        res = await forget.delete_conversation(uid, cb)
        assert res["conversations"] == 2 and res["revoked"] == [react["id"]] and res["restored"] == [vue["id"]]
        r = await ms.open_item(uid, react["id"])
        assert r["status"] == "revoked" and r["content"] == ms.WIPED and r["evidence"] == []   # 抹掉内容
        v = await ms.open_item(uid, vue["id"])
        assert v["status"] == "active" and v["superseded_by"] is None and v["invalid_at"] is None  # 当 B 没发生过
        assert [e["op"] for e in await ms.events(uid, vue["id"])] == ["add", "supersede", "restore"]
        n = await ms.open_item(uid, nut["id"])
        assert n["status"] == "active" and [m["id"] for m in n["evidence"]] == [ma2]           # 还有别的对话支撑：保留
        assert await count("SELECT memory_block FROM conversations WHERE id = $1", ca) is None  # 冻结块重拼
        for sql in ("SELECT count(*) FROM conversations WHERE id = ANY($1::uuid[])",
                    "SELECT count(*) FROM messages WHERE conversation_id = ANY($1::uuid[])",
                    "SELECT count(*) FROM runs WHERE conversation_id = ANY($1::uuid[])",
                    "SELECT count(*) FROM tool_outputs WHERE conversation_id = ANY($1::uuid[])"):
            assert await count(sql, [cb, child]) == 0
        assert await count("SELECT count(*) FROM run_events WHERE run_id = $1", rb) == 0
        assert len(await db.list_messages(ca)) == 2                                             # 别的对话不动
        assert await forget.delete_conversation(uid, cb) is None
    with_fresh_db(body)


def test_restore_only_when_the_deleted_one_was_current_and_owner_only():
    async def body():
        uid = await db.create_user("alice", "ta")
        bob = await db.create_user("bob", "tb")
        _, _, m1 = await said(uid, "用 Vue")
        cb, _, m2 = await said(uid, "改 React")
        _, _, m3 = await said(uid, "又换 Svelte")
        vue = await ms.add(uid, kind="fact", content="Vue", source="user_stated", evidence=[m1], actor="extractor")
        react = await ms.add(uid, kind="fact", content="React", source="user_stated", evidence=[m2], actor="extractor",
                             supersedes=[vue["id"]])
        await ms.add(uid, kind="fact", content="Svelte", source="user_stated", evidence=[m3], actor="extractor",
                     supersedes=[react["id"]])
        assert await forget.delete_conversation(bob, cb) is None                    # 别人的对话当不存在
        res = await forget.delete_conversation(uid, cb)
        assert res["wiped"] == [react["id"]] and res["revoked"] == [] and res["restored"] == []
        assert (await ms.open_item(uid, react["id"]))["status"] == "superseded"       # 链不断，只抹内容
        assert [i["content"] for i in await ms.active(uid)] == ["Svelte"]             # Vue 不回来
    with_fresh_db(body)


def test_running_conversation_is_not_deleted():
    async def body():
        uid = await db.create_user("alice", "ta")
        cid = await db.create_conversation(uid)
        await db.start_run(cid, [text("还在跑")])
        with pytest.raises(forget.Busy):
            await forget.delete_conversation(uid, cid)
        assert len(await db.list_messages(cid)) == 1
    with_fresh_db(body)


def test_page_edit_and_delete_over_http(monkeypatch):
    async def fake_index(mid, content):
        return False  # 不调向量服务
    monkeypatch.setattr(ms, "index", fake_index)

    async def body():
        uid = await db.create_user("alice", "ta")
        await db.create_user("bob", "tb")
        cid, _, m = await said(uid, "下周三去成都出差")
        until = datetime.now(timezone.utc) + timedelta(days=3)
        trip = await ms.add(uid, kind="commitment", content="周三去成都", source="user_stated", evidence=[m],
                            actor="extractor", valid_until=until)
        await db.pool().execute("UPDATE conversations SET memory_block = $2 WHERE id = $1", cid, f"[#{trip['id']}] 周三去成都")
        A, B = {"Authorization": "Bearer ta"}, {"Authorization": "Bearer tb"}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
            assert (await c.put(f"/memories/{trip['id']}", json={"content": "周四去成都"}, headers=B)).status_code == 404
            r = await c.put(f"/memories/{trip['id']}", json={"content": "周四去成都"}, headers=A)
            new = r.json()["id"]
            assert r.status_code == 200 and r.json()["superseded"] == [trip["id"]]
            item = await ms.open_item(uid, new)
            assert item["source"] == "user_stated" and item["invalid_at"] == until and item["replaced"][0]["id"] == trip["id"]
            assert [(e["op"], e["actor"]) for e in await ms.events(uid, trip["id"])][-1] == ("supersede", "user")
            assert await count("SELECT memory_block FROM conversations WHERE id = $1", cid) is None
            assert (await c.put(f"/memories/{trip['id']}", json={"content": "x"}, headers=A)).status_code == 404  # 已作废
            assert (await c.delete(f"/memories/{new}", headers=B)).status_code == 404
            assert (await c.delete(f"/memories/{new}", headers=A)).status_code == 204
            gone = await ms.open_item(uid, new)
            assert gone["status"] == "revoked" and gone["content"] == ms.WIPED
            assert (await ms.open_item(uid, trip["id"]))["status"] == "superseded"   # 删记忆不恢复旧条
            listed = (await c.get("/memories", headers=A)).json()["memories"]
            assert [x["id"] for x in listed] == [new, trip["id"]] and (await c.get("/memories", headers=B)).json()["memories"] == []
            assert (await c.delete(f"/conversations/{cid}", headers=B)).status_code == 404
            assert (await c.delete(f"/conversations/{cid}", headers=A)).status_code == 200
    with_fresh_db(body)


def test_delete_after_a_real_tool_run_clears_every_table():
    """真跑一个调了工具的 run（假模型），tool_executions / spans / run_events 都有行，删除时外键顺序不能错。"""
    from app import runner
    calls = []

    async def stream(**kw):
        calls.append(1)
        if len(calls) == 1:
            yield {"type": "final", "stop_reason": "tool_use", "usage": {},
                   "content": [{"type": "tool_use", "id": "r1", "name": "recall", "input": {"keywords": ["预算"]}}]}
        else:
            yield {"type": "final", "stop_reason": "end_turn", "usage": {}, "content": [text("没找到")]}

    async def body():
        uid = await db.create_user("alice", "ta")
        cid = await db.create_conversation(uid)
        rid = await db.start_run(cid, [text("之前说的预算是多少")])
        await runner._run(cid, rid, model_stream=stream, limits={"trigger": 10**9, "target": 10**9})
        await db.start_execution(rid, "t9", "transfer", {})  # 只有有副作用的工具才记执行行：直接补一行
        tables = {t: await count(f"SELECT count(*) FROM {t} WHERE run_id = $1", rid)
                  for t in ("tool_executions", "spans", "run_events")}
        assert all(tables.values()), tables  # 确实写了这些表，删除才有意义
        await forget.delete_conversation(uid, cid)
        for t in tables:
            assert await count(f"SELECT count(*) FROM {t} WHERE run_id = $1", rid) == 0
    with_fresh_db(body)


def test_page_edit_can_change_or_clear_the_expiry():
    async def body():
        uid = await db.create_user("alice", "ta")
        _, _, m = await said(uid, "下周三去成都出差")
        old = await ms.add(uid, kind="commitment", content="18 号去成都", source="user_stated", evidence=[m],
                           actor="extractor", valid_until=datetime.now(timezone.utc) + timedelta(days=3))
        day = (datetime.now() + timedelta(days=4)).date().isoformat()
        moved = await ms.edit(uid, old["id"], "19 号去成都", valid_until=day)
        assert ms.fmt_until((await ms.open_item(uid, moved["id"]))["invalid_at"]) == day   # 跟着内容一起改
        cleared = await ms.edit(uid, moved["id"], "长期在成都", valid_until="")
        assert (await ms.open_item(uid, cleared["id"]))["invalid_at"] is None               # 清空 = 没有期限
    with_fresh_db(body)
