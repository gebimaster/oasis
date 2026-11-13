"""M6 上传 API：异步入库有状态、更新走增量、不能动别人的文档、处理中不许并发改、超长拒收。
需要 PG（同 test_db.py，用全新的 oasis_test 库）；Chroma 和 embedding 用假的。"""

import asyncio

import httpx

from app import db, kb_api, kb_tool, main
from kb_fakes import make_kb
from test_db import with_fresh_db

DOC = "错误码 E1024 表示设备离线，请重启网关。\n\n错误码 E1025 表示认证失败，请重置密钥。"


def api_test(monkeypatch, body):
    """建两个用户、换上假 KB，把 (client, kb, emb) 交给 body。ASGITransport 不跑 lifespan，库由 with_fresh_db 连好。"""
    async def run():
        kb, emb = make_kb()
        monkeypatch.setattr(kb_tool, "deps", lambda: (kb, None))
        alice = await db.create_user("alice", "ta")
        await db.create_user("bob", "tb")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
            await body(c, kb, emb, str(alice))
    with_fresh_db(run)


A, B = {"Authorization": "Bearer ta"}, {"Authorization": "Bearer tb"}


def test_upload_is_async_then_ready_and_isolated(monkeypatch):
    async def body(c, kb, emb, alice):
        r = await c.post("/kb/documents", json={"title": "手册", "text": DOC}, headers=A)
        assert r.status_code == 202 and r.json()["status"] == "processing"
        doc_id = r.json()["doc_id"]
        await kb_api.drain()
        doc = (await c.get(f"/kb/documents/{doc_id}", headers=A)).json()
        assert doc["status"] == "ready" and doc["stats"]["added"] >= 1
        assert (await c.get(f"/kb/documents/{doc_id}", headers=B)).status_code == 404
        assert (await c.get("/kb/documents", headers=B)).json() == []
        hits = await kb.keyword_search(alice, "E1024", 5)  # 块按 alice 的 user_id、服务端 doc_id 落库
        assert hits and hits[0].doc_id == doc_id and hits[0].title == "手册"
    api_test(monkeypatch, body)


def test_put_updates_incrementally_and_only_owner(monkeypatch):
    async def body(c, kb, emb, alice):
        doc_id = (await c.post("/kb/documents", json={"title": "手册", "text": DOC}, headers=A)).json()["doc_id"]
        await kb_api.drain()
        r = await c.put(f"/kb/documents/{doc_id}", json={"title": "手册", "text": DOC + "（已更新）"}, headers=B)
        assert r.status_code == 404  # bob 改不了，也看不出这篇存在
        await c.put(f"/kb/documents/{doc_id}", json={"title": "手册 v2", "text": DOC}, headers=A)
        await kb_api.drain()
        doc = (await c.get(f"/kb/documents/{doc_id}", headers=A)).json()
        assert doc["title"] == "手册 v2" and doc["status"] == "ready"
        assert len((await c.get("/kb/documents", headers=A)).json()) == 1  # 更新不会多出一篇
        hits = await kb.keyword_search(alice, "E1024", 5)
        assert hits and {h.title for h in hits} == {"手册 v2"}  # 改了标题：旧标题的块全换掉了
    api_test(monkeypatch, body)


def test_concurrent_change_while_processing_is_409(monkeypatch):
    async def body(c, kb, emb, alice):
        # 假 embedder 卡在闸门上，后台 ingest 停在 embedding：「处理中」由测试控制，不靠时序
        # （原来只靠「还没 drain」，假 embedder 太快，DELETE 前 ingest 常常已经跑完 → 204，偶发失败）
        emb.gate = asyncio.Event()
        doc_id = (await c.post("/kb/documents", json={"title": "手册", "text": DOC}, headers=A)).json()["doc_id"]
        assert (await c.put(f"/kb/documents/{doc_id}", json={"title": "x", "text": "y"}, headers=A)).status_code == 409
        assert (await c.delete(f"/kb/documents/{doc_id}", headers=A)).status_code == 409
        assert (await c.get(f"/kb/documents/{doc_id}", headers=A)).json()["status"] == "processing"
        emb.gate.set()
        await kb_api.drain()
        assert (await c.delete(f"/kb/documents/{doc_id}", headers=A)).status_code == 204
        assert await kb.keyword_search(alice, "E1024", 5) == []  # 块也删了
        assert (await c.get(f"/kb/documents/{doc_id}", headers=A)).status_code == 404
    api_test(monkeypatch, body)


def test_too_long_is_413_and_nothing_created(monkeypatch):
    async def body(c, kb, emb, alice):
        r = await c.post("/kb/documents", json={"title": "大", "text": "字" * (kb_api.MAX_CHARS + 1)}, headers=A)
        assert r.status_code == 413 and (await c.get("/kb/documents", headers=A)).json() == []
        assert emb.calls == []
    api_test(monkeypatch, body)


def test_ingest_failure_and_restart_recovery(monkeypatch):
    async def body(c, kb, emb, alice):
        emb.fail = True
        doc_id = (await c.post("/kb/documents", json={"title": "手册", "text": DOC}, headers=A)).json()["doc_id"]
        await kb_api.drain()
        doc = (await c.get(f"/kb/documents/{doc_id}", headers=A)).json()
        assert (doc["status"], doc["error"]) == ("failed", "TimeoutError")
        assert await db.claim_kb_document(await db.user_id_by_token("ta"), doc_id) == "ok"  # 模拟进程在处理中退出
        assert await db.recover_kb_documents() == 1
        doc = (await c.get(f"/kb/documents/{doc_id}", headers=A)).json()
        assert (doc["status"], doc["error"]) == ("failed", "interrupted")
    api_test(monkeypatch, body)
