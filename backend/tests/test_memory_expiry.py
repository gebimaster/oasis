"""P2 追加：记忆的有效期（借鉴 mem0 的 expiration_date、EverOS 的 Foresight）。
用户的选择：过期的不进摘要块、但还能搜到并标「已过期」（A）；只认用户说出来的时间，日期不合法就当没给（A）。"""

from datetime import date, datetime, timedelta, timezone

from app import db, memory_extract as mx, memory_search as msr, memory_store as ms, memory_tools as mt, tools
from test_db import with_fresh_db
from test_memory_extract import fake_model
from test_memory_search import FakeEmbedder

SOON = date.today() + timedelta(days=5)          # 测试跑在真实时钟上：用「5 天后」当还没到的日期
AGO = date.today() - timedelta(days=1)


def text(t):
    return {"type": "text", "text": t}


async def _said(cid, t):
    rid = await db.start_run(cid, [text(t)])
    return rid, (await db.list_messages(cid))[-1]["id"]


async def _said2(cid, prev_run, t):
    await db.finish_run(prev_run, "completed", "end_turn")   # 同一会话开下一个 run 前，先把上一个收尾
    return await _said(cid, t)


async def _expire(memory_id):
    """把到期时间拨到一分钟前，模拟「日子过去了」。"""
    await db.pool().execute("UPDATE memory_items SET invalid_at = now() - interval '1 minute' WHERE id = $1", memory_id)


async def _invalid_at(memory_id):
    return await db.pool().fetchval("SELECT invalid_at FROM memory_items WHERE id = $1", memory_id)


def test_parse_valid_until():
    today = date(2026, 11, 13)
    dt = ms.parse_valid_until("2026-11-18", today)
    assert dt.tzinfo is not None and ms.fmt_until(dt) == "2026-11-18" and dt.astimezone().hour == 23  # 那一天结束才到期
    assert ms.parse_valid_until("2026-11-13", today) is not None                 # 今天到期：今天还算数
    for bad in ("2026-11-12", "下周三", "2026/11/18", "", None, 20261118):        # 过去的、不是日期的：都当没给
        assert ms.parse_valid_until(bad, today) is None


def test_expired_leaves_block_but_is_kept_and_searchable():
    async def body():
        emb = FakeEmbedder()
        ms.set_embedder(emb)
        alice = await db.create_user("alice", "tok-a")
        c1 = await db.create_conversation(alice)
        until = ms.parse_valid_until(SOON.isoformat())

        async def put(content, **kw):
            r = await ms.add(alice, kind="commitment", content=content, source="user_stated", evidence=[],
                             actor="extractor", **kw)
            await ms.index(r["id"], content)
            return r["id"]
        trip = await put("用户要去成都出差", valid_until=until)
        pref = await put("用户喜欢深色主题")
        block = mt.render_block(await ms.active(alice, source="user_stated"))
        assert f"用户要去成都出差（{date.today():%Y}" in block and f"到 {SOON} 为止）" in block   # 写明到哪天
        assert block.count("为止") == 1                                              # 长期有效的不写
        tools.current.set(tools.ToolContext(user_id=alice, conversation_id=c1))
        out = await msr._search_tool({"query": "成都出差"})
        assert f"（到 {SOON} 为止）" in out and "已过期" not in out
        assert f"有效期到 {SOON} 为止" in await mt._open({"id": trip})
        assert await mt.tail_notes(alice, block, c1) == [mt.SNAPSHOT_NOTE]

        await _expire(trip)                                                           # —— 日子过去了 ——
        assert [i["id"] for i in await ms.active(alice)] == [pref]                    # 不再算「此刻成立」
        c2 = await db.create_conversation(alice)
        assert "成都" not in mt.render_block(await ms.active(alice, source="user_stated"))   # 新对话的摘要块里没有
        row = await ms.open_item(alice, trip)
        assert row["status"] == "active" and row["invalid_at"] is not None            # 行还在，也没被标成「错了」
        out = await msr._search_tool({"query": "成都出差"})
        assert f"[#{trip}] 用户要去成都出差（已过期，只到" in out                      # 还搜得到，标明已过期
        emb.fail = True                                                               # 两路分开验：只剩关键词这一路
        assert trip in [h["id"] for h in (await msr.search(alice, "成都出差"))[0]]
        emb.fail = False                                                              # 字面不沾边：只可能来自向量这一路
        hits = (await msr.search(alice, "最近的安排"))[0]
        assert trip in [h["id"] for h in hits] and all("keyword" not in h["ranks"] for h in hits)
        assert "已过期，只到" in await mt._open({"id": trip})
        notes = await mt.tail_notes(alice, block, c1)                                 # 旧对话的冻结块里还有它：尾部点名
        assert len(notes) == 2 and f"记忆 #{trip}「用户要去成都出差」已过期（只到" in notes[1]
        assert c2 is not None
    with_fresh_db(body)


def test_supersede_and_revoke_end_a_future_dated_item_now():
    async def body():
        ms.set_embedder(FakeEmbedder())
        alice = await db.create_user("alice", "tok-a")
        until = ms.parse_valid_until(SOON.isoformat())
        a = (await ms.add(alice, kind="commitment", content="周五去成都", source="user_stated", evidence=[],
                          actor="agent", valid_until=until))["id"]
        b = (await ms.add(alice, kind="commitment", content="周五去重庆", source="user_stated", evidence=[],
                          actor="agent", valid_until=until))["id"]
        await ms.add(alice, kind="commitment", content="改成周六去成都", source="user_stated", evidence=[],
                     actor="agent", supersedes=[a])
        await ms.revoke(alice, b, actor="user")
        now = datetime.now(timezone.utc) + timedelta(seconds=5)
        assert await _invalid_at(a) <= now and await _invalid_at(b) <= now          # 不会一直「有效」到原定日期
    with_fresh_db(body)


def test_extractor_sets_expiry_only_from_valid_dates():
    async def body():
        ms.set_embedder(FakeEmbedder())
        uid = await db.create_user("alice", "tok-a")
        cid = await db.create_conversation(uid)
        rid, m = await _said(cid, "我下周要去成都出差，另外我喜欢深色主题")
        await db.insert_message(cid, rid, "assistant", [text("好的")])
        model = fake_model([
            {"op": "ADD", "kind": "commitment", "content": "用户要去成都出差", "evidence": [m],
             "valid_until": SOON.isoformat()},
            {"op": "ADD", "kind": "commitment", "content": "日期写成了中文", "evidence": [m], "valid_until": "下周三"},
            {"op": "ADD", "kind": "commitment", "content": "日期已经过去", "evidence": [m], "valid_until": AGO.isoformat()},
            {"op": "ADD", "kind": "preference", "content": "用户喜欢深色主题", "evidence": [m]}])
        assert await mx.extract(uid, rid, complete=model) == {"add": 4, "update": 0, "delete": 0, "dropped": 0}
        got = {i["content"]: i["invalid_at"] for i in await ms.active(uid)}
        assert ms.fmt_until(got["用户要去成都出差"]) == SOON.isoformat()
        assert got["日期写成了中文"] is None and got["日期已经过去"] is None and got["用户喜欢深色主题"] is None
        assert "valid_until" in mx.SYSTEM
        # 兜底：memory_write 当场写了却没填期限（真实复测 3 次里出现 1 次），抽取器看到它没标「到……为止」，用 UPDATE 补上
        r2, m2 = await _said2(cid, rid, "记一下：我月底前要交报销单")
        bare = (await ms.add(uid, kind="commitment", content="用户月底前要交报销单", source="user_stated",
                             evidence=[m2], actor="agent", run_id=r2))["id"]
        items = await ms.active(uid)
        assert "用户月底前要交报销单\n" in mx.render(items, [(m2, "x")], "") + "\n"      # 列给模型时后面没有期限
        n = [i["id"] for i in items].index(bare) + 1
        fix = fake_model([{"op": "UPDATE", "target": n, "kind": "commitment", "content": "用户月底前要交报销单",
                           "evidence": [m2], "valid_until": SOON.isoformat()}])
        assert (await mx.extract(uid, r2, complete=fix))["update"] == 1
        now = {i["content"]: i for i in await ms.active(uid)}["用户月底前要交报销单"]
        assert now["id"] != bare and ms.fmt_until(now["invalid_at"]) == SOON.isoformat()
        assert (await ms.open_item(uid, bare))["status"] == "superseded" and "补上 valid_until" in mx.SYSTEM
        assert f"用户要去成都出差（到 {SOON} 为止）" in mx.render(await ms.active(uid), [(1, "x")], "")  # 对账时看得到期限
    with_fresh_db(body)


def test_memory_write_valid_until_and_echo_skips_expired():
    async def body():
        ms.set_embedder(FakeEmbedder())
        uid = await db.create_user("alice", "tok-a")
        cid = await db.create_conversation(uid)
        seed = (await ms.add(uid, kind="fact", content="用户住杭州", source="user_stated", evidence=[],
                             actor="extractor"))["id"]
        block = mt.render_block(await ms.active(uid, source="user_stated"))
        rid, _ = await _said(cid, "记住：我过几天要去成都出差")
        tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, run_id=rid))
        out = await mt._write({"kind": "commitment", "content": "用户要去成都出差", "valid_until": SOON.isoformat()})
        trip = (await ms.active(uid))[0]
        assert f"有效期到 {SOON} 为止" in out and ms.fmt_until(trip["invalid_at"]) == SOON.isoformat()
        out = await mt._write({"kind": "commitment", "content": "日期乱写的", "valid_until": "soon"})
        assert "已记住" in out and "有效期" not in out                               # 日期不合法：照常记，不带期限
        echo = (await mt.tail_notes(uid, block, cid))[1]
        assert f"[#{trip['id']}]" in echo
        await _expire(trip["id"])
        echo = (await mt.tail_notes(uid, block, cid))[1]
        assert f"[#{trip['id']}]" not in echo and "日期乱写的" in echo and seed is not None   # 过期的不回显
    with_fresh_db(body)
