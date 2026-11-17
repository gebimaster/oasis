"""用户记忆的存储层（P2）：只放 SQL。规则：

- 所有查询都带 user_id；状态过滤（只要 active）写在 SQL 里，不交给模型判断。
- 只作废、不删除（D4=A）：新条目取代旧条目时，旧行改成 superseded 并记下被谁取代，行本身留着。
- 每次变化都在同一个事务里追加一条 memory_events。
- 向量按文本 '[0.1,0.2,...]' 传、在 SQL 里 ::vector 转，不为 asyncpg 另装编解码包。
"""

from datetime import date, datetime, time
from uuid import UUID

from app import db, tail_refs

KINDS = ("preference", "fact", "commitment", "note")
SOURCES = ("user_stated", "agent_inferred", "tool_derived")


def parse_valid_until(s, today: date | None = None) -> datetime | None:
    """有效期（用户选 A：只认用户说出来的时间）。'2026-11-18' → 那一天结束的时刻（本地时区）。
    格式不对、不是字符串、早于今天的一律当没给：宁可不过期，也不让一条记忆因为日期写错而消失。"""
    try:
        d = date.fromisoformat(s) if isinstance(s, str) else None
    except ValueError:
        d = None
    if d is None or d < (today or date.today()):
        return None
    return datetime.combine(d, time(23, 59, 59)).astimezone()


def fmt_until(dt: datetime) -> str:
    return dt.astimezone().strftime("%Y-%m-%d")  # 库里取出来是 UTC，换回本地日期再显示


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


async def _event(conn, user_id, memory_id, op, actor, run_id, detail=None) -> None:
    await conn.execute("INSERT INTO memory_events (user_id, memory_id, op, actor, run_id, detail) "
                       "VALUES ($1, $2, $3, $4, $5, $6)", user_id, memory_id, op, actor, run_id, detail or {})


async def add(user_id: UUID, *, kind: str, content: str, source: str, evidence: list[int],
              actor: str, run_id: UUID | None = None, supersedes: list[int] = (),
              valid_until: datetime | None = None) -> dict:
    """新增一条；supersedes 里的旧条目同时作废。返回 {id, superseded: 实际作废了哪些}。
    valid_until：这条记忆到什么时候为止成立（出差、截止日期）。直接写进 invalid_at，状态仍是 active：
    过期不是「错了」，只是过去了，所以不进摘要块、但还搜得到（用户选 A）。
    旧条目必须是本人的、而且还 active（FOR UPDATE 锁住再判）：并发两次取代同一条，只有一次生效。
    证据消息也必须属于本人，否则整笔回滚（防止模型编一个别人的消息 id 当证据）。"""
    if kind not in KINDS or source not in SOURCES:
        raise ValueError(f"非法的 kind/source：{kind}/{source}")
    async with db.pool().acquire() as conn, conn.transaction():
        mid = await conn.fetchval(
            "INSERT INTO memory_items (user_id, kind, content, source, run_id, invalid_at) "
            "VALUES ($1, $2, $3, $4, $5, $6) RETURNING id", user_id, kind, content, source, run_id, valid_until)
        if evidence:
            ok = await conn.fetchval(
                "SELECT count(*) FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.id = ANY($1::bigint[]) AND c.user_id = $2", list(set(evidence)), user_id)
            if ok != len(set(evidence)):
                raise ValueError("证据消息不存在或不属于该用户")
            await conn.executemany("INSERT INTO memory_evidence (memory_id, message_id) VALUES ($1, $2)",
                                   [(mid, e) for e in sorted(set(evidence))])
        await _event(conn, user_id, mid, "add", actor, run_id, {"supersedes": list(supersedes)})
        done = []
        for old in supersedes:
            row = await conn.fetchrow("SELECT id FROM memory_items WHERE id = $1 AND user_id = $2 "
                                      "AND status = 'active' FOR UPDATE", old, user_id)
            if row is None:
                continue  # 别人的、已作废的、不存在的：一律跳过，不报错（模型引用错编号很常见）
            await conn.execute("UPDATE memory_items SET status = 'superseded', superseded_at = now(), "
                               "superseded_by = $2, invalid_at = LEAST(invalid_at, now()) WHERE id = $1",
                               old, mid)
            await _event(conn, user_id, old, "supersede", actor, run_id, {"by": mid})
            done.append(old)
    return {"id": mid, "superseded": done}


async def revoke(user_id: UUID, memory_id: int, *, actor: str, run_id: UUID | None = None,
                 reason: str = "") -> bool:
    """撤销（用户说「这条不对了 / 忘掉它」）。只能撤销本人 active 的条目；返回是否真的撤销了。"""
    async with db.pool().acquire() as conn, conn.transaction():
        n = await conn.fetchval(
            "WITH u AS (UPDATE memory_items SET status = 'revoked', invalid_at = LEAST(invalid_at, now()) "
            "WHERE id = $1 AND user_id = $2 AND status = 'active' RETURNING 1) SELECT count(*) FROM u",
            memory_id, user_id)
        if n:
            await _event(conn, user_id, memory_id, "revoke", actor, run_id, {"reason": reason})
    return bool(n)


async def set_embedding(memory_id: int, vec: list[float]) -> None:
    await db.pool().execute("UPDATE memory_items SET embedding = $2::vector WHERE id = $1",
                            memory_id, vec_literal(vec))


_embedder = None  # 复用知识库的 bge-m3（SiliconFlow）；测试用 set_embedder 注入假的


def set_embedder(e) -> None:
    global _embedder
    _embedder = e


def embedder():
    global _embedder
    if _embedder is None:
        from app.kb import SiliconFlowEmbedder
        _embedder = SiliconFlowEmbedder()
    return _embedder


async def index(memory_id: int, content: str) -> bool:
    """给新条目算向量。失败只记日志、返回 False：向量是加分项，关键词那一路照样能搜到它。"""
    try:
        [v] = await embedder().embed([content])
        await set_embedding(memory_id, v)
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("memory embedding failed: %s", type(e).__name__)
        return False


async def active(user_id: UUID, *, source: str | None = None) -> list[dict]:
    """本人此刻成立的条目（active 且没过期），新的在前。摘要块（只要 user_stated）和抽取对账都用它。"""
    rows = await db.pool().fetch(
        "SELECT id, kind, content, source, created_at, invalid_at FROM memory_items "
        "WHERE user_id = $1 AND status = 'active' AND (invalid_at IS NULL OR invalid_at > now()) "
        "AND ($2::text IS NULL OR source = $2) ORDER BY id DESC", user_id, source)
    return [dict(r) for r in rows]


async def open_item(user_id: UUID, memory_id: int) -> dict | None:
    """memory_open 用：条目本身 + 证据消息原文 + 取代链（它取代了谁、被谁取代）。别人的条目当不存在。"""
    pool = db.pool()
    item = await pool.fetchrow(
        "SELECT id, kind, content, source, status, valid_from, invalid_at, created_at, superseded_at, superseded_by "
        "FROM memory_items WHERE id = $1 AND user_id = $2", memory_id, user_id)
    if item is None:
        return None
    evidence = await pool.fetch(
        "SELECT m.id, m.role, m.content, m.created_at FROM memory_evidence e JOIN messages m ON m.id = e.message_id "
        "WHERE e.memory_id = $1 ORDER BY m.id", memory_id)
    replaced = await pool.fetch("SELECT id, content FROM memory_items WHERE superseded_by = $1 AND user_id = $2 "
                                "ORDER BY id", memory_id, user_id)
    return {**dict(item), "evidence": [dict(r) for r in evidence], "replaced": [dict(r) for r in replaced]}


async def events(user_id: UUID, memory_id: int) -> list[dict]:
    rows = await db.pool().fetch("SELECT op, actor, run_id, detail, created_at FROM memory_events "
                                 "WHERE user_id = $1 AND memory_id = $2 ORDER BY id", user_id, memory_id)
    return [dict(r) for r in rows]


# ---------------- P3：记忆管理页 + 删除对话的级联（D8=A） ----------------
WIPED = "（内容已删除）"  # 作废并抹掉内容后留下的字：内容本身就是用户原话的转述，删除要连它一起删


async def list_items(user_id: UUID, *, status: str | None = None) -> list[dict]:
    """管理页的列表：本人全部（或某个状态的）条目，新的在前，带证据条数。"""
    rows = await db.pool().fetch(
        "SELECT i.id, i.kind, i.content, i.source, i.status, i.invalid_at, i.created_at, i.superseded_by, "
        "(SELECT count(*) FROM memory_evidence e WHERE e.memory_id = i.id) AS evidence "
        "FROM memory_items i WHERE i.user_id = $1 AND ($2::text IS NULL OR i.status = $2) ORDER BY i.id DESC",
        user_id, status)
    return [dict(r) for r in rows]


async def _forget_blocks(conn, user_id: UUID, ids: list[int]) -> None:
    """冻结的记忆摘要块里逐字写着这些条目：清空，下个 run 重拼（那个对话的缓存失效一次，换来内容真的删掉）。"""
    if ids:
        await conn.execute("UPDATE conversations SET memory_block = NULL WHERE user_id = $1 AND memory_block ~ $2",
                           user_id, r"\[#(" + "|".join(str(i) for i in ids) + r")\]")


async def edit(user_id: UUID, memory_id: int, content: str, kind: str | None = None,
               valid_until: str | None = None) -> dict | None:
    """用户在管理页改内容（用户选 A）：新写一条用户亲口说的取代旧的，旧版本可追溯。只能改本人 active 的条目。
    valid_until：不传（None）沿用旧条目还没过的期限；传 '' 表示没有期限；传日期按 parse_valid_until 校验。"""
    old = await db.pool().fetchrow("SELECT kind, invalid_at FROM memory_items WHERE id = $1 AND user_id = $2 "
                                   "AND status = 'active'", memory_id, user_id)
    if old is None:
        return None
    until = old["invalid_at"] if old["invalid_at"] and old["invalid_at"] > datetime.now(old["invalid_at"].tzinfo) else None
    if valid_until is not None:  # 页面上改了日期：内容写 19 号、期限还是 18 号，会提前一天从摘要块里消失
        until = parse_valid_until(valid_until)
    r = await add(user_id, kind=kind or old["kind"], content=content, source="user_stated", evidence=[],
                  actor="user", supersedes=[memory_id], valid_until=until)
    async with db.pool().acquire() as conn:
        await _forget_blocks(conn, user_id, [memory_id])
    return r


async def forget(user_id: UUID, memory_id: int) -> bool:
    """管理页的「删除」（用户选 A）：作废 + 抹掉内容和向量，不恢复它取代过的旧条目（用户说的是「忘掉」，不是「回到以前」）。"""
    async with db.pool().acquire() as conn, conn.transaction():
        n = await conn.fetchval(
            "WITH u AS (UPDATE memory_items SET status = 'revoked', content = $3, embedding = NULL, "
            "invalid_at = LEAST(invalid_at, now()) WHERE id = $1 AND user_id = $2 AND status = 'active' RETURNING 1) "
            "SELECT count(*) FROM u", memory_id, user_id, WIPED)
        if n:
            await _event(conn, user_id, memory_id, "revoke", "user", None, {"reason": "user_deleted"})
            await _forget_blocks(conn, user_id, [memory_id])
            # P5：尾部里只有占位符，内容抹掉后展开成「已删除」：前缀从这里变了，之前的 thinking 作废
            await tail_refs.bump(conn, user_id, memory_ids=[memory_id])
    return bool(n)


async def cascade_messages(conn, user_id: UUID, message_ids: list[int]) -> dict:
    """删除对话时调用（在调用方的事务里）。按证据引用计数（用户选 A）：
    去掉这些消息的证据；一条证据都不剩的条目抹掉内容，还 active 的改成 revoked；
    被它取代过的旧条目恢复成 active（用户选 A：当那次对话没发生过），链上后面还有更新的就不动。"""
    ids = [r["memory_id"] for r in await conn.fetch(
        "SELECT DISTINCT memory_id FROM memory_evidence WHERE message_id = ANY($1::bigint[])", message_ids)]
    if not ids:
        return {"revoked": [], "wiped": [], "restored": []}
    await conn.execute("DELETE FROM memory_evidence WHERE message_id = ANY($1::bigint[])", message_ids)
    orphans = await conn.fetch(
        "SELECT id, status FROM memory_items i WHERE id = ANY($1::bigint[]) AND user_id = $2 "
        "AND NOT EXISTS (SELECT 1 FROM memory_evidence e WHERE e.memory_id = i.id) FOR UPDATE", ids, user_id)
    wiped = [r["id"] for r in orphans]
    revoked = [r["id"] for r in orphans if r["status"] == "active"]
    await conn.execute("UPDATE memory_items SET content = $2, embedding = NULL, status = CASE WHEN status = 'active' "
                       "THEN 'revoked' ELSE status END, invalid_at = LEAST(invalid_at, now()) "
                       "WHERE id = ANY($1::bigint[])", wiped, WIPED)
    for mid in revoked:
        await _event(conn, user_id, mid, "revoke", "user", None, {"reason": "conversation_deleted"})
    # 取代时 invalid_at 被压成了取代那一刻；恢复时这种改回 NULL（原来的到期日这里丢了，见不足），更早就到期的照旧
    restored = await conn.fetch(
        "UPDATE memory_items SET status = 'active', superseded_by = NULL, superseded_at = NULL, "
        "invalid_at = CASE WHEN invalid_at >= superseded_at THEN NULL ELSE invalid_at END "
        "WHERE user_id = $1 AND status = 'superseded' AND superseded_by = ANY($2::bigint[]) "
        "AND NOT (id = ANY($3::bigint[])) RETURNING id", user_id, revoked, wiped)
    for r in restored:
        await _event(conn, user_id, r["id"], "restore", "user", None, {"after_revoke": True})
    await _forget_blocks(conn, user_id, wiped)
    await tail_refs.bump(conn, user_id, memory_ids=wiped)  # P5：同 forget
    return {"revoked": revoked, "wiped": wiped, "restored": [r["id"] for r in restored]}
