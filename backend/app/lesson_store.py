"""经验记忆的存储层（P4）：只放 SQL。规则：

- 所有查询都带 user_id：经验只对本人生效（用户选 A），A 的 run 提炼的经验永远查不到 B 身上。
- 证据必须是本人的消息，否则整笔回滚（和 P2 的 memory_store.add 同一条防线）。
- 状态只往前走：candidate → rejected / active → quarantined；门禁结果整段写进 gate，留作复盘。
- 删除对话时，从它提炼的经验连同证据和使用记录一起删掉（cascade）：来源都没了，经验不该留。
"""

from uuid import UUID

from app import db, memory_store as ms, tail_refs

OUTCOMES = ("failed", "recovered", "corrected")
QUARANTINE_MIN = 3  # 至少违反 3 次、并且违反次数过半，才自动隔离：一两次失败可能只是工具本身抖了一下

_COLS = ("id, user_id, precondition, behavior, expected, verify, tools, rationale, outcome, source_run, "
         "status, gate, uses, violations, created_at, decided_at")


async def add_candidate(user_id: UUID, run_id: UUID, h: dict, evidence: list[tuple[int, str]]) -> int:
    """写一条候选。h 是提炼器给的结构化假设；evidence = [(消息 id, tool_use_id 或 '')]。"""
    if h["outcome"] not in OUTCOMES:
        raise ValueError(f"非法的 outcome：{h['outcome']}")
    async with db.pool().acquire() as conn, conn.transaction():
        lid = await conn.fetchval(
            "INSERT INTO memory_lessons (user_id, precondition, behavior, expected, verify, tools, rationale, "
            "outcome, source_run) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id",
            user_id, h["precondition"], h["behavior"], h["expected"], h["verify"], list(h["tools"]),
            h["rationale"], h["outcome"], run_id)
        mids = sorted({m for m, _ in evidence})
        if mids:
            ok = await conn.fetchval(
                "SELECT count(*) FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.id = ANY($1::bigint[]) AND c.user_id = $2", mids, user_id)
            if ok != len(mids):
                raise PermissionError("证据里有不属于本人的消息")
            await conn.executemany("INSERT INTO lesson_evidence (lesson_id, message_id, tool_use_id) "
                                   "VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                                   [(lid, m, t or "") for m, t in dict.fromkeys(evidence)])
    return lid


async def decide(lesson_id: int, stage: str, result: dict, status: str | None = None) -> None:
    """记一级门禁的结果（stage = l0 / l1）；status 给了就同时改状态。只改还是 candidate 的：已判过的不翻案。"""
    await db.pool().execute(
        "UPDATE memory_lessons SET gate = gate || jsonb_build_object($2::text, $3::jsonb), "
        "status = COALESCE($4, status), decided_at = CASE WHEN $4 IS NULL THEN decided_at ELSE now() END "
        "WHERE id = $1 AND status = 'candidate'", lesson_id, stage, result, status)


async def get(user_id: UUID, lesson_id: int) -> dict | None:
    row = await db.pool().fetchrow(f"SELECT {_COLS} FROM memory_lessons WHERE id = $1 AND user_id = $2",
                                   lesson_id, user_id)
    return dict(row) if row else None


async def by_status(user_id: UUID, status: str) -> list[dict]:
    rows = await db.pool().fetch(f"SELECT {_COLS} FROM memory_lessons WHERE user_id = $1 AND status = $2 "
                                 "ORDER BY id", user_id, status)
    return [dict(r) for r in rows]


async def evidence(user_id: UUID, lesson_id: int) -> list[dict]:
    """证据行 + 原消息 + 对应的工具 span（有的话）。只能看本人的。"""
    rows = await db.pool().fetch(
        "SELECT e.message_id, e.tool_use_id, m.run_id, m.role, m.content, m.created_at, s.id AS span_id, "
        "s.status AS span_status FROM lesson_evidence e JOIN memory_lessons l ON l.id = e.lesson_id "
        "JOIN messages m ON m.id = e.message_id "
        "LEFT JOIN spans s ON e.tool_use_id <> '' AND s.attrs->>'tool_use_id' = e.tool_use_id AND s.run_id = m.run_id "
        "WHERE e.lesson_id = $1 AND l.user_id = $2 ORDER BY e.message_id", lesson_id, user_id)
    return [dict(r) for r in rows]


async def index(lesson_id: int, precondition: str) -> bool:
    """前提的向量；失败了留空，读取时退回关键词那一路。"""
    try:
        [v] = await ms.embedder().embed([precondition])
        await db.pool().execute("UPDATE memory_lessons SET embedding = $2::vector WHERE id = $1",
                                lesson_id, ms.vec_literal(v))
        return True
    except Exception:
        return False


async def record_uses(run_id: UUID, lesson_ids: list[int]) -> None:
    """注入了就记一笔（同一个 run 重复记不重复计数）。"""
    async with db.pool().acquire() as conn, conn.transaction():
        new = await conn.fetch("INSERT INTO lesson_uses (lesson_id, run_id) SELECT unnest($1::bigint[]), $2 "
                               "ON CONFLICT DO NOTHING RETURNING lesson_id", lesson_ids, run_id)
        await conn.execute("UPDATE memory_lessons SET uses = uses + 1 WHERE id = ANY($1::bigint[])",
                           [r["lesson_id"] for r in new])


async def judge_uses(run_id: UUID, failed_tools: set[str]) -> list[int]:
    """run 结束时判：注入的经验管的工具在这个 run 里失败了，就算一次违反。违反够多就隔离，返回被隔离的编号。"""
    async with db.pool().acquire() as conn, conn.transaction():
        rows = await conn.fetch("SELECT u.lesson_id, l.tools FROM lesson_uses u JOIN memory_lessons l "
                                "ON l.id = u.lesson_id WHERE u.run_id = $1 AND u.violated IS NULL FOR UPDATE OF u",
                                run_id)
        bad = [r["lesson_id"] for r in rows if set(r["tools"]) & failed_tools]
        await conn.execute("UPDATE lesson_uses SET violated = (lesson_id = ANY($2::bigint[])) "
                           "WHERE run_id = $1 AND violated IS NULL", run_id, bad)
        await conn.execute("UPDATE memory_lessons SET violations = violations + 1 WHERE id = ANY($1::bigint[])", bad)
        gone = await conn.fetch(
            "UPDATE memory_lessons SET status = 'quarantined', decided_at = now() WHERE id = ANY($1::bigint[]) "
            "AND status = 'active' AND violations >= $2 AND violations * 2 > uses RETURNING id", bad, QUARANTINE_MIN)
        await _bump_owners(conn, [r["id"] for r in gone])  # P5 复核二：隔离也让展开结果变了
    return [r["id"] for r in gone]


async def _bump_owners(conn, ids: list[int]) -> None:
    """P5：尾部里的经验是占位符，删除 / 隔离后展开成「已停用或已删除」，引用它的对话前缀变了，按主人分组抬水位。"""
    owners: dict = {}
    for r in await conn.fetch("SELECT id, user_id FROM memory_lessons WHERE id = ANY($1::bigint[])", ids):
        owners.setdefault(r["user_id"], []).append(r["id"])
    for user_id, lids in owners.items():
        await tail_refs.bump(conn, user_id, lesson_ids=lids)


async def cascade(conn, runs: list[UUID], message_ids: list[int]) -> int:
    """删除对话时调用（同一个事务里）：源 run 或证据消息要被删的经验整条删掉；被删 run 的使用记录也删。返回删了几条经验。"""
    ids = [r["id"] for r in await conn.fetch(
        "SELECT id FROM memory_lessons WHERE source_run = ANY($1::uuid[]) UNION "
        "SELECT lesson_id FROM lesson_evidence WHERE message_id = ANY($2::bigint[])", runs, message_ids)]
    await conn.execute("DELETE FROM lesson_uses WHERE lesson_id = ANY($1::bigint[]) OR run_id = ANY($2::uuid[])",
                       ids, runs)
    await conn.execute("DELETE FROM lesson_evidence WHERE lesson_id = ANY($1::bigint[])", ids)
    await conn.execute("DELETE FROM lesson_scans WHERE run_id = ANY($1::uuid[])", runs)
    await _bump_owners(conn, ids)  # P5：删除前查主人（删了就查不到了）
    await conn.execute("DELETE FROM memory_lessons WHERE id = ANY($1::bigint[])", ids)
    return len(ids)
