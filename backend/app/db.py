"""数据库访问层（asyncpg）。只放 SQL，不放业务流程。

DATABASE_URL 默认指向 docker-compose 里的 PG。
"""

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg

from . import surrogates

DEFAULT_URL = "postgresql://oasis:oasis@127.0.0.1:5433/oasis"  # 用 IP：Windows 上 localhost 解析慢
_pool: asyncpg.Pool | None = None
# O1：本进程这一次启动的身份。重启就换一个，旧进程留下的租约和新进程无关
INSTANCE_ID = uuid4()
LEASE = timedelta(seconds=float(os.getenv("RUN_LEASE_SECONDS", "30")))  # 续租间隔的 3 倍：连丢两次续租也不会被误判
_ACTIVE = "status IN ('running', 'cancelling')"


class RunConflict(Exception):
    """同一会话已有 running 的 run（来自部分唯一索引）。"""


class LeaseLost(Exception):
    """O1：这个 run 已经不归本进程了（租约过期被别的实例接管）。写入被拒，task 应该悄悄退出。"""


def _encode_jsonb(value) -> str:
    """O6 第三轮复核 B（用户选甲）：所有 jsonb 写库前把孤立代理换成 U+FFFD，否则 Postgres 拒收，run 会 internal_error。
    工具结果（外部 API 的 JSON）、用户消息、事件、执行和审批记录都从这里过；text 列不经过这里，没兜住。"""
    return json.dumps(surrogates.scrub(value))


async def _init_conn(conn: asyncpg.Connection) -> None:
    # 让 jsonb 列直接收发 Python dict/list
    await conn.set_type_codec("jsonb", encoder=_encode_jsonb, decoder=json.loads, schema="pg_catalog")


async def _migrate(conn: asyncpg.Connection, schema: str) -> None:
    """O1：整份 schema 再跑一遍要拿表级 ACCESS EXCLUSIVE 锁（ADD COLUMN IF NOT EXISTS 列已存在也拿，实测），
    多条语句又在一个隐式事务里一直持有；正在跑的实例「先写 messages 再更新 runs」顺序相反 → 实测死锁。
    所以 schema 没变就不跑；两个实例同时启动，用事务级 advisory 锁排队。"""
    sha = hashlib.sha256(schema.encode()).hexdigest()
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(7231001)")  # 常数键，只给建表用
        await conn.execute("CREATE TABLE IF NOT EXISTS schema_version (sha text NOT NULL)")
        if await conn.fetchval("SELECT sha FROM schema_version") == sha:
            return
        await conn.execute(schema)
        await conn.execute("DELETE FROM schema_version")
        await conn.execute("INSERT INTO schema_version (sha) VALUES ($1)", sha)


async def connect(url: str | None = None) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(url or os.getenv("DATABASE_URL", DEFAULT_URL), init=_init_conn)
    async with _pool.acquire() as conn:
        await _migrate(conn, (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8"))
    return _pool


async def close() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    assert _pool is not None, "先调用 db.connect()"
    return _pool


# ---------------- 用户与会话 ----------------

async def create_user(name: str, api_token: str) -> UUID:
    return await pool().fetchval(
        "INSERT INTO users (name, api_token) VALUES ($1, $2) RETURNING id", name, api_token)


async def user_id_by_token(api_token: str) -> UUID | None:
    return await pool().fetchval("SELECT id FROM users WHERE api_token = $1", api_token)


async def grant_permissions(user_id: UUID, permissions: list[str]) -> None:
    await pool().executemany("INSERT INTO user_permissions (user_id, permission) VALUES ($1, $2) "
                             "ON CONFLICT DO NOTHING", [(user_id, p) for p in permissions])


async def get_permissions(user_id: UUID) -> list[str]:
    rows = await pool().fetch("SELECT permission FROM user_permissions WHERE user_id = $1", user_id)
    return [r["permission"] for r in rows]


async def get_rules(user_id: UUID) -> list[dict]:
    """M5：本人的规则 + 全局规则（user_id 为空）。判定取最严，所以顺序无所谓。"""
    rows = await pool().fetch("SELECT tool_pattern, conditions, effect FROM tool_rules "
                              "WHERE user_id = $1 OR user_id IS NULL", user_id)
    return [dict(r) for r in rows]


async def add_rule(user_id: UUID | None, tool_pattern: str, conditions: list[dict], effect: str) -> None:
    await pool().execute("INSERT INTO tool_rules (user_id, tool_pattern, conditions, effect) VALUES ($1, $2, $3, $4)",
                         user_id, tool_pattern, conditions, effect)


async def create_conversation(user_id: UUID, title: str | None = None) -> UUID:
    return await pool().fetchval(
        "INSERT INTO conversations (user_id, title) VALUES ($1, $2) RETURNING id", user_id, title)


async def owns_conversation(user_id: UUID, conversation_id: UUID) -> bool:
    return await pool().fetchval(
        "SELECT EXISTS (SELECT 1 FROM conversations WHERE id = $1 AND user_id = $2)",
        conversation_id, user_id)


# ---------------- run 与消息 ----------------

async def start_run(conversation_id: UUID, user_content: list[dict], close_batch_blocks=None) -> UUID:
    """同一事务里建 run + 存用户消息：run 被拒时用户消息也不落库。
    M5：还有没收口的审批批次时，pending 改成 superseded，整批 tool_result 先存一条（API 要求 tool_use 后紧跟
    tool_result），用户消息紧接着另存一条。close_batch_blocks 由调用方传入：行 → tool_result 块。
    O4：以前拼成一条，压缩认不出当前提问；已经存成一条的旧数据不迁移。"""
    async with pool().acquire() as conn, conn.transaction():
        ids, prefix = [], []
        if close_batch_blocks:
            # 先锁批次再建 run（和批准接口同一顺序，否则两边各持一把锁互等 → 死锁）。谁先拿到行锁谁算数
            await conn.execute("SELECT 1 FROM tool_approval WHERE conversation_id = $1 AND result_message_id IS NULL "
                               "ORDER BY id FOR UPDATE", conversation_id)
        run_id = await _new_run(conn, conversation_id)  # 撞唯一索引抛 RunConflict
        if close_batch_blocks:
            await conn.execute("UPDATE tool_approval SET status = 'superseded', decided_at = now() "
                               "WHERE conversation_id = $1 AND result_message_id IS NULL AND status IN ('pending', 'deferred')",
                               conversation_id)
            rows = await open_batch(conversation_id, conn)
            ids, prefix = [r["id"] for r in rows], close_batch_blocks(rows) if rows else []
        insert = ("INSERT INTO messages (conversation_id, run_id, role, content) VALUES ($1, $2, 'user', $3) "
                  "RETURNING id")
        if ids:
            result_id = await conn.fetchval(insert, conversation_id, run_id, prefix)
            await conn.execute("UPDATE tool_approval SET result_message_id = $2 WHERE id = ANY($1::bigint[])",
                               ids, result_id)
        await conn.fetchval(insert, conversation_id, run_id, user_content)
    return run_id


async def start_child_run(parent_conversation_id: UUID, parent_run_id: UUID, tool_use_id: str,
                          title: str, task: str, index: int = 0) -> tuple[UUID, UUID]:
    """M7：子会话 + 子 run + 任务说明消息，同一事务，返回 (子会话, 子 run)。子会话的 user_id 抄父会话的，不来自模型。"""
    async with pool().acquire() as conn, conn.transaction():
        conv = await conn.fetchval(
            "INSERT INTO conversations (user_id, title, parent_conversation_id) "
            "SELECT user_id, $2, id FROM conversations WHERE id = $1 RETURNING id", parent_conversation_id, title)
        run_id = await conn.fetchval(  # O1：子 run 和父同一个属主，续租时跟着父一起续
            "INSERT INTO runs (conversation_id, parent_run_id, parent_tool_use_id, title, child_index, owner_id, "
            "  lease_until) VALUES ($1, $2, $3, $4, $5, $6, now() + $7::interval) RETURNING id",
            conv, parent_run_id, tool_use_id, title, index, INSTANCE_ID, LEASE)
        await conn.execute("INSERT INTO messages (conversation_id, run_id, role, content) VALUES ($1, $2, 'user', $3)",
                           conv, run_id, [{"type": "text", "text": task}])
    return conv, run_id


async def save_run_usage(run_id: UUID, usage: dict) -> None:
    await pool().execute("UPDATE runs SET usage = $2 WHERE id = $1", run_id, usage)


async def child_runs(parent_run_id: UUID) -> list[dict]:
    rows = await pool().fetch("SELECT id, conversation_id, title, status, stop_reason, usage, created_at, finished_at "
                              "FROM runs WHERE parent_run_id = $1 ORDER BY created_at", parent_run_id)
    return [dict(r) for r in rows]


async def list_subagents(conversation_id: UUID) -> list[dict]:
    """M7：一个会话派出过的全部子 run（刷新页面时画胶囊用，以库为准）。"""
    rows = await pool().fetch(
        "SELECT r.id AS run_id, r.parent_tool_use_id AS tool_use_id, r.child_index AS index, r.title, r.status, "
        "  r.stop_reason FROM runs r JOIN conversations c ON c.id = r.conversation_id "
        "WHERE c.parent_conversation_id = $1 ORDER BY r.created_at", conversation_id)
    return [{**dict(r), "run_id": str(r["run_id"])} for r in rows]


# ---------------- 事件日志（M2） ----------------

# 一条语句里「号码 +1」和「插入事件」：同一事务，事件行和 seq 要么都提交、要么都回滚
# O1 fencing：只有属主能拿号。被接管后旧属主再写，UPDATE 0 行 → 不插入 → 抛 LeaseLost，同事务的消息一起回滚
# O2：同一条语句里 NOTIFY 各实例的 hub「这个 run 有新事件」。提交才送出、回滚不送，按提交顺序送达（PG 16 文档 NOTIFY）；
# payload 只带 run_id（默认上限 8000 字节，事件本身可能更长），内容由 hub 按游标拉。插入 0 行时不发
_APPEND = (
    "WITH r AS (UPDATE runs SET last_seq = last_seq + 1 WHERE id = $1 AND owner_id = $4 RETURNING last_seq), "
    "i AS (INSERT INTO run_events (run_id, seq, type, data) SELECT $1, last_seq, $2::text, $3::jsonb FROM r "
    "RETURNING seq) "
    "SELECT seq, pg_notify('run_events', $1::text) FROM i")


async def _append(conn, run_id: UUID, event: dict) -> int:
    seq = await conn.fetchval(_APPEND, run_id, event["type"], event, INSTANCE_ID)
    if seq is None:
        raise LeaseLost(str(run_id))
    return seq


async def append_event(run_id: UUID, event: dict) -> dict:
    """写一条事件，返回带 seq 的事件（拿到 seq 才能推送）。"""
    return {**event, "seq": await _append(pool(), run_id, event)}


async def list_events(run_id: UUID, after_seq: int) -> list[dict]:
    """续播：游标下推到 SQL，走 (run_id, seq) 主键做范围扫描。"""
    rows = await pool().fetch(
        "SELECT seq, data FROM run_events WHERE run_id = $1 AND seq > $2 ORDER BY seq", run_id, after_seq)
    return [{**r["data"], "seq": r["seq"]} for r in rows]


async def _insert_message(conn, conversation_id: UUID, run_id: UUID, role: str, content: list[dict],
                          status: str = "completed", usage: dict | None = None) -> dict:
    message_id = await conn.fetchval(
        "INSERT INTO messages (conversation_id, run_id, role, content, status, usage) "
        "VALUES ($1, $2, $3, $4, $5, $6) RETURNING id", conversation_id, run_id, role, content, status, usage)
    ev = {"type": "message_saved", "message_id": message_id, "role": role}
    return {**ev, "seq": await _append(conn, run_id, ev)}


async def insert_message(conversation_id: UUID, run_id: UUID, role: str, content: list[dict],
                         usage: dict | None = None) -> dict:
    """存消息 + 写 message_saved 放同一事务：前端收到 message_saved 时，这条消息一定查得到。"""
    async with pool().acquire() as conn, conn.transaction():
        return await _insert_message(conn, conversation_id, run_id, role, content, usage=usage)


async def list_messages(conversation_id: UUID, from_id: int = 0) -> list[dict]:
    """from_id：只取水位之后的原文（拼给模型用）；前端展示传 0 取全部。"""
    rows = await pool().fetch(
        "SELECT id, run_id, role, content, status, usage, created_at FROM messages "
        "WHERE conversation_id = $1 AND id >= $2 ORDER BY id", conversation_id, from_id)
    return [dict(r) for r in rows]


# ---------------- 召回（M3） ----------------

# 先把 content 里的 text / tool_result / tool_use 拼成纯文本再搜：直接对 content::text 做 ILIKE，
# 搜 "text"、"type" 这种词会命中 JSON 的键名。thinking 不进搜索；召回工具自己的调用和结果也不进（防自我循环）
_SEARCH = """
SELECT m.id, m.conversation_id, m.role, m.created_at, t.txt
FROM messages m JOIN conversations c ON c.id = m.conversation_id
CROSS JOIN LATERAL (
    SELECT string_agg(CASE b->>'type'
             WHEN 'text' THEN b->>'text'
             WHEN 'tool_result' THEN b->>'content'
             WHEN 'tool_use' THEN (b->>'name') || ' ' || (b->'input')::text END, E'\\n') AS txt
    FROM jsonb_array_elements(m.content) b
    WHERE NOT (b->>'type' = 'tool_use' AND b->>'name' = $5)
      AND NOT (b->>'type' = 'tool_result' AND starts_with(b->>'content', $6))
) t
WHERE c.user_id = $1                                     -- 只搜本人的会话（身份来自服务端，不来自模型）
  AND NOT (m.conversation_id = $2 AND m.id >= c.keep_from_id)  -- 已在上下文里的不返回
  AND t.txt IS NOT NULL                                  -- 全是被排除的块：NOT ILIKE NULL 会让下一句恒真
  AND NOT EXISTS (SELECT 1 FROM unnest($3::text[]) kw WHERE t.txt NOT ILIKE '%' || kw || '%')  -- 每个词都要命中
ORDER BY m.id DESC LIMIT $4
"""


def _escape_like(s: str) -> str:
    """用户的关键词里可能有 % 和 _，不转义就成了通配符。"""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def search_messages(user_id: UUID, conversation_id: UUID, keywords: list[str], *, limit: int,
                          skip_tool: str, skip_mark: str) -> list[dict]:
    rows = await pool().fetch(_SEARCH, user_id, conversation_id, [_escape_like(k) for k in keywords],
                              limit, skip_tool, skip_mark)
    return [dict(r) for r in rows]


# ---------------- 压缩状态（M3） ----------------

async def get_compaction(conversation_id: UUID) -> dict:
    row = await pool().fetchrow(
        "SELECT summary, keep_from_id, compacted_at_id, revision, replay_prefix FROM conversations WHERE id = $1",
        conversation_id)
    return dict(row)


async def commit_compaction(conversation_id: UUID, summary: str, keep_from_id: int, expected_revision: int,
                            replay_prefix: list[dict] | None = None) -> bool:
    """CAS：只有 revision 还是开始压缩前读到的值才写入，否则说明别人抢先了，返回 False。
    compacted_at_id 取此刻最大的消息 id：runner 是先存库再让循环继续，所以历史里的消息此刻都已落库。
    replay_prefix（O4）整份覆盖：普通切点传空，上一次单轮压缩留下的副本随之清掉。"""
    result = await pool().execute(
        "UPDATE conversations SET summary = $2, keep_from_id = $3, revision = revision + 1, replay_prefix = $5, "
        "  compacted_at_id = (SELECT COALESCE(max(id), 0) FROM messages WHERE conversation_id = $1) "
        "WHERE id = $1 AND revision = $4", conversation_id, summary, keep_from_id, expected_revision,
        replay_prefix or [])
    return result == "UPDATE 1"


async def get_loaded_tools(conversation_id: UUID) -> list[str]:
    return await pool().fetchval("SELECT loaded_tools FROM conversations WHERE id = $1", conversation_id)


async def save_loaded_tools(conversation_id: UUID, names: list[str]) -> None:
    """整列覆盖：同一会话同时只有一个 running 的 run（部分唯一索引保证），不会并发写。"""
    await pool().execute("UPDATE conversations SET loaded_tools = $2 WHERE id = $1", conversation_id, names)


async def conversation_user(conversation_id: UUID) -> dict:
    """会话属于谁：{id, name}。召回工具的搜索范围就用这个 id。"""
    row = await pool().fetchrow(
        "SELECT u.id, u.name FROM conversations c JOIN users u ON u.id = c.user_id WHERE c.id = $1", conversation_id)
    return dict(row)


async def _finish(conn, run_id: UUID, status: str, stop_reason: str | None, error: str | None) -> dict | None:
    ev = {"type": "run_end", "status": status, "stop_reason": stop_reason, "error": error}
    result = await conn.execute(  # O1：不是属主就 0 行（被接管的 run 由接管者收尾）
        "UPDATE runs SET status = $2, stop_reason = $3, error = $4, finished_at = now() "
        f"WHERE id = $1 AND {_ACTIVE} AND owner_id = $5", run_id, status, stop_reason, error, INSTANCE_ID)
    if result == "UPDATE 0":
        return None
    return {**ev, "seq": await _append(conn, run_id, ev)}


async def finish_run(run_id: UUID, status: str, stop_reason: str | None, error: str | None = None) -> dict | None:
    """改终态 + 写 run_end 放同一事务。只改 running 的：已结束的返回 None，不会二次覆盖或重复写 run_end。"""
    async with pool().acquire() as conn, conn.transaction():
        return await _finish(conn, run_id, status, stop_reason, error)


async def active_run_id(conversation_id: UUID) -> UUID | None:
    """刷新页面时用：找出这个会话正在跑的 run，前端好重新订阅。"""
    return await pool().fetchval(
        "SELECT id FROM runs WHERE conversation_id = $1 AND status IN ('running', 'cancelling')", conversation_id)


async def get_run(run_id: UUID) -> dict | None:
    row = await pool().fetchrow("SELECT * FROM runs WHERE id = $1", run_id)
    return dict(row) if row else None


# 最后一个 message_saved / retract 之后的 text_delta：只生成了一半、还没成为消息的文字。
# O7 复核①：retract 之前的字已经撤回（那次模型调用作废了），不能在停止 / 接管 / 强制取消时又拼回来落库
_PARTIAL = ("SELECT string_agg(data->>'text', '' ORDER BY seq) FROM run_events "
            "WHERE run_id = $1 AND type = 'text_delta' AND seq > COALESCE("
            "  (SELECT max(seq) FROM run_events WHERE run_id = $1 AND type IN ('message_saved', 'retract')), 0)")


async def save_stopped(conversation_id: UUID, run_id: UUID) -> dict:
    """M5 补 M1 遗留：用户点了停止 → 存一条 status=stopped 的助手消息（带上已生成的半截文字）。
    拼给模型时加「已停止」说明，下一轮模型就不会接着做被停掉的事。"""
    async with pool().acquire() as conn, conn.transaction():
        partial = await conn.fetchval(_PARTIAL, run_id) or ""
        return await _insert_message(conn, conversation_id, run_id, "assistant",
                                     [{"type": "text", "text": partial or "（无输出）"}], "stopped")


async def request_cancel(run_id: UUID) -> list[UUID]:
    """M7：先写库再取消。父和它的子 run 一条语句改成 cancelling；已结束的不动。返回改到的 run。"""
    rows = await pool().fetch("UPDATE runs SET status = 'cancelling' WHERE status = 'running' "
                              "AND (id = $1 OR parent_run_id = $1) RETURNING id", run_id)
    return [r["id"] for r in rows]


async def start_execution(run_id: UUID, tool_use_id: str, tool_name: str, args) -> int:
    return await pool().fetchval("INSERT INTO tool_executions (run_id, tool_use_id, tool_name, args, status) "
                                 "VALUES ($1, $2, $3, $4, 'executing') RETURNING id", run_id, tool_use_id, tool_name, args)


async def finish_execution(execution_id: int, status: str, result: dict | None) -> None:
    await pool().execute("UPDATE tool_executions SET status = $2, result = $3, finished_at = now() "
                         "WHERE id = $1 AND status = 'executing'", execution_id, status, result)


async def spawn_children(conn, parent_run_id: UUID, tool_use_id: str) -> list[dict]:
    """M7：某次 spawn_subagents 派出的子 run，带它的全部消息（O7 复核④：交回父的结论用 context.final_answer
    取，续写过的要拼回来；原来只取最后一条助手消息）和结果未知的写操作。"""
    rows = await conn.fetch(
        "SELECT r.child_index, r.title, r.status, r.stop_reason, "
        "  (SELECT COALESCE(jsonb_agg(jsonb_build_object('role', m.role, 'content', m.content) ORDER BY m.id), '[]') "
        "     FROM messages m WHERE m.run_id = r.id) AS msgs, "
        "  (SELECT COALESCE(jsonb_agg(jsonb_build_object('tool', e.tool_name, 'args', e.args)), '[]') "
        "     FROM tool_executions e WHERE e.run_id = r.id AND e.status = 'outcome_unknown') AS unknown "
        "FROM runs r WHERE r.parent_run_id = $1 AND r.parent_tool_use_id = $2", parent_run_id, tool_use_id)
    return [dict(r) for r in rows]


async def renew_leases(run_ids: list[UUID]) -> int:
    """O1 心跳：只续本进程确实还有 task 在跑的 run 和它们的子 run（用户选 B）。
    按属主续会把「task 因 bug 没收尾就没了」的僵尸 run 永远续下去，会话被唯一索引锁死。"""
    if not run_ids:
        return 0
    result = await pool().execute(
        f"UPDATE runs SET lease_until = now() + $3::interval WHERE owner_id = $1 AND {_ACTIVE} "
        "AND (id = ANY($2::uuid[]) OR parent_run_id = ANY($2::uuid[]))", INSTANCE_ID, run_ids, LEASE)
    return int(result.split()[-1])


async def recover_expired_runs(close_blocks=None) -> int:
    """O1：启动时和周期扫描都调它。只接管租约已过期的 run（属主崩了或卡死），活着的属主在续租，不会被误判。
    改终态的同时补写 run_end 事件：否则续播的前端回放完旧事件后会一直等一个永远不来的结束信号。
    M3：先把最后一个 message_saved 之后的 text_delta 拼成一条 interrupted 助手消息，下一轮模型才知道自己说过什么。"""
    async with pool().acquire() as conn, conn.transaction():
        # 子 run 排前面：先给子定终态，父的汇总才读得到。SKIP LOCKED：两个实例同时扫，各拿各的，不互等
        # lease_until 为空 = O1 之前建的 run，当作过期
        runs = await conn.fetch(f"SELECT id, conversation_id, status, parent_run_id FROM runs WHERE {_ACTIVE} "
                                "AND (lease_until IS NULL OR lease_until < now()) "
                                "ORDER BY parent_run_id IS NULL FOR UPDATE SKIP LOCKED")
        if runs:
            await _settle(conn, runs, close_blocks)
    return len(runs)


async def _settle(conn, runs, close_blocks, *, forced: bool = False) -> None:
    """收尾一批 run（子排前面）：过期接管和强制取消共用。forced = 用户停止后协作取消超时，根 run 记 stopped 消息。"""
    ids = [r["id"] for r in runs]
    # 接管：属主改成自己，下面补写事件才过得了 fencing；旧属主醒来再写就被拒
    await conn.execute("UPDATE runs SET owner_id = $2 WHERE id = ANY($1::uuid[])", ids, INSTANCE_ID)
    # M7：正在执行的 high 风险工具 → 结果未知（不重试）。放最前面：下面给父补汇总时要列出它们
    await conn.execute("UPDATE tool_executions SET status = 'outcome_unknown', finished_at = now() "
                       "WHERE status = 'executing' AND run_id = ANY($1::uuid[])", ids)
    # M5：执行中的审批调用 → 结果未知。一个会话同时只有一个活着的 run，executing 的那行一定属于被接管的 run
    # O10：记下结果未知的时刻（finished_at），按键对账要据此判断「查不到」能不能信
    await conn.execute("UPDATE tool_approval SET status = 'outcome_unknown', finished_at = now() WHERE status = 'executing' "
                       "AND conversation_id = ANY($1::uuid[])", [r["conversation_id"] for r in runs])
    # M8：被接管 run 还开着的 span 标 interrupted（以前启动时全表改，会把别的实例正在跑的 span 也改掉）
    await conn.execute("UPDATE spans SET status = 'interrupted', ended_at = clock_timestamp() "
                       "WHERE status = 'running' AND run_id = ANY($1::uuid[])", ids)
    for r in runs:
        # M7：父历史里没收口的 spawn_subagents 按库里子 run 的状态补一条真 tool_result（由调用方拼）
        if close_blocks and r["parent_run_id"] is None and (
                blocks := await close_blocks(conn, r["conversation_id"], r["id"])):
            await _insert_message(conn, r["conversation_id"], r["id"], "user", blocks)
        partial = await conn.fetchval(_PARTIAL, r["id"])
        # 顺序：message_saved 在 run_end 之前，前端先补消息再收尾
        if forced and r["parent_run_id"] is None and not await open_batch(r["conversation_id"], conn):
            await _insert_message(conn, r["conversation_id"], r["id"], "assistant",  # 同 save_stopped
                                  [{"type": "text", "text": partial or "（无输出）"}], "stopped")
        elif partial:
            await _insert_message(conn, r["conversation_id"], r["id"], "assistant",
                                  [{"type": "text", "text": partial}], "interrupted")
        if r["status"] == "cancelling":  # 停止意图已落库：按用户停止收尾，而不是「不知道发生了什么」
            await _finish(conn, r["id"], "cancelled", "parent_cancel" if r["parent_run_id"] else "user_cancel",
                          "force_cancel" if forced else None)
        else:
            await _finish(conn, r["id"], "failed", "interrupted", None)


async def force_cancel(run_id: UUID, close_blocks=None) -> bool:
    """O1：协作取消超时（有代码吞了取消）。库里直接按用户停止收尾，再把属主换成墓碑 uuid：
    卡住的 task 哪天醒来再写库，被 fencing 拒（LeaseLost），悄悄退出。返回是否真收尾了。"""
    async with pool().acquire() as conn, conn.transaction():
        runs = await conn.fetch(f"SELECT id, conversation_id, status, parent_run_id FROM runs WHERE {_ACTIVE} "
                                "AND owner_id = $2 AND (id = $1 OR parent_run_id = $1) "
                                "ORDER BY parent_run_id IS NULL FOR UPDATE", run_id, INSTANCE_ID)
        if not runs:
            return False
        await _settle(conn, runs, close_blocks, forced=True)
        await conn.execute("UPDATE runs SET owner_id = gen_random_uuid() WHERE id = ANY($1::uuid[])",
                           [r["id"] for r in runs])
    return True


async def cancelling_among(run_ids: list[UUID]) -> list[UUID]:
    """O1：本进程在跑的 run 里，哪些被（任意实例上的）用户点了停止。停止意图就是库里的 cancelling。"""
    if not run_ids:
        return []
    rows = await pool().fetch("SELECT id FROM runs WHERE id = ANY($1::uuid[]) AND status = 'cancelling' "
                              "AND owner_id = $2", run_ids, INSTANCE_ID)
    return [r["id"] for r in rows]


# ---------------- 人在回路确认（M5） ----------------

# 还在等人的行：待确认，或结果未知但用户还没核实
_BLOCKING = "(b.status = 'pending' OR (b.status = 'outcome_unknown' AND b.resolution IS NULL))"


async def suspend_run(conversation_id: UUID, run_id: UUID, message_id: int, items: list[dict]) -> list[dict]:
    """挂起：写整批审批记录 + approval_request 事件 + run_end，同一事务。
    卡片和 run 结束一起提交：用户看到卡片时 run 一定已结束，批准时开续跑不会撞上会话锁。"""
    async with pool().acquire() as conn, conn.transaction():
        await conn.executemany(
            "INSERT INTO tool_approval (conversation_id, run_id, message_id, position, tool_use_id, tool_name, "
            "  args_json, args_sha256, decision, status, result) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            [(conversation_id, run_id, message_id, i["position"], i["tool_use_id"], i["tool_name"],
              i["args_json"], i["args_sha256"], i["decision"], i["status"], i["result"]) for i in items])
        ev = {"type": "approval_request", "approvals": [
            {"tool_use_id": i["tool_use_id"], "tool_name": i["tool_name"], "args": i["args"], "status": i["status"]}
            for i in items if i["status"] == "pending"]}
        ev = {**ev, "seq": await _append(conn, run_id, ev)}
        return [ev, await _finish(conn, run_id, "completed", "awaiting_approval", None)]


async def _new_run(conn, conversation_id: UUID) -> UUID:
    """建 run 时就写上属主和租约：谁建谁跑（发送 / 批准 / 启动续跑都是建完立刻 runner.start）。"""
    try:
        return await conn.fetchval("INSERT INTO runs (conversation_id, owner_id, lease_until) "
                                   "VALUES ($1, $2, now() + $3::interval) RETURNING id",
                                   conversation_id, INSTANCE_ID, LEASE)
    except asyncpg.UniqueViolationError:
        raise RunConflict(str(conversation_id)) from None


async def decide_approval(conversation_id: UUID, tool_use_id: str, set_sql: str, where_sql: str) -> dict | None:
    """人的决定（批准 / 拒绝 / 核实结果未知）。None = 不存在；否则 {status, resolution, updated, run_id}。
    先锁住整批：同一批两个调用被同时决定时，只有后提交的那个看到「没有等人的了」，续跑只开一次。
    建续跑 run 也在这个事务里：撞上会话锁就整体回滚（RunConflict），这次决定不算数。"""
    async with pool().acquire() as conn, conn.transaction():
        message_id = await conn.fetchval(
            "SELECT message_id FROM tool_approval WHERE tool_use_id = $1 AND conversation_id = $2",
            tool_use_id, conversation_id)
        if message_id is None:
            return None
        await conn.execute("SELECT 1 FROM tool_approval WHERE message_id = $1 ORDER BY id FOR UPDATE", message_id)
        row = await conn.fetchrow(  # 条件 UPDATE：状态已经变了（晚到、重复点、另一个标签页）就是 0 行
            f"UPDATE tool_approval SET {set_sql}, decided_at = now() "
            f"WHERE tool_use_id = $1 AND result_message_id IS NULL AND {where_sql} "  # 批次已收口的不再接受决定
            "RETURNING status, resolution, resolved_by", tool_use_id)
        if row is None:
            cur = await conn.fetchrow("SELECT status, resolution, resolved_by FROM tool_approval WHERE tool_use_id = $1", tool_use_id)
            return {**dict(cur), "updated": False, "run_id": None}
        waiting = await conn.fetchval(
            f"SELECT EXISTS (SELECT 1 FROM tool_approval b WHERE b.message_id = $1 AND {_BLOCKING})", message_id)
        run_id = None if waiting else await _new_run(conn, conversation_id)
        return {**dict(row), "updated": True, "run_id": run_id}


async def claim(approval_id: int) -> dict | None:
    """抢执行权：approved → executing。两个 run 同时到这里，PG 行锁让后来者等前者提交后重新判断 WHERE，
    拿到 0 行。返回 None 说明已经不是 approved（别人抢到了，或被拒绝 / 作废），绝不能执行。"""
    row = await pool().fetchrow(
        # O10：deferred（排在要确认的调用后面、本来放行的）和 approved 一样可以抢；keyed 标记这次执行带着幂等键
        "UPDATE tool_approval SET status = 'executing', keyed = true WHERE id = $1 AND status IN ('approved', 'deferred') "
        "RETURNING args_json, args_sha256", approval_id)
    return dict(row) if row else None


async def finish_approval(approval_id: int, status: str, result: dict | None) -> None:
    await pool().execute("UPDATE tool_approval SET status = $2, result = $3, finished_at = now() "
                         "WHERE id = $1 AND status = 'executing'", approval_id, status, result)


async def get_approval(approval_id: int) -> dict:
    return dict(await pool().fetchrow("SELECT *, EXTRACT(EPOCH FROM now() - finished_at)::float AS age_s FROM tool_approval WHERE id = $1", approval_id))


async def open_batch(conversation_id: UUID, conn=None) -> list[dict]:
    """还没收口（tool_result 没写进历史）的那一批，按 position 排好。同一会话最多一批。"""
    rows = await (conn or pool()).fetch(
        "SELECT *, EXTRACT(EPOCH FROM now() - finished_at)::float AS age_s FROM tool_approval WHERE conversation_id = $1 AND result_message_id IS NULL "
        "ORDER BY message_id, position", conversation_id)
    return [dict(r) for r in rows]


async def close_batch(conversation_id: UUID, run_id: UUID, ids: list[int], blocks: list[dict]) -> dict:
    """续跑：把整批 tool_result 存成一条 user 消息，并标记批次已收口，同一事务。"""
    async with pool().acquire() as conn, conn.transaction():
        ev = await _insert_message(conn, conversation_id, run_id, "user", blocks)
        await conn.execute("UPDATE tool_approval SET result_message_id = $2 WHERE id = ANY($1::bigint[])",
                           ids, ev["message_id"])
        return ev


async def list_approvals(conversation_id: UUID) -> list[dict]:
    """刷新页面时用：本会话全部审批记录（不只 pending），前端按 tool_use_id 对到历史里的卡片上。"""
    rows = await pool().fetch(
        "SELECT tool_use_id, tool_name, args_json, decision, status, resolution, resolved_by, result FROM tool_approval "
        "WHERE conversation_id = $1 ORDER BY id", conversation_id)
    return [{**{k: v for k, v in r.items() if k != "args_json"}, "args": json.loads(r["args_json"])} for r in rows]


# O1：原来的 recover_tool_approvals（启动时全表 executing → outcome_unknown）并进了 recover_expired_runs，
# 只改被接管的会话。approved 照旧不动：claim 在执行之前，approved 说明一定还没开始，续跑时照常执行是安全的


async def ready_batches() -> list[tuple[UUID, UUID]]:
    """启动时：已经不等人、但 tool_result 还没写的批次（上次续跑中途崩了）→ 为它们建续跑 run。"""
    convs = await pool().fetch(
        "SELECT DISTINCT t.conversation_id FROM tool_approval t WHERE t.result_message_id IS NULL AND NOT EXISTS ("
        f"  SELECT 1 FROM tool_approval b WHERE b.message_id = t.message_id AND {_BLOCKING})")
    out = []
    for r in convs:
        try:
            async with pool().acquire() as conn:
                out.append((r["conversation_id"], await _new_run(conn, r["conversation_id"])))
        except RunConflict:
            pass
    return out


# ---------------- M6 知识库文档 ----------------

async def create_kb_document(user_id: UUID, title: str) -> UUID:
    return await pool().fetchval(
        "INSERT INTO kb_documents (user_id, title, status) VALUES ($1, $2, 'processing') RETURNING id",
        user_id, title)


async def claim_kb_document(user_id: UUID, doc_id: UUID, title: str | None = None) -> str:
    """更新 / 删除前先「占住」：一条 UPDATE 完成判断加改状态，两个并发请求只有一个能拿到。
    返回 ok / missing（不存在或不是你的）/ busy（还在处理）。"""
    async with pool().acquire() as conn:
        got = await conn.fetchval(
            "UPDATE kb_documents SET status = 'processing', title = COALESCE($3, title), error = NULL, "
            "updated_at = now() WHERE id = $1 AND user_id = $2 AND status <> 'processing' RETURNING id",
            doc_id, user_id, title)
        if got:
            return "ok"
        mine = await conn.fetchval("SELECT 1 FROM kb_documents WHERE id = $1 AND user_id = $2", doc_id, user_id)
        return "busy" if mine else "missing"


async def finish_kb_document(doc_id: UUID, status: str, stats: dict | None = None, error: str | None = None) -> None:
    await pool().execute("UPDATE kb_documents SET status = $2, stats = COALESCE($3, stats), error = $4, "
                         "updated_at = now() WHERE id = $1", doc_id, status, stats, error)


_KB_COLS = "id, title, status, error, stats, created_at, updated_at"


async def get_kb_document(user_id: UUID, doc_id: UUID) -> dict | None:
    row = await pool().fetchrow(f"SELECT {_KB_COLS} FROM kb_documents WHERE id = $1 AND user_id = $2",
                                doc_id, user_id)
    return dict(row) if row else None


async def list_kb_documents(user_id: UUID) -> list[dict]:
    rows = await pool().fetch(f"SELECT {_KB_COLS} FROM kb_documents WHERE user_id = $1 ORDER BY created_at",
                              user_id)
    return [dict(r) for r in rows]


async def delete_kb_document(doc_id: UUID) -> None:
    await pool().execute("DELETE FROM kb_documents WHERE id = $1", doc_id)


async def recover_kb_documents() -> int:
    """启动时调用：还是 processing 的一定是上次进程退出时没做完的。块可能只写了一半，标 failed 让用户重传。"""
    return len(await pool().fetch("UPDATE kb_documents SET status = 'failed', error = 'interrupted', "
                                  "updated_at = now() WHERE status = 'processing' RETURNING id"))



# ---------------- O9：大结果全文、清旧结果的水位 ----------------

async def save_tool_output(ref: str, user_id: UUID, conversation_id: UUID | None, tool_name: str, content: str) -> None:
    await pool().execute(
        "INSERT INTO tool_outputs (id, user_id, conversation_id, tool_name, content) VALUES ($1, $2, $3, $4, $5) "
        "ON CONFLICT (id) DO NOTHING", ref, user_id, conversation_id, tool_name, content)


async def get_tool_output(ref: str, user_id: UUID) -> str | None:
    """只按 (ref, user_id) 查：别人的 ref 和不存在的 ref 一样查不到。"""
    return await pool().fetchval("SELECT content FROM tool_outputs WHERE id = $1 AND user_id = $2", ref, user_id)


async def save_cleared(conversation_id: UUID, run_id: UUID, ev: dict) -> dict:
    """O9 第二步：水位只升不降（GREATEST），和事件同一个事务。"""
    async with pool().acquire() as conn, conn.transaction():
        await conn.execute(
            "UPDATE conversations SET cleared_until_id = GREATEST(cleared_until_id, $2), "
            "  cleared_at_id = GREATEST(cleared_at_id, $3) WHERE id = $1",
            conversation_id, ev["until_id"], ev["at_id"])
        return {**ev, "seq": await _append(conn, run_id, ev)}


async def get_cleared(conversation_id: UUID) -> dict:
    row = await pool().fetchrow("SELECT cleared_until_id, cleared_at_id FROM conversations WHERE id = $1",
                                conversation_id)
    return dict(row)



# ---------------- O10：按幂等键对账 ----------------

async def unresolved_unknown(limit: int = 50) -> list[dict]:
    """O10 复核②：后台对账要扫的行：结果未知、还没人核实、批次没收口、带着幂等键执行的。"""
    rows = await pool().fetch(
        "SELECT a.*, EXTRACT(EPOCH FROM now() - a.finished_at)::float AS age_s, c.user_id FROM tool_approval a "
        "JOIN conversations c ON c.id = a.conversation_id WHERE a.status = 'outcome_unknown' AND a.resolution IS NULL "
        "AND a.result_message_id IS NULL AND a.keyed ORDER BY a.id LIMIT $1", limit)
    return [dict(r) for r in rows]


async def resolve_by_reconcile(approval_id: int, verdict: str) -> bool:
    """系统查到结论：只改还没人核实的结果未知行（和人的决定互斥，先到先得）。"""
    r = await pool().execute(
        "UPDATE tool_approval SET resolution = $2, resolved_by = 'reconcile', decided_at = now() "
        "WHERE id = $1 AND status = 'outcome_unknown' AND resolution IS NULL", approval_id, verdict)
    return r == "UPDATE 1"
