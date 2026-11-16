"""P3：删除整个对话（用户选 A：硬删除）。P2 留下的「忘掉只作废了记忆，原文还在」在这里补上。

原文不只在 messages：流式事件、工具执行记录、trace、大结果表、审批批次、子智能体的子对话里都有，
所以一个事务里按外键顺序把这些行一起删掉；引用这些消息作证据的记忆按引用计数级联（memory_store.cascade_messages）。
有 run 还在跑就拒绝（409），不和正在写库的 run 抢。
"""

from uuid import UUID

from app import db, lesson_store, memory_store


class Busy(Exception):
    """对话里还有 run 在跑。"""


async def delete_conversation(user_id: UUID, conversation_id: UUID) -> dict | None:
    """返回级联结果 {conversations, messages, revoked, wiped, restored}；不是本人的或不存在返回 None。"""
    async with db.pool().acquire() as conn, conn.transaction():
        # 自己 + 子智能体的子对话（递归）。FOR UPDATE：并发的 start_run 插 runs 时要等这把锁，删完它的外键检查就失败
        convs = [r["id"] for r in await conn.fetch(
            "WITH RECURSIVE c AS (SELECT id FROM conversations WHERE id = $1 AND user_id = $2 "
            "UNION SELECT k.id FROM conversations k JOIN c ON k.parent_conversation_id = c.id) SELECT id FROM c",
            conversation_id, user_id)]
        if not convs:
            return None
        await conn.execute("SELECT 1 FROM conversations WHERE id = ANY($1::uuid[]) FOR UPDATE", convs)
        if await conn.fetchval("SELECT count(*) FROM runs WHERE conversation_id = ANY($1::uuid[]) "
                               "AND status IN ('running', 'cancelling')", convs):
            raise Busy()
        runs = [r["id"] for r in await conn.fetch("SELECT id FROM runs WHERE conversation_id = ANY($1::uuid[])", convs)]
        msgs = [r["id"] for r in await conn.fetch("SELECT id FROM messages WHERE conversation_id = ANY($1::uuid[])", convs)]
        result = await memory_store.cascade_messages(conn, user_id, msgs)
        result["lessons"] = await lesson_store.cascade(conn, runs, msgs)  # P4：从这些对话提炼的经验一起删
        # 记忆行和事件只是「哪个 run 写的」的记录：run 要删了，断开引用，记忆本身按上面的规则处理
        await conn.execute("UPDATE memory_items SET run_id = NULL WHERE run_id = ANY($1::uuid[])", runs)
        await conn.execute("UPDATE memory_events SET run_id = NULL WHERE run_id = ANY($1::uuid[])", runs)
        for sql in ("DELETE FROM memory_jobs WHERE run_id = ANY($1::uuid[])",
                    "DELETE FROM tool_executions WHERE run_id = ANY($1::uuid[])",
                    "DELETE FROM spans WHERE run_id = ANY($1::uuid[])",
                    "DELETE FROM run_events WHERE run_id = ANY($1::uuid[])"):
            await conn.execute(sql, runs)
        for sql in ("DELETE FROM tool_approval WHERE conversation_id = ANY($1::uuid[])",
                    "DELETE FROM tool_outputs WHERE conversation_id = ANY($1::uuid[])",
                    "DELETE FROM sandbox_contexts WHERE conversation_id = ANY($1::uuid[])",
                    "DELETE FROM messages WHERE conversation_id = ANY($1::uuid[])"):
            await conn.execute(sql, convs)
        await conn.execute("DELETE FROM runs WHERE id = ANY($1::uuid[])", runs)  # 父子 run 在同一条语句里删，自引用外键不挡
        await conn.execute("DELETE FROM conversations WHERE id = ANY($1::uuid[])", convs)
    return {"conversations": len(convs), "messages": len(msgs), **result}
