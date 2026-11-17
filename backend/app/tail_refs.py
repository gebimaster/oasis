"""P5 复核二（用户选 B）：尾部里的记忆、经验只存占位符，这里按编号查现值、在删除时抬水位。

为什么不在删除时去擦（复核一的做法）：原文一旦落库，副本会跑到擦除找不到的地方，比如单轮压缩的 replay_prefix、
擦完以后才写库的尾部、正在跑的 run 的内存。只存占位符，库里就没有原文可漏。

剩下的问题是 Claude 的 thinking：展开结果变了（删了、隔离了），那个尾部之后的 thinking 签名就过期了。
- 删除时：引用了这些编号的对话，cleared_at_id 抬到当前最大消息 id（bump）；
- 正在跑的 run：循环每轮重查，发现变了就抬自己的水位，并发事件让 runner 写库；
- 删除发生在 run 的最后一次调用期间：runner 在 run 结束后再查一次（changed）。
"""

from uuid import UUID

from app import db


def lesson_body(x: dict) -> str:
    """经验的展开文字（lesson_tools.line 也用它，两边一致）。"""
    src = "（来自失败的 run）" if x["outcome"] == "failed" else ""
    return f"前提：{x['precondition']}；做法：{x['behavior']}；预期：{x['expected']}{src}"


async def lookup(user_id: UUID, keys) -> dict:
    """{(种类, 编号): 现值}。只查本人的；查不到、抹掉了、经验不是 active 的都不放（展开时按「已删除」）。"""
    from app.memory_store import WIPED  # 避免循环导入：memory_store 要 import 这里的 bump
    mids = [i for k, i in keys if k == "m"]
    lids = [i for k, i in keys if k == "l"]
    out = {}
    if mids:
        for r in await db.pool().fetch("SELECT id, content FROM memory_items WHERE user_id = $1 "
                                       "AND id = ANY($2::bigint[]) AND content <> $3", user_id, mids, WIPED):
            out[("m", r["id"])] = r["content"]
    if lids:
        for r in await db.pool().fetch("SELECT id, precondition, behavior, expected, outcome FROM memory_lessons "
                                       "WHERE user_id = $1 AND id = ANY($2::bigint[]) AND status = 'active'",
                                       user_id, lids):
            out[("l", r["id"])] = lesson_body(dict(r))
    return out


async def changed(user_id: UUID, used: dict) -> bool:
    """run 用过的展开值（循环写在 exp_state 里）和库里现值比，有不一样的就是 True。"""
    if not used:
        return False
    now = await lookup(user_id, set(used))
    return any(now.get(k) != v for k, v in used.items())


_CONVS = """
SELECT DISTINCT m.conversation_id FROM messages m JOIN conversations c ON c.id = m.conversation_id
WHERE c.user_id = $1 AND m.role = 'user' AND EXISTS (
    SELECT 1 FROM jsonb_array_elements(m.content) b
    WHERE b ? '_tail' AND (
        EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(b->'_tail'->'refs'->'m', '[]'::jsonb)) x
                WHERE CASE WHEN x ~ '^[0-9]{1,18}$' THEN x::bigint END = ANY($2::bigint[]))
     OR EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(b->'_tail'->'refs'->'l', '[]'::jsonb)) x
                WHERE CASE WHEN x ~ '^[0-9]{1,18}$' THEN x::bigint END = ANY($3::bigint[]))))
"""
# 复核三：先用正则挡掉不是 1~18 位数字的 ref 再转 bigint（CASE 保证先判断后转换；AND 不保证求值顺序），
# 库里万一有坏数据，也只是这一行不参与匹配，不会让整个删除事务回滚


async def bump(conn, user_id: UUID, *, memory_ids=(), lesson_ids=()) -> list[UUID]:
    """删除 / 隔离之后：引用了这些编号的对话，前缀从下一次请求起会变，之前的 thinking 作废。返回涉及的对话。"""
    if not memory_ids and not lesson_ids:
        return []
    convs = [r["conversation_id"] for r in await conn.fetch(_CONVS, user_id, list(memory_ids), list(lesson_ids))]
    if convs:
        await conn.execute(
            "UPDATE conversations c SET cleared_at_id = GREATEST(c.cleared_at_id, "
            "(SELECT COALESCE(max(id), 0) FROM messages WHERE conversation_id = c.id)) WHERE c.id = ANY($1::uuid[])",
            convs)
    return convs
