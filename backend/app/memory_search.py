"""记忆检索（P2 第 5 步）：关键词 + 向量两路，RRF 合并。状态过滤（只要 active）和用户隔离都写在 SQL 里。

- 关键词（用户选 A）：中文按两字一组切（Lucene CJK 的老办法），英文、数字按词；每组单独 ILIKE，命中组数多的排前。
  含虚字（的、么、我……）的组和「用户」直接丢掉：记忆都写成「用户……」，留着它每条都会命中。
- 向量：bge-m3 余弦相似度。实测（7 条记忆 × 15 个问法）相关最低 0.424、无关最高 0.679，rerank 也分不开：
  「下午茶零食 → 花生过敏」这种要推理才连得上的，rerank 分是 0.0000，比无关的「车牌 → 工位」0.056 还低。
  所以不硬过滤（用户最后选 A）：只留 0.4 的垃圾线，RRF 取前几条交给模型判断；隐含的关联靠常驻的摘要块。
- 一路挂了另一路照样出结果，结果里标明「向量这一路没搜」，模型就不会把「没搜到」当成「没有」。
- 过期的条目（有效期已过，status 仍是 active）照样能搜到，结果里标「已过期」：它当时是真的，只是过去了（用户选 A）。
  被取代、被撤销的才不返回。
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from app import db, memory_store as ms, tools
from app.kb_search import rrf

MIN_SIM = float(os.getenv("MEMORY_MIN_SIM", "0.4"))  # 垃圾线，不是相关性门槛：无关平均 0.43，相关最低 0.424
STOP_CHARS = set("的了吗呢吧啊么什哪怎我你他她它这那是")
STOP_TERMS = {"用户"}
MAX_TERMS, PER_PATH = 24, 10
_TOKEN = re.compile(r"[一-鿿]+|[A-Za-z]+|\d+")


def terms(query: str) -> list[str]:
    """「我喜欢什么主题」→ [喜欢, 主题]。单个汉字成段时保留（「猫」），英文少于 2 个字母的丢掉。"""
    out: list[str] = []
    for tok in _TOKEN.findall(query):
        if "一" <= tok[0] <= "鿿":
            grams = [tok] if len(tok) == 1 else [tok[i:i + 2] for i in range(len(tok) - 1)]
            out += [g for g in grams if not (set(g) & STOP_CHARS) and g not in STOP_TERMS]
        elif tok.isdigit() or len(tok) >= 2:
            out.append(tok.lower())
    return list(dict.fromkeys(out))[:MAX_TERMS]  # 去重保序


_COLS = "id, kind, content, source, created_at, invalid_at"
_KEYWORD = f"""
SELECT * FROM (
    SELECT {_COLS}, (SELECT count(*) FROM unnest($2::text[]) t
                     WHERE content ILIKE '%' || t || '%' ESCAPE '\\') AS score
    FROM memory_items WHERE user_id = $1 AND status = 'active'
) s WHERE score > 0 ORDER BY score DESC, id DESC LIMIT $3"""
_VECTOR = f"""
SELECT {_COLS}, 1 - (embedding <=> $2::vector) AS score
FROM memory_items WHERE user_id = $1 AND status = 'active' AND embedding IS NOT NULL
  AND 1 - (embedding <=> $2::vector) >= $3
ORDER BY embedding <=> $2::vector LIMIT $4"""


@dataclass
class MemHit:
    id: int     # rrf 只用 id
    row: dict


async def search(user_id: UUID, query: str, k: int = 5) -> tuple[list[dict], list[str]]:
    """返回 (结果, 降级标记)。结果按 RRF 排序，每条带 ranks（各路名次）。"""
    pool, paths, flags = db.pool(), {}, []
    if kw := terms(query):
        rows = await pool.fetch(_KEYWORD, user_id, [db._escape_like(t) for t in kw], PER_PATH)
        paths["keyword"] = [MemHit(r["id"], dict(r)) for r in rows]
    try:
        [v] = await ms.embedder().embed([query])
        rows = await pool.fetch(_VECTOR, user_id, ms.vec_literal(v), MIN_SIM, PER_PATH)
        paths["vector"] = [MemHit(r["id"], dict(r)) for r in rows]
    except Exception:  # embedding 服务挂了：只用关键词，标出来
        flags.append("vector_failed")
    fused = rrf(paths)[:k]
    return [{**f.hit.row, "ranks": f.ranks} for f in fused], flags


async def _search_tool(args: dict) -> str:
    ctx = tools.context()
    if ctx.user_id is None:
        raise LookupError("memory_search 缺少服务端注入的用户身份")
    hits, flags = await search(ctx.user_id, args["query"], k=args.get("limit", 5))
    note = "（向量检索这次失败了，只按字面搜了，没搜到不代表没有）" if "vector_failed" in flags else ""
    if not hits:
        return f"没有找到相关的长期记忆。{note}"
    lines = [f"找到 {len(hits)} 条长期记忆（按相关程度排序，但不保证都相关，请自己判断；# 后是编号，"
             f"可用 memory_open 看来源原文）。{note}"]
    for h in hits:
        tag = "" if h["source"] == "user_stated" else "（推断，用户没亲口说过）"
        if h["invalid_at"]:  # 带有效期的写明到哪天；已经过了的标出来，别当成现状
            gone = h["invalid_at"] <= datetime.now(timezone.utc)
            tag += f"（{'已过期，只' if gone else ''}到 {ms.fmt_until(h['invalid_at'])} 为止）"
        lines.append(f"- [#{h['id']}] {h['content']}{tag}（{h['created_at']:%Y-%m-%d}）")
    return "\n".join(lines)


tools.register(tools.Tool(
    name="memory_search",
    description=("在这位用户的长期记忆里搜索（偏好、事实、约定、要求记下的事），不返回已被取代或撤销的条目；"
                 "有效期已过的会标「已过期」，它们是过去的事，不是现状。"
                 "<user_memory> 摘要里没有、或者想确认最新情况时使用。query 用一句自然语言描述要找什么。"),
    input_schema={"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 200},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
        "required": ["query"], "additionalProperties": False},
    handler=_search_tool, permission=tools.PUBLIC, risk=tools.LOW, concurrency_safe=True))
