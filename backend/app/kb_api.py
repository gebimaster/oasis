"""知识库上传的后台处理（M6）：路由在 main.py，这里只管「入库 → 写回状态」。

用户的三个选择：① 异步（202 + 查状态）② doc_id 服务端生成 ③ 正文限 MAX_CHARS 字。
②b 的代价：重新上传如果再 POST，会多出一份旧版本，检索时新旧答案打架。所以另给 PUT /kb/documents/{id}
做更新，走第 1 步的增量 ingest（只重算变了的块）。
"""

import asyncio
import logging
from uuid import UUID

from app import db, kb_tool

log = logging.getLogger(__name__)
# 块数看条目长短（O5 后不短于 20 字的条目不再合并，第三轮复核改的估算）：拿评测语料拼满 10 万字约 1600 块 → 51 次
# embedding 调用（EMBED_BATCH=32）；全是 150 字的段落约 660 块 → 21 次。50MB 的 txt 会是几千次
MAX_CHARS = 100_000
_tasks: set[asyncio.Task] = set()  # 持有引用，防止后台 task 被垃圾回收


async def _ingest(user_id: UUID, doc_id: UUID, title: str, text: str) -> None:
    try:
        kb, _ = kb_tool.deps()
        stats = await kb.ingest(str(user_id), str(doc_id), text, title)
        await db.finish_kb_document(doc_id, "ready", stats=stats)
    except Exception as e:
        log.warning("知识库文档 %s 入库失败：%s", doc_id, type(e).__name__)  # 不记原文，防止回显凭据
        await db.finish_kb_document(doc_id, "failed", error=type(e).__name__)


def start(user_id: UUID, doc_id: UUID, title: str, text: str) -> None:
    """调用前文档必须已是 processing（create 或 claim 过），保证同一篇同时只有一个 ingest 在跑。"""
    task = asyncio.create_task(_ingest(user_id, doc_id, title, text))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def drain() -> None:
    """等所有后台入库结束（测试用）。"""
    while _tasks:
        await asyncio.gather(*list(_tasks), return_exceptions=True)


async def delete(user_id: UUID, doc_id: UUID) -> None:
    """先删块再删登记行：删块失败时登记行还在（标 failed），用户可以重试；反过来会留下查不到归属的孤儿块。"""
    kb, _ = kb_tool.deps()
    try:
        await kb.delete_document(str(user_id), str(doc_id))
    except Exception as e:
        await db.finish_kb_document(doc_id, "failed", error=type(e).__name__)
        raise
    await db.delete_kb_document(doc_id)
