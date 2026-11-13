"""M8 第 2 步：trace（调用链追踪）。存 PG 的 spans 表（用户选 a），概念照 OpenTelemetry：trace_id / parent / attrs。

树的形状：run → llm（每轮一次）/ tool（按时间排在 llm 后面，是兄弟不是儿子：工具在 LLM 调用结束后才跑，
挂成儿子会跑出父的时间段）→ 子 run（挂在派它的 spawn_subagents 工具 span 下）→ 子 run 自己的 llm / tool。
父子关系靠 ContextVar：run 开始时设成 run span，工具执行期间设成工具 span；gather 给每个子任务拷一份上下文，
所以子 run 一开始读到的就是派它的那个工具 span，兄弟之间互不干扰。
不在 run 里（单测直接调工具，run_id 为空）时什么都不记（没启用即 no-op）。
"""

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import AsyncIterator, Callable
from uuid import UUID, uuid4

from app import db


@dataclass(frozen=True)
class Span:
    id: UUID
    trace_id: UUID
    run_id: UUID


current: ContextVar[Span | None] = ContextVar("trace_span", default=None)


async def start(kind: str, name: str, run_id: UUID, *, parent: Span | None = None, attrs: dict | None = None) -> Span:
    """没有父 span 就是根：trace_id = 本 run 的 id。"""
    s = Span(uuid4(), parent.trace_id if parent else run_id, run_id)
    await db.pool().execute(
        "INSERT INTO spans (id, trace_id, parent_id, run_id, kind, name, attrs) VALUES ($1, $2, $3, $4, $5, $6, $7)",
        s.id, s.trace_id, parent.id if parent else None, run_id, kind, name, attrs or {})
    return s


async def end(s: Span, status: str, attrs: dict | None = None, error: str | None = None) -> None:
    """WHERE status='running'：只收尾一次（取消路径和正常路径都可能走到）。"""
    await db.pool().execute(
        "UPDATE spans SET status = $2, attrs = attrs || $3, error = $4, ended_at = clock_timestamp() "
        "WHERE id = $1 AND status = 'running'", s.id, status, attrs or {}, error)


@asynccontextmanager
async def span(kind: str, name: str, run_id: UUID | None, attrs: dict | None = None) -> AsyncIterator[dict]:
    """包住一段工作。yield 出一个 dict：里面写的键收尾时并进 attrs；写 status / error 可以覆盖结果
    （工具报错不抛异常，只返回 is_error，要靠它标 error）。"""
    out: dict = {}
    if run_id is None:
        yield out
        return
    s = await start(kind, name, run_id, parent=current.get(), attrs=attrs)
    token = current.set(s)
    try:
        yield out
    except Exception as e:
        await end(s, "error", out, f"{type(e).__name__}: {e}"[:300])
        raise
    except BaseException:  # 取消（停止 / 子任务超时）、生成器被关闭：shield 保证这一笔更新写完
        await asyncio.shield(end(s, "cancelled", out))
        raise
    else:
        status, error = out.pop("status", "ok"), out.pop("error", None)
        await end(s, status, out, error)
    finally:
        current.reset(token)


RUN_STATUS = {"completed": "ok", "failed": "error", "cancelled": "cancelled"}


def traced_llm(inner: Callable | None, run_id: UUID) -> Callable:
    """包一层模型调用：每轮一个 llm span，记 usage、stop_reason，O3 加本轮命中率和首字时间。循环本身不碰库，所以在 runner 这里包。
    inner 为空时调用时再取 llm.stream_message（测试用 monkeypatch 替换的是模块属性）。"""
    from app import budget, llm  # 避免 import 环
    turn = 0

    async def stream(**kw):
        nonlocal turn
        turn += 1
        fn = inner or llm.stream_message
        async with span("llm", f"llm #{turn}", run_id, {"turn": turn, "final_turn": kw["tool_choice"]["type"] == "none"}) as out:
            async for ev in fn(**kw):
                if ev["type"] == "final":
                    out.update(stop_reason=ev["stop_reason"], usage=ev.get("usage"),
                               hit_rate=budget.hit_rate([ev.get("usage") or {}]),
                               first_token_ms=ev.get("first_token_ms"))
                yield ev
    return stream


# O1：原来的 recover()（启动时全表 running → interrupted）并进了 db.recover_expired_runs，只改被接管 run 的 span


def build_tree(rows: list[dict]) -> list[dict]:
    """平铺的 span 行 → 树（按开始时间排）。父不在这批里的当根（只取了子 trace 的一段时也能画）。"""
    nodes = {r["id"]: {**r, "children": []} for r in sorted(rows, key=lambda r: (r["started_at"], str(r["id"])))}
    roots = []
    for n in nodes.values():
        end_at = n.get("ended_at")
        n["duration_ms"] = round((end_at - n["started_at"]).total_seconds() * 1000) if end_at else None
        (nodes[n["parent_id"]]["children"] if n.get("parent_id") in nodes else roots).append(n)
    return roots


async def load(trace_id: UUID) -> list[dict]:
    rows = await db.pool().fetch("SELECT * FROM spans WHERE trace_id = $1", trace_id)
    return build_tree([dict(r) for r in rows])
