"""M9：沙箱池。一个用户一个 Cube 沙箱，每个会话在里面一个代码上下文（第 2、3 步讨论的结果）。

- 惰性创建：第一次调用沙箱工具才建。同一用户加锁，DB 的 user_id 主键再兜一层，并发首调不会建出两个。
- 上限：全局最多 SANDBOX_LIMIT 个。满了先回收最久没用的空闲沙箱；都在用就抛 SandboxBusy（代码没开始，可以重试）。
- 回收两层：reap() 杀空闲超过 IDLE_SECONDS 的；创建时给 Cube 一个 TTL、每次使用续期，进程死了 Cube 到点自己回收。
- 先写库：执行前置 needs_reset=true，正常结束才清。取消 / 超时 / 断连时 shield 着重启内核；
  重启失败或进程直接崩了，标记留在库里，下次用之前先重启（对应 M7「先写库再取消」）。
- 对账：启动时按 metadata 列出 Cube 里我们的沙箱。孤儿（库里没有或用户对不上）杀掉，库里有但 Cube 没了的删行，
  两边都有的留着复用（用户选 b）。
- sandbox_id 只在服务端流转，身份取自 ToolContext，模型传不进来。
"""

import asyncio
import logging
import os
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from app import db

log = logging.getLogger(__name__)
# 实测（2026-11-02）：16GB 虚拟机上第 7 个沙箱能建出来，但内核起不来（建上下文 500），再多就报 no more resource。
# 讨论时定的 10 超过了后端真实容量，评测里每题一个新用户把它撑满 → 改 5，留余量。上限必须 ≤ 后端容量
SANDBOX_LIMIT = int(os.getenv("SANDBOX_LIMIT", "5"))
IDLE_SECONDS = 30 * 60
TTL_SECONDS = 35 * 60  # 比空闲回收多 5 分钟：平时由 reap 杀，进程死了才轮到 Cube 兜底
RESET_TIMEOUT = 5      # 取消时重启内核最多等这么久，等不到就把标记留给下次

REBUILT = "沙箱已重建（空闲回收或失效）：之前的变量和文件都没有了。"
RESTARTED = "上次执行被中断，内核已重启：之前的变量没有了，文件还在。"


class SandboxBusy(Exception):
    """沙箱数到上限且都在使用：代码没开始执行，可以重试。"""


@dataclass
class Output:
    """一次执行的结果，后端把 SDK 的对象转成这个，工具层只认它（假后端也好造）。"""
    stdout: str = ""
    stderr: str = ""
    result: str | None = None      # run_code：最后一行表达式的值
    error: str | None = None       # run_code：「类型: 信息」+ traceback 尾部，用户代码报错（不是基础设施错误）
    images: list[bytes] = field(default_factory=list)
    exit_code: int | None = None   # run_shell：非 0 也是正常返回，不抛异常


class Backend(Protocol):
    """沙箱后端。真实实现包 e2b 的 AsyncSandbox（第 2 段），测试用假的。
    执行超时、断连这类基础设施错误直接抛异常；用户代码的错误放在 Output 里。"""
    async def run_code(self, sandbox_id: str, context_id: str, code: str, timeout: float) -> Output: ...
    async def run_shell(self, sandbox_id: str, command: str, timeout: float) -> Output: ...
    async def read_file(self, sandbox_id: str, path: str) -> str: ...
    async def write_file(self, sandbox_id: str, path: str, data: str | bytes) -> None: ...
    async def create(self, user_id: str) -> str: ...
    async def alive(self, sandbox_id: str) -> bool: ...
    async def kill(self, sandbox_id: str) -> None: ...
    async def list_ours(self) -> dict[str, str]: ...  # sandbox_id → metadata 里的 user_id
    async def keepalive(self, sandbox_id: str, seconds: int) -> None: ...
    async def new_context(self, sandbox_id: str) -> str: ...
    async def restart_context(self, sandbox_id: str, context_id: str) -> bool: ...  # False = 上下文已经不在了


@dataclass
class Lease:
    sandbox_id: str
    context_id: str
    notes: list[str]  # 要告诉模型的状态变化（沙箱重建 / 内核重启），工具拼在结果前面


class Pool:
    def __init__(self, backend: Backend):
        self.backend = backend
        self._user_locks: defaultdict[UUID, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._conv_locks: defaultdict[UUID, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._capacity = asyncio.Lock()  # 数数 + 建 / 杀要原子，否则两个用户同时建会超上限
        self._busy: Counter[str] = Counter()  # 正在执行的沙箱，腾位置和 reap 时跳过

    @asynccontextmanager
    async def use(self, user_id: UUID | None, conversation_id: UUID | None):
        """执行一次沙箱操作。同一会话串行：同一个内核同时跑两段代码会互相踩变量。
        工具把预期内的错误（代码报错、文件不存在）转成返回值；从这里漏出去的异常一律当「可能跑了一半」，重启内核。"""
        if user_id is None or conversation_id is None:
            raise RuntimeError("沙箱需要服务端注入的 user_id 和 conversation_id")
        async with self._conv_locks[conversation_id]:
            lease = await self._acquire(user_id, conversation_id)
            self._busy[lease.sandbox_id] += 1
            try:
                await self._mark(conversation_id, True)  # 先写库：执行前就标脏，崩在执行中也留得下来
                yield lease
            except BaseException:  # 取消 / 超时 / 断连
                await asyncio.shield(self._reset(conversation_id, lease))
                raise
            else:
                await self._mark(conversation_id, False)
            finally:
                self._busy[lease.sandbox_id] -= 1
                await db.pool().execute("UPDATE sandboxes SET last_used_at = now() WHERE sandbox_id = $1",
                                        lease.sandbox_id)

    async def _acquire(self, user_id: UUID, conversation_id: UUID) -> Lease:
        notes: list[str] = []
        async with self._user_locks[user_id]:
            sid = await db.pool().fetchval("SELECT sandbox_id FROM sandboxes WHERE user_id = $1", user_id)
            if sid and not await self.backend.alive(sid):  # Cube 那边没了（TTL 到期、虚拟机重启）
                await db.pool().execute("DELETE FROM sandboxes WHERE sandbox_id = $1", sid)
                sid = None
            if sid is None:
                sid = await self._create(user_id)
            row = await db.pool().fetchrow(
                "SELECT sandbox_id, context_id, needs_reset FROM sandbox_contexts WHERE conversation_id = $1",
                conversation_id)
            if row is None or row["sandbox_id"] != sid:
                if row is not None:  # 这个会话用过沙箱，但那个沙箱已经不在了
                    notes.append(REBUILT)
                ctx_id = await self.backend.new_context(sid)
                await db.pool().execute(
                    "INSERT INTO sandbox_contexts (conversation_id, sandbox_id, context_id) VALUES ($1, $2, $3) "
                    "ON CONFLICT (conversation_id) DO UPDATE "
                    "SET sandbox_id = EXCLUDED.sandbox_id, context_id = EXCLUDED.context_id, needs_reset = false",
                    conversation_id, sid, ctx_id)
            else:
                ctx_id = row["context_id"]
                if row["needs_reset"]:  # 上次取消时没重启成功，或者进程崩在执行中
                    if not await self.backend.restart_context(sid, ctx_id):  # 实测：重启不存在的上下文报 404，新开一个
                        ctx_id = await self.backend.new_context(sid)
                        await db.pool().execute("UPDATE sandbox_contexts SET context_id = $2 WHERE conversation_id = $1",
                                                conversation_id, ctx_id)
                    await self._mark(conversation_id, False)
                    notes.append(RESTARTED)
            await self.backend.keepalive(sid, TTL_SECONDS)  # 续期：Cube 的兜底回收从现在重新计时
        return Lease(sid, ctx_id, notes)

    async def _create(self, user_id: UUID) -> str:
        async with self._capacity:
            if await db.pool().fetchval("SELECT count(*) FROM sandboxes") >= SANDBOX_LIMIT:
                busy = [s for s, n in self._busy.items() if n > 0]
                victim = await db.pool().fetchval(
                    "SELECT sandbox_id FROM sandboxes WHERE NOT (sandbox_id = ANY($1)) ORDER BY last_used_at LIMIT 1",
                    busy)
                if victim is None:
                    raise SandboxBusy(f"沙箱数已达上限 {SANDBOX_LIMIT}，且都在使用")
                await self._destroy(victim)
            sid = await self.backend.create(str(user_id))
            try:
                await db.pool().execute("INSERT INTO sandboxes (user_id, sandbox_id) VALUES ($1, $2)", user_id, sid)
            except BaseException:  # 落库失败或被取消：马上杀掉，别留孤儿（真崩在这里，启动对账按 metadata 找得到）
                await asyncio.shield(self._kill_quietly(sid))
                raise
            return sid

    async def reap(self) -> int:
        """杀掉空闲超过 IDLE_SECONDS 的沙箱。main 里每分钟调一次。"""
        async with self._capacity:
            rows = await db.pool().fetch(
                "SELECT sandbox_id FROM sandboxes WHERE last_used_at < now() - make_interval(secs => $1)",
                IDLE_SECONDS)
            victims = [r["sandbox_id"] for r in rows if not self._busy[r["sandbox_id"]]]
            for sid in victims:
                await self._destroy(sid)
        return len(victims)

    async def reconcile(self) -> tuple[int, int]:
        """启动对账，返回 (杀掉的孤儿数, 删掉的失效行数)。"""
        ours = await self.backend.list_ours()
        rows = {r["sandbox_id"]: str(r["user_id"])
                for r in await db.pool().fetch("SELECT sandbox_id, user_id FROM sandboxes")}
        orphans = [s for s, uid in ours.items() if rows.get(s) != uid]  # 库里没有，或者用户对不上
        for sid in orphans:
            await self._kill_quietly(sid)
        gone = [s for s in rows if s not in ours or s in orphans]
        await db.pool().execute("DELETE FROM sandboxes WHERE sandbox_id = ANY($1)", gone)
        return len(orphans), len(gone)

    async def _destroy(self, sid: str) -> None:
        # 先删行再杀：崩在中间只会留下孤儿，下次启动对账会杀掉；反过来会留下指向死沙箱的行（alive 也兜得住）
        await db.pool().execute("DELETE FROM sandboxes WHERE sandbox_id = $1", sid)
        await self._kill_quietly(sid)

    async def _kill_quietly(self, sid: str) -> None:
        try:
            await self.backend.kill(sid)
        except Exception as e:  # 杀不掉也不影响主流程：Cube 的 TTL 会兜底
            log.warning("kill sandbox %s failed: %s", sid, e)

    async def _reset(self, conversation_id: UUID, lease: Lease) -> None:
        try:
            async with asyncio.timeout(RESET_TIMEOUT):
                ok = await self.backend.restart_context(lease.sandbox_id, lease.context_id)
            if ok:  # 上下文不在了就留着标记，下次 _acquire 新开一个
                await self._mark(conversation_id, False)
        except Exception as e:  # 重启不了：标记留着，下次用之前再重启
            log.warning("restart context %s failed: %s", lease.context_id, e)

    async def _mark(self, conversation_id: UUID, dirty: bool) -> None:
        await db.pool().execute("UPDATE sandbox_contexts SET needs_reset = $2 WHERE conversation_id = $1",
                                conversation_id, dirty)


pool: Pool | None = None  # main 启动时装上真实后端（第 2 段）
