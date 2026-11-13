"""M9 第 1 段：沙箱池的生命周期、并发首调、隔离、上限、回收、取消标记、对账。用假后端，不连 Cube。"""

import asyncio
import contextlib
from collections import defaultdict

import pytest

from app import db, sandbox, tools
from test_db import with_fresh_db


class FakeBackend:
    def __init__(self):
        self.boxes: dict[str, str] = {}  # sandbox_id → user_id（相当于 Cube 里的 metadata）
        self.contexts: defaultdict[str, list[str]] = defaultdict(list)
        self.created, self.restarts, self.killed, self.keepalives = [], [], [], []
        self.fail_restart = False
        self.files: dict[tuple[str, str], str | bytes] = {}
        self.on_run = None  # async (code, timeout) -> sandbox.Output
        self.last_timeout = None

    async def create(self, user_id):
        await asyncio.sleep(0.01)  # 让并发首调真的交错
        sid = f"sb{len(self.created) + 1}"
        self.created.append(sid)
        self.boxes[sid] = user_id
        return sid

    async def alive(self, sid):
        return sid in self.boxes

    async def kill(self, sid):
        self.killed.append(sid)
        self.boxes.pop(sid, None)

    async def list_ours(self):
        return dict(self.boxes)

    async def keepalive(self, sid, seconds):
        self.keepalives.append((sid, seconds))

    async def new_context(self, sid):
        ctx = f"{sid}-ctx{sum(map(len, self.contexts.values())) + len(self.restarts) + 1}"  # 不复用旧名字
        self.contexts[sid].append(ctx)
        return ctx

    async def restart_context(self, sid, ctx):
        if self.fail_restart:
            raise ConnectionError("sandbox unreachable")
        self.restarts.append(ctx)
        return ctx in self.contexts[sid]

    # ---- 执行：test_sandbox_tools 用，on_run(code, timeout) 决定返回什么 Output ----
    async def run_code(self, sid, ctx, code, timeout):
        self.last_timeout = timeout
        return await self.on_run(code, timeout)

    async def run_shell(self, sid, command, timeout):
        self.last_timeout = timeout
        return await self.on_run(command, timeout)

    async def write_file(self, sid, path, data):
        self.files[(sid, path)] = data

    async def read_file(self, sid, path):
        if (sid, path) not in self.files:
            raise FileNotFoundError(path)
        return self.files[(sid, path)]


async def _user(name):
    uid = await db.create_user(name, f"token-{name}")
    return uid, await db.create_conversation(uid), await db.create_conversation(uid)


async def _lease(p, uid, cid):
    async with p.use(uid, cid) as lease:
        return lease


async def _dirty(cid):
    return await db.pool().fetchval("SELECT needs_reset FROM sandbox_contexts WHERE conversation_id = $1", cid)


def test_one_sandbox_per_user_one_context_per_conversation():
    async def body():
        b = FakeBackend()
        p = sandbox.Pool(b)
        alice, a1, a2 = await _user("alice")
        bob, b1, _ = await _user("bob")
        # 同一用户两个会话同时第一次调用：只建一个沙箱（用户锁），各自一个上下文
        la1, la2 = await asyncio.gather(_lease(p, alice, a1), _lease(p, alice, a2))
        assert la1.sandbox_id == la2.sandbox_id and la1.context_id != la2.context_id
        assert b.created == ["sb1"]
        # 同一会话再来：复用同一个上下文（变量还在）
        assert (await _lease(p, alice, a1)).context_id == la1.context_id
        # 另一个用户：拿到自己的沙箱，碰不到 alice 的
        lb = await _lease(p, bob, b1)
        assert lb.sandbox_id != la1.sandbox_id and b.boxes[lb.sandbox_id] == str(bob)
        assert all(s == 35 * 60 for _, s in b.keepalives)  # 每次使用都续期
    with_fresh_db(body)


def test_cancel_marks_dirty_first_then_restarts_kernel():
    async def body():
        b = FakeBackend()
        p = sandbox.Pool(b)
        uid, cid, _ = await _user("alice")
        started, seen = asyncio.Event(), []

        async def work():
            async with p.use(uid, cid) as lease:
                seen.append((lease.context_id, await _dirty(cid)))  # 执行中：标记已经在库里
                started.set()
                await asyncio.sleep(10)

        t = asyncio.create_task(work())
        await started.wait()
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t
        ctx, dirty_during = seen[0]
        assert dirty_during is True
        assert b.restarts == [ctx] and await _dirty(cid) is False  # 重启成功才清标记
    with_fresh_db(body)


def test_failed_restart_keeps_mark_and_next_use_restarts():
    async def body():
        b = FakeBackend()
        p = sandbox.Pool(b)
        uid, cid, _ = await _user("alice")
        b.fail_restart = True
        with pytest.raises(TimeoutError):  # 比如 e2b 的执行超时从工具里漏出来
            async with p.use(uid, cid):
                raise TimeoutError
        assert await _dirty(cid) is True  # 重启不了：标记留着
        b.fail_restart = False
        lease = await _lease(p, uid, cid)  # 下次用之前先重启，并告诉模型变量没了
        assert b.restarts == [lease.context_id] and lease.notes == [sandbox.RESTARTED]
        assert await _dirty(cid) is False
        # 上下文已经不在了（实测重启它报 404）：新开一个，不能卡死在「一直重启失败」
        await db.pool().execute("UPDATE sandbox_contexts SET needs_reset = true")
        b.contexts[lease.sandbox_id].clear()
        again = await _lease(p, uid, cid)
        assert again.context_id != lease.context_id and again.notes == [sandbox.RESTARTED]
        assert await _dirty(cid) is False
    with_fresh_db(body)


def test_limit_evicts_idle_lru_and_refuses_when_all_busy(monkeypatch):
    monkeypatch.setattr(sandbox, "SANDBOX_LIMIT", 2)

    async def body():
        b = FakeBackend()
        p = sandbox.Pool(b)
        (u1, c1, _), (u2, c2, _), (u3, c3, _) = [await _user(n) for n in ("u1", "u2", "u3")]
        await _lease(p, u1, c1)
        await _lease(p, u2, c2)
        await _lease(p, u3, c3)  # 满了：回收最久没用的 u1
        assert b.killed == ["sb1"] and set(b.boxes) == {"sb2", "sb3"}
        # u2、u3 都在执行中时，u1 再来：不杀正在用的，直接报忙，也不新建
        hold, release = asyncio.Event(), asyncio.Event()

        async def busy(u, c):
            async with p.use(u, c):
                hold.set()
                await release.wait()
        tasks = [asyncio.create_task(busy(u2, c2)), asyncio.create_task(busy(u3, c3))]
        await asyncio.sleep(0.05)
        with pytest.raises(sandbox.SandboxBusy):
            await _lease(p, u1, c1)
        assert len(b.created) == 3
        release.set()
        await asyncio.gather(*tasks)
    with_fresh_db(body)


def test_reap_kills_idle_but_not_busy():
    async def body():
        b = FakeBackend()
        p = sandbox.Pool(b)
        (u1, c1, _), (u2, c2, _) = await _user("u1"), await _user("u2")
        await _lease(p, u1, c1)
        await db.pool().execute("UPDATE sandboxes SET last_used_at = now() - interval '1 hour'")
        p._busy["sb1"] += 1  # 正在执行的不杀
        assert await p.reap() == 0
        p._busy["sb1"] -= 1
        assert await p.reap() == 1 and b.killed == ["sb1"]
        lease = await _lease(p, u1, c1)  # 回来再用：新沙箱 + 告诉模型之前的都没了
        assert lease.sandbox_id == "sb2" and lease.notes == [sandbox.REBUILT]
        assert (await _lease(p, u2, c2)).notes == []  # 没用过沙箱的会话不会收到「重建」提示
    with_fresh_db(body)


def test_reconcile_kills_orphans_and_drops_dead_rows():
    async def body():
        b = FakeBackend()
        p = sandbox.Pool(b)
        (u1, c1, _), (u2, c2, _), (u3, c3, _) = await _user("u1"), await _user("u2"), await _user("u3")
        keep = (await _lease(p, u1, c1)).sandbox_id
        dead = (await _lease(p, u2, c2)).sandbox_id
        mixed = (await _lease(p, u3, c3)).sandbox_id
        b.boxes.pop(dead)  # 库里有、Cube 没了（虚拟机重启过）
        b.boxes["orphan"] = str(u1)  # 建好了但没来得及落库就崩了
        b.boxes["stranger"] = "someone-else"  # metadata 对不上任何行
        b.boxes[mixed] = str(u1)  # 库里记着是 u3 的，metadata 却是 u1：宁可杀掉，也不能让 u3 用到 u1 的沙箱
        assert await p.reconcile() == (3, 2)
        assert set(b.killed) == {"orphan", "stranger", mixed} and set(b.boxes) == {keep}
        assert (await _lease(p, u1, c1)).notes == []  # 两边都有：连回去复用（用户选 b）
        assert (await _lease(p, u2, c2)).notes == [sandbox.REBUILT]
        assert (await _lease(p, u3, c3)).notes == [sandbox.REBUILT]
    with_fresh_db(body)


def test_low_risk_tool_with_record_execution_is_recorded():
    async def body():
        uid, cid, _ = await _user("alice")
        run_id = await db.start_run(cid, [{"type": "text", "text": "算一下"}])

        async def handler(args):
            return "ok"
        tool = tools.Tool("fake_run_code", "测试用", {"type": "object"}, handler, tools.PUBLIC, tools.LOW,
                          record_execution=True)
        tools.register(tool)
        token = tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, run_id=run_id))
        try:
            await tools.run(tools.Call("t1", tool.name, {}, tool=tool))
        finally:
            tools.current.reset(token)
            tools.unregister(tool.name)
        rows = await db.pool().fetch("SELECT tool_name, status FROM tool_executions WHERE run_id = $1", run_id)
        assert [(r["tool_name"], r["status"]) for r in rows] == [("fake_run_code", "succeeded")]
    with_fresh_db(body)
