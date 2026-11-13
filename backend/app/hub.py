"""RunHub：把一个 run 的事件推给本进程的订阅者（多个标签页）。

每个订阅者一个 Queue：asyncio.Queue 的 get() 会取走元素，多个消费者共用一个会互相抢事件。
M2：hub 不再分配 seq。事件先写 run_events 拿到 seq（见 runner 的 EventWriter）。
O2：「通知，再按游标拉」。hub 不再接收事件内容，只接收「某个 run 有新事件」的信号：
- 信号来源：本进程写完库直接 wake；别的实例写的经 PG NOTIFY（db._APPEND 在同一事务里发，提交才送出）。
- 每个有订阅者的 run 一个拉取任务：按游标查 run_events（主键范围扫描），每批只查一次，再分发给全部订阅者。
  推送顺序 = 库里的 seq 顺序（同一 run 的提交被 runs 行锁串行，提交顺序就是 seq 顺序）。
- 订阅者队列有上限：满了不丢事件，而是清空并放 RESET，SSE 结束、客户端带游标重连从库里补。
"""

import asyncio
import logging
import os
from uuid import UUID

import asyncpg

from app import db

log = logging.getLogger(__name__)
END = None         # run 结束（已推过 run_end），SSE 收到后关闭连接
RESET = "reset"    # O2：断开让客户端带游标重连（队列满、拉取出错）。不带 run_end，前端会自动重连
CHANNEL = "run_events"
QUEUE_MAX = int(os.getenv("STREAM_QUEUE_MAX", "256"))  # 约每秒 10 个事件：能容忍客户端落后 ~25 秒


class _Run:
    def __init__(self, run_id: UUID, upto: int) -> None:
        self.run_id = run_id
        self.upto = upto              # 已经从库里拉过的最大 seq
        self.subs: set[asyncio.Queue] = set()
        self.wake = asyncio.Event()
        self.wake.set()               # 一创建就拉一次：订阅之前提交的、游标之后的事件
        self.task: asyncio.Task | None = None


class RunHub:
    def __init__(self) -> None:
        self._runs: dict[UUID, _Run] = {}

    def subscribe(self, run_id: UUID, after: int = 0) -> asyncio.Queue:
        """先订阅、再回放：从这里起有新提交都会进队列。after 只在这个 run 第一个订阅者时当拉取起点。"""
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)
        r = self._runs.get(run_id)
        if r is None:
            r = self._runs[run_id] = _Run(run_id, after)
            r.task = asyncio.create_task(self._pump(r))
        r.subs.add(q)
        return q

    def unsubscribe(self, run_id: UUID, q: asyncio.Queue) -> None:
        r = self._runs.get(run_id)
        if r is None:
            return
        r.subs.discard(q)
        if not r.subs:  # 没人看了：停掉拉取任务
            del self._runs[run_id]
            r.task.cancel()

    def wake(self, run_id: UUID) -> None:
        """这个 run 有新事件了（本机写完 / 收到 NOTIFY）。本机没人订阅就什么都不做。"""
        if r := self._runs.get(run_id):
            r.wake.set()

    def _put(self, r: _Run, q: asyncio.Queue, item) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:  # 慢消费者：断开它，而不是丢事件（丢了游标照样前进，就再也补不回来）
            log.warning("subscriber of run %s fell %d events behind; disconnecting", r.run_id, q.maxsize)
            while not q.empty():
                q.get_nowait()
            q.put_nowait(RESET)
            r.subs.discard(q)

    async def _pump(self, r: _Run) -> None:
        try:
            while True:
                await r.wake.wait()
                r.wake.clear()  # 先清再查：查的期间来的信号会让下一轮再查一次，不会漏
                for ev in await db.list_events(r.run_id, r.upto):
                    r.upto = ev["seq"]
                    for q in list(r.subs):
                        self._put(r, q, ev)
                    if ev["type"] == "run_end":  # 不管是谁写的（属主、接管者、强制收尾），各实例的流都在这里关
                        for q in list(r.subs):
                            self._put(r, q, END)
                        return
                if not r.subs:  # 订阅者都因为太慢被断开了：不再空拉，重连时会新建
                    return
        except Exception:  # 库出错：断开订阅者，它们带游标重连
            log.exception("pump of run %s failed", r.run_id)
            for q in list(r.subs):
                self._put(r, q, RESET)
        finally:
            if self._runs.get(r.run_id) is r:
                del self._runs[r.run_id]

    async def listen(self, url: str) -> None:
        """O2：每个实例一条专用 LISTEN 连接，不从连接池借（还回池时 asyncpg 会 reset：清监听器、UNLISTEN *）。
        NOTIFY 不持久：断开期间的通知全丢。所以每次（重新）连上后唤醒所有拉取任务，按游标补拉。"""
        delay = 0.5
        while True:
            conn = None
            try:
                # application_name 带进程号：在 pg_stat_activity 里认出是哪个实例的 LISTEN 连接（运维、O2 实测杀连接用）
                conn = await asyncpg.connect(url, server_settings={"application_name": f"oasis-listen-{os.getpid()}"})
                lost = asyncio.Event()
                conn.add_termination_listener(lambda _: lost.set())
                await conn.add_listener(CHANNEL, lambda _c, _pid, _ch, payload: self.wake(UUID(payload)))
                for r in list(self._runs.values()):
                    r.wake.set()
                delay = 0.5
                await lost.wait()
                log.warning("LISTEN connection lost; reconnecting")
            except (OSError, asyncpg.PostgresError) as e:
                log.warning("LISTEN connect failed: %s: %s; retry in %ss", type(e).__name__, e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10)
            finally:
                if conn is not None and not conn.is_closed():
                    await conn.close()


hub = RunHub()
