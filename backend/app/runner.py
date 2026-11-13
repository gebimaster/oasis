"""Runner：把 Agent 循环放到后台 task 里跑，负责「先写 run_events 拿 seq，再广播」。

SSE 断开不影响这里；只有用户点停止（cancel）或循环自己结束，run 才会结束。
"""

import asyncio
import logging
import os
from uuid import UUID

from app import (approvals, compaction, context, db, kb_tool, recall, subagent, subagent_tool,  # noqa: F401  import 即注册工具
                 tool_search, tools, trace, transfer, web_tools)
from app.agent_loop import run_loop
from app.hub import hub

log = logging.getLogger(__name__)
_tasks: dict[UUID, asyncio.Task] = {}  # 持有引用，防止 task 被垃圾回收；也用于取消
_user_stopped: set[UUID] = set()  # 用户点停止的 run（进程退出时的取消不算，不加「已停止」标记）


def start(conversation_id: UUID, run_id: UUID, **loop_kwargs) -> None:
    task = asyncio.create_task(_run(conversation_id, run_id, **loop_kwargs))
    _tasks[run_id] = task
    task.add_done_callback(lambda _: _tasks.pop(run_id, None))


async def shutdown() -> None:
    """进程退出前：取消所有 run 并等它们收尾（写 cancelled）。"""
    tasks = list(_tasks.values())
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def cancel(run_id: UUID) -> bool:
    """M7：先写库再取消。停止意图（父和子都改成 cancelling）先提交，再发取消信号：
    进程死在两步之间，重启时库里是 cancelling，对账按「用户停止」收尾；先 cancel 后写的话，崩在两步之间就永远停在 RUNNING。
    O1：task 不在本进程也照样写意图：属主的 stop_loop 1 秒内发现；属主死了，过期扫描按 cancelling 收尾。
    返回这次有没有写进意图（已结束的 run 返回 False）。"""
    changed = await db.request_cancel(run_id)
    _begin_stop(run_id)  # 本进程就有 task：不等轮询
    return bool(changed)


STOP_POLL = float(os.getenv("STOP_POLL_SECONDS", "1"))  # O1：属主多久看一次库里的停止意图
# O1：协作取消等多久才强制。必须大于正常收尾最长的一步：取消路径里沙箱重启内核最多等 sandbox.RESET_TIMEOUT=5 秒
FORCE_AFTER = float(os.getenv("FORCE_CANCEL_SECONDS", "10"))
_stopping: set[UUID] = set()           # 已发过取消、看门狗正在等的 run
_watchdogs: set[asyncio.Task] = set()  # 持有看门狗 task 的引用


def _begin_stop(run_id: UUID) -> None:
    task = _tasks.get(run_id)
    if task is None or run_id in _stopping:
        return
    _stopping.add(run_id)
    _user_stopped.add(run_id)
    task.cancel()
    wd = asyncio.create_task(_watchdog(run_id, task))
    _watchdogs.add(wd)
    wd.add_done_callback(_watchdogs.discard)


async def _watchdog(run_id: UUID, task: asyncio.Task) -> None:
    """先协作：等 task 自己收尾（存 stopped、补子任务结果、重启沙箱内核）。超时没结束 = 有代码吞了取消。"""
    try:
        await asyncio.wait([task], timeout=FORCE_AFTER)
        if task.done():
            return
        log.warning("run %s ignored cancel for %ss; forcing", run_id, FORCE_AFTER)
        await db.force_cancel(run_id, close_blocks=subagent_tool.closing_blocks)  # 库里收尾 + 属主换墓碑
        task.cancel()               # 再取消一次：打断卡在收尾里的 await
        _tasks.pop(run_id, None)    # 不再续租（属主已是墓碑，续也续不上）
        hub.wake(run_id)            # O2：run_end 已在库里，各实例的拉取任务拉到它就关流（这里只是本机不等 NOTIFY）
    except Exception:
        log.exception("force cancel of run %s failed", run_id)
    finally:
        _stopping.discard(run_id)


async def stop_loop() -> None:
    """O1：每个实例一个。用户可能在任意实例上点停止，意图只落在库里；属主轮询自己的 run。"""
    while True:
        await asyncio.sleep(STOP_POLL)
        try:
            for run_id in await db.cancelling_among([r for r in _tasks if r not in _stopping]):
                _begin_stop(run_id)
        except Exception as e:  # 库短暂不可用：下一轮再来
            log.warning("stop poll failed: %s: %s", type(e).__name__, e)


LEASE_TICK = float(os.getenv("LEASE_TICK_SECONDS", "10"))  # O1：续租 + 扫描的间隔，租约 30 秒的三分之一


async def lease_loop() -> None:
    """O1：每个实例一个。先给本进程的 run 续租，再接管别人过期的 run：A 崩了、B 不重启也能在一个租约 + 一个间隔内收尾。"""
    while True:
        await asyncio.sleep(LEASE_TICK)
        try:
            await db.renew_leases(list(_tasks))
            if n := await db.recover_expired_runs(close_blocks=subagent_tool.closing_blocks):
                log.warning("took over %d runs with expired leases", n)
            # O10 复核②：接管记成结果未知的调用不会续跑；按幂等键查得清的在这里收掉，批次不再等人就续跑
            for conversation_id, run_id in await approvals.reconcile_pending():
                start(conversation_id, run_id)
        except Exception as e:  # 这一轮失败（比如库短暂不可用）下一轮再来，循环不能死
            log.warning("lease tick failed: %s: %s", type(e).__name__, e)


FLUSH_INTERVAL = 0.1  # 文字增量最多攒 100ms 写一次库
FLUSH_CHARS = 200     # 或攒够 200 字立刻写，避免一批太大


class EventWriter:
    """一个 run 的唯一出口：先写 run_events 拿到 seq，再广播。

    主循环和后台定时器都会写库，用一把 asyncio.Lock 把「取缓冲 → 写库 → 推送」整段串起来，
    否则两边交错时，seq 的分配顺序和推送顺序可能不一致。
    O2：锁在被 shield 的任务里拿。原来锁在外层，调用方被取消时锁先释放，
    取消路径的写入插到在途写入前面：推送倒序、或 run_end 之后又写进一条（test_cancel_mid_commit_keeps_push_order）。
    """

    def __init__(self, run_id: UUID) -> None:
        self.run_id = run_id
        self._lock = asyncio.Lock()
        self._buf: list[str] = []
        self._chars = 0
        self.span: trace.Span | None = None  # M8：本 run 的根 span，收尾时跟着 run 的终态一起结束
        self._ticker = asyncio.create_task(self._tick())  # 模型停顿时也按时 flush

    async def text(self, s: str) -> None:
        self._buf.append(s)
        self._chars += len(s)
        if self._chars >= FLUSH_CHARS:
            await self._deliver(None)

    async def emit(self, write) -> None:
        """非增量事件：先把缓冲里的字写掉，再写这个事件，保证顺序。write() 写库并返回带 seq 的事件。"""
        await self._deliver(write)

    async def finish(self, status: str, stop_reason: str | None, error: str | None = None, write=None) -> None:
        """停定时器 → 写掉剩余文字 → run_end（与改状态同一事务）。在途写入不用单独等：都排在同一把锁上。
        O2：订阅不在这里关，各实例的拉取任务拉到 run_end 时关。
        write 可替换收尾的那次写库（M5 挂起：审批记录 + 卡片事件 + run_end 同一事务）。"""
        self._ticker.cancel()
        await asyncio.wait([self._ticker])  # wait 不会把 CancelledError 抛给我们
        await self._deliver(write or (lambda: db.finish_run(self.run_id, status, stop_reason, error)))
        if self.span:  # 所有收尾路径（正常 / 挂起 / 取消 / 崩溃）都经过这里，span 不会漏收
            await trace.end(self.span, trace.RUN_STATUS.get(status, "error"), {"stop_reason": stop_reason}, error)

    async def abandon(self) -> None:
        """O1：run 已被别的实例接管（它写过 run_end 了）。什么都不再写，只停定时器。
        O2：本机订阅者由拉取任务从库里拉到接管者写的 run_end 后关流；wake 一下，不等 NOTIFY。"""
        self._ticker.cancel()
        await asyncio.wait([self._ticker])
        hub.wake(self.run_id)

    async def _tick(self) -> None:
        try:
            while True:
                await asyncio.sleep(FLUSH_INTERVAL)
                if self._buf:
                    await self._deliver(None)
        except db.LeaseLost:  # O1：被接管了，主循环下一次写库也会撞上并悄悄退出，这里不算崩溃
            pass
        except Exception:  # 写库失败：定时器停掉，主循环下一次写库会再遇到错误并把 run 收尾
            log.exception("flush ticker of run %s crashed", self.run_id)

    async def _deliver(self, write) -> None:
        """持锁「flush 缓冲 → 写 write → 推送」，整段在一个被 shield 的任务里。
        调用方被取消只是不再等它：锁仍由这个任务持有到推送完，后来的写入（取消路径、finish）按先来后到排队。"""
        async def locked():
            async with self._lock:
                await self._flush()
                if write is not None:
                    await self._commit(write)
        await asyncio.shield(asyncio.ensure_future(locked()))

    async def _flush(self) -> None:  # 调用方须持锁
        if not self._buf:
            return
        text = "".join(self._buf)
        self._buf.clear()
        self._chars = 0
        await self._commit(lambda: db.append_event(self.run_id, {"type": "text_delta", "text": text}))

    async def _commit(self, write) -> None:  # 调用方须持锁；外层 _deliver 的 shield 保证取消时也写完并通知
        await write()
        hub.wake(self.run_id)  # O2：只发信号，内容由拉取任务按游标从库里读；别的实例靠同一事务里的 NOTIFY


async def _run(conversation_id: UUID, run_id: UUID, **loop_kwargs) -> None:
    w = EventWriter(run_id)
    try:
        w.span = await trace.start("run", "run", run_id)  # M8：顶层 run 是 trace 的根
        trace.current.set(w.span)
        # M3：只加载水位之后的原文，前面由摘要代替；压缩前产生的 usage 已过期，不拿来打底
        user = await db.conversation_user(conversation_id)
        # 只在这个 task 的上下文里生效；工具从这里拿身份、权限和确认规则，模型传不进来
        tools.current.set(tools.ToolContext(user_id=user["id"], conversation_id=conversation_id,
                                            loaded=await db.get_loaded_tools(conversation_id),
                                            permissions=frozenset(await db.get_permissions(user["id"])),
                                            rules=await db.get_rules(user["id"]),
                                            run_id=run_id, emit=w.emit))  # M7：子任务状态经本 run 的出口写
        # M5 续跑：有人决定过的批次先执行、写回 tool_result，再读历史进循环。普通 run 没有未收口批次，直接跳过
        if await approvals.resume(conversation_id, run_id, w):
            await w.finish("completed", "awaiting_approval")  # 执行超时 → 结果未知，再次等人核实
            return
        cp = await db.get_compaction(conversation_id)
        cl = await db.get_cleared(conversation_id)  # O9：清旧结果的水位
        # O4：摘要 → 被水位跳过但要原样保留的副本（单轮压缩时的当前提问）→ 水位之后的原文
        history = [{**m, "replayed": True} for m in cp["replay_prefix"]] + [
            {"id": m["id"], "role": m["role"], "content": m["content"], "status": m["status"],
             "usage": m["usage"] if m["id"] > max(cp["compacted_at_id"], cl["cleared_at_id"]) else None}
            for m in await db.list_messages(conversation_id, from_id=cp["keep_from_id"])]
        tools.context().seen_urls.update(web_tools.seen_from_history(history))  # M9：用户贴过的链接 fetch 不用确认
        last_assistant = None  # 挂起时要知道这批 tool_use 在哪条消息里
        compactor = compaction.make_compactor(conversation_id, cp["revision"])
        stream = trace.traced_llm(loop_kwargs.pop("model_stream", None), run_id)  # M8：每轮一个 llm span
        async for ev in run_loop(history, user_name=user["name"], summary=cp["summary"],
                                 compactor=compactor, model_stream=stream,
                                 cleared_until=cl["cleared_until_id"], **loop_kwargs):
            if ev["type"] == "text_delta":
                await w.text(ev["text"])
            elif ev["type"] == "message":
                msg = ev["message"]  # 存消息和 message_saved 事件在同一事务里

                async def save(msg=msg):
                    nonlocal last_assistant
                    saved = await db.insert_message(conversation_id, run_id, msg["role"], msg["content"],
                                                    usage=msg.get("usage"))
                    if msg["role"] == "assistant":
                        last_assistant = saved["message_id"]
                    msg["id"] = saved["message_id"]  # O4：回填给循环里的同一个对象，单轮压缩的水位要用它
                    return saved
                await w.emit(save)
            elif ev["type"] == "cleared":  # O9：抬水位和事件同一个事务
                await w.emit(lambda ev=ev: db.save_cleared(conversation_id, run_id, ev))
            elif ev["type"] == "run_end" and ev.get("approvals"):  # M5 挂起
                rows = approvals.to_rows(ev["approvals"])
                await w.finish(ev["status"], ev["stop_reason"],
                               write=lambda: db.suspend_run(conversation_id, run_id, last_assistant, rows))
            elif ev["type"] == "run_end":
                await w.finish(ev["status"], ev["stop_reason"], ev.get("error"))
            else:  # tool_start / tool_end / compacted / loop_guard（O6）/ continued、retract（O7）
                await w.emit(lambda: db.append_event(run_id, ev))
    except asyncio.CancelledError:
        try:
            # 有未收口的审批批次时不加标记：tool_use 后面必须先跟它的 tool_result，批次由下一条消息收口
            # M7：没收口的 spawn_subagents 先按库里子 run 的状态补真 tool_result（子 run 此时都已收尾）
            if blocks := await subagent_tool.closing_blocks(db.pool(), conversation_id, run_id):
                await w.emit(lambda: db.insert_message(conversation_id, run_id, "user", blocks))
            if run_id in _user_stopped and not await db.open_batch(conversation_id):
                await w.emit(lambda: db.save_stopped(conversation_id, run_id))  # emit 先 flush 缓冲，半截文字不丢
            await w.finish("cancelled", "user_cancel")  # 被取消后仍可 await 做收尾
        except db.LeaseLost:  # O1：被强制取消（属主已是墓碑）或被接管：库里已经收好了，不再写
            await w.abandon()
        finally:
            _user_stopped.discard(run_id)
        raise  # 必须重新抛出，否则 task 不算被取消
    except db.LeaseLost:  # O1：卡太久租约过期、被别的实例接管了。不写 stopped / run_end，悄悄退出
        log.warning("run %s lost its lease; exiting without further writes", run_id)
        await w.abandon()
    except Exception as e:  # 代码 bug 或数据库出错：也要把 run 收尾，不能一直 running
        log.exception("run %s crashed", run_id)
        await w.finish("failed", "internal_error", f"{type(e).__name__}: {e}")


SUBAGENT_MAX_TURNS = 6  # M7：子智能体的轮数上限，比父的 10 小
# M7：子智能体的上下文上限（token）。超了下一轮强制交汇报；配合轮数上限，总花费 ≤ 6 × 这个数
SUBAGENT_CONTEXT_TOKENS = int(os.getenv("SUBAGENT_CONTEXT_TOKENS", "20000"))


async def run_child(conversation_id: UUID, run_id: UUID, ctx: tools.ToolContext, cm: asyncio.Timeout,
                    **loop_kwargs) -> dict:
    """M7：在调用方的 task 里跑一个子 run。不另起 task：父被取消、子超时，取消都直接落到这里。
    子 run 有自己的 EventWriter 和 run_events，前端点开胶囊订阅 /runs/{run_id}/stream，续播照 M2。"""
    tools.current.set(ctx)  # gather 给每个子任务一个 task（上下文是拷贝），这里 set 不影响父和兄弟
    w = EventWriter(run_id)
    produced, usage = [], {}  # O7：交回父的汇报用 context.final_answer 从本 run 的消息里取（续写过的要拼回来）
    try:
        # M8：子 run 挂在派它的 spawn_subagents 工具 span 下（gather 拷来的上下文里 current 就是它）
        w.span = await trace.start("run", "subagent", run_id, parent=trace.current.get())
        trace.current.set(w.span)
        user = await db.conversation_user(conversation_id)
        history = [{"role": m["role"], "content": m["content"]} for m in await db.list_messages(conversation_id)]
        loop_kwargs["model_stream"] = trace.traced_llm(loop_kwargs.get("model_stream"), run_id)
        async for ev in run_loop(history, user_name=user["name"], persona=subagent.CONTRACT,
                                 **({"max_turns": SUBAGENT_MAX_TURNS,
                                     "context_limit": SUBAGENT_CONTEXT_TOKENS} | loop_kwargs)):
            if ev["type"] == "text_delta":
                await w.text(ev["text"])
            elif ev["type"] == "message":
                msg = ev["message"]
                produced.append(msg)
                if msg["role"] == "assistant":
                    for k, v in (msg.get("usage") or {}).items():
                        usage[k] = usage.get(k, 0) + v
                await w.emit(lambda msg=msg: db.insert_message(conversation_id, run_id, msg["role"], msg["content"],
                                                               usage=msg.get("usage")))
            elif ev["type"] == "run_end":
                await db.save_run_usage(run_id, usage)
                await w.finish(ev["status"], ev["stop_reason"], ev.get("error"))
                out = {"status": ev["status"], "text": context.final_answer(produced), "run_id": run_id, "stop_reason": ev["stop_reason"]}
                return out | ({"error": ev["stop_reason"]} if ev["status"] != "completed" else {})  # 只给原因代号，不给原文
            else:  # tool_start / tool_end / continued、retract（O7）
                await w.emit(lambda ev=ev: db.append_event(run_id, ev))
    except asyncio.CancelledError:  # 超时和父取消都走这里；cm.expired() 区分二者（3.13 实测）
        await db.save_run_usage(run_id, usage)
        await (w.finish("failed", "timeout") if cm.expired() else w.finish("cancelled", "parent_cancel"))
        raise
    except db.LeaseLost:  # O1：父子同属主，父下一次写库也会被拒，由父的 _run 退出
        await w.abandon()
        raise
    except Exception as e:
        log.exception("child run %s crashed", run_id)
        await w.finish("failed", "internal_error", type(e).__name__)
        raise
