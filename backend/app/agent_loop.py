"""Agent 循环：async generator，只负责「模型 → 工具 → 模型」并 yield 事件。

不碰数据库、不碰 SSE；外层（runner）拿到事件后决定存库和推送。
对外事件：
- text_delta {text}                          逐字输出
- message    {message: {role, content}}      一条完整消息，外层负责 INSERT
- tool_start {tool_use_id, name, input}
- tool_end   {tool_use_id, is_error, content}
- run_end    {status, stop_reason, error?}   最后一个事件
  M5：有调用要人确认时，以 stop_reason=awaiting_approval 结束，并带 approvals（整批调用的记录，外层落库）
取消（用户点停止）由外层 cancel task 处理，这里不管。

O3 出口表（tests/test_agent_loop.py::EXITS 每行一个用例）：
  completed：end_turn / max_turns / budget / awaiting_approval
  failed：api_error / invalid_tool_json / refusal / max_tokens / final_turn_violation / pause_turn_limit
         / context_overflow / rate_limited / server_error（O4，llm.classify_error 分类）
         / repeated_action（O6：同一调用连续到硬阈值，走收尾轮强制总结后判 failed，见 loop_guard）
         / network（O7：断网）/ empty_response（O7：提醒过一次仍是空回复）
  completed 另有：tool_unavailable（O6：同一工具累计依赖不可用到阈值，走收尾轮强制总结）
  O7：max_tokens 不再直接 failed，先续写（纯文本、只有 thinking、截断的工具调用三种，共用 MAX_CONTINUATIONS），
      额度用完才 failed/max_tokens；拿到响应头以后的断网 / 流内过载重试 MAX_STREAM_RETRIES 次。
配对不变量：任何出口之后历史里都没有孤儿 tool_use。不执行的调用当场补「未执行」结果，
唯一例外是 awaiting_approval（整批结果在审批记录里，续跑时写回）。
O7 已输出水位：一次模型调用推出过 text_delta、却没产出 final（要重试或失败），先发 retract 让前端丢掉这段，
所以「推给用户的字」要么落进了某条 assistant 消息，要么被撤回，不会重复、也不会留着库里没有的半截。
"""

import asyncio

from datetime import datetime
from typing import AsyncIterator, Callable

from app import budget, compaction, context, llm, loop_guard, microcompact, rules, tools

MAX_TURNS = 10
MAX_JSON_RETRIES = 2  # 入参 JSON 解析失败时重发同一轮的次数，不占轮数
MAX_PAUSES = 3  # O3：连续 pause_turn 的上限。续写不占轮数，所以要单独封顶，否则可能永远出不去
MAX_COMPACTIONS = 2  # O4：每个 run 最多压几次（每次都多一次 LLM 调用）。拍的数，压完仍超线的交给被动恢复
# O7：一个 run 里因 max_tokens 续写的总次数，三种截断共用（每次都是一次付费调用，分开计额度等于放大上限）；不占轮数
MAX_CONTINUATIONS = 3
MAX_EMPTY_RETRIES = 1  # O7：空回复提醒一次，还空就 failed/empty_response
# O7：流开始以后的失败每轮最多重试几次。SDK 自带的 2 次只管拿到响应头之前，这里只接它不管的那段，两者不叠加
MAX_STREAM_RETRIES = 2
RETRY_DELAYS = (1.0, 2.0)  # 退避秒数，测试改成 0
RETRYABLE = {"network", "server_error", "rate_limited"}
NOT_RUN_FINAL = "已达工具调用上限，不能再调用工具。请基于已有信息直接给出最终回答"
NOT_RUN_STOP = {"refusal": "这条回复被拒绝"}
RETRACT = {"type": "retract"}  # O7：撤回本次调用已推出的文字，走 runner 的同一个出口（O2 的 _append + seq）


def repair_orphan_tool_uses(messages: list[dict]) -> list[dict]:
    """方案 B：读历史时，给没有配对 tool_result 的 tool_use 补一条错误结果。"""
    fixed = [dict(m) for m in messages]  # 浅拷贝，不改调用方的数据
    for i, m in enumerate(fixed):
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            continue
        ids = [b["id"] for b in m["content"] if b.get("type") == "tool_use"]
        nxt = fixed[i + 1] if i + 1 < len(fixed) else None
        nxt_blocks = nxt["content"] if nxt and nxt["role"] == "user" and isinstance(nxt["content"], list) else []
        answered = {b.get("tool_use_id") for b in nxt_blocks}
        fill = [{"type": "tool_result", "tool_use_id": x, "is_error": True,
                 "content": f"{tools.ERROR_PREFIX} run interrupted before the tool finished"}
                for x in ids if x not in answered]
        if not fill:
            continue
        if any(b.get("type") == "tool_result" for b in nxt_blocks):  # 下一条已有部分结果：补进同一条，放最前面
            nxt["content"] = fill + nxt_blocks
        else:  # O4：下一条是用户提问时也单独插一条，不拼进去（拼了压缩就认不出当前提问）
            fixed.insert(i + 1, {"role": "user", "content": fill})
    return fixed


def _approval_item(position: int, c: "tools.Call", result: dict | None) -> dict:
    """批次里的一个调用 → 一行审批记录。要确认的是 pending；排在它后面、本来放行的是 deferred（O10 复核③，
    批准后续跑按顺序执行）；其余已有结果（执行过 / 被拒 / 参数错）。"""
    if result is None:
        ask = c.decision == rules.ASK
        return {"position": position, "tool_use_id": c.id, "tool_name": c.name, "args": c.args,
                "decision": rules.ASK if ask else c.decision, "status": "pending" if ask else "deferred", "result": None}
    status = "denied" if c.decision == rules.DENY else ("failed" if result["is_error"] else "succeeded")
    return {"position": position, "tool_use_id": c.id, "tool_name": c.name, "args": c.args,
            "decision": c.decision, "status": status,
            "result": {"content": result["content"], "is_error": result["is_error"]}}


def _not_run(tool_uses: list[dict], why: str) -> dict:
    """O3：不执行的 tool_use 当场配一条错误结果，历史里不留孤儿。"""
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": u["id"], "is_error": True,
                                         "content": f"{tools.ERROR_PREFIX} 未执行：{why}"} for u in tool_uses]}


def context_tokens(usage: dict | None) -> int:
    """这一轮之后的上下文长度。DeepSeek 的 input_tokens 不含缓存命中（M3 实测），要把 cache 两项加回来；
    output 下一轮也会成为上下文。"""
    u = usage or {}
    return sum(u.get(k, 0) for k in ("input_tokens", "cache_read_input_tokens",
                                     "cache_creation_input_tokens", "output_tokens"))


async def run_loop(messages: list[dict], *, model_stream: Callable | None = None,
                   max_turns: int = MAX_TURNS, user_name: str = "用户",
                   now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
                   summary: str | None = None, compactor: Callable | None = None,
                   limits: dict | None = None, persona: str = context.PERSONA,
                   context_limit: int | None = None, cleared_until: int = 0) -> AsyncIterator[dict]:
    """messages 是水位之后的原文（可带 status/usage/id）；summary 是已有摘要。
    compactor 由 runner 注入（循环不碰数据库）；model_stream、now、limits 可注入做测试；persona 子智能体换成合同。
    context_limit（M7 子智能体预算）：某一轮的上下文超过它，下一轮就当最后一轮（不许调工具、必须交汇报）。
    cleared_until（O9）：清旧工具结果的水位，由 runner 从库里读出来；抬水位时发 cleared 事件，runner 存库。"""
    model_stream = model_stream or llm.stream_message  # 调用时再取：子 run 由工具发起，测试只能替换模块属性
    history = repair_orphan_tool_uses(messages)
    limits = limits or budget.current_budget()
    tool_defs = tools.list_tools()  # 常驻 + call_tool：整个 run 不变，加载新工具不改它
    turn, json_retries, pauses, compactions = 0, 0, 0, 0
    over_budget = False  # M7：上下文超了预算，下一轮强制收尾（复用 max_turns 的最后一轮）
    retried_final = False  # O3：收尾轮违约后已经多给过一次纯文本机会
    recovered: set[int] = set()  # O4：超窗被动恢复过的轮次
    windowed = False  # O4：截过窗口后 history 不完整，本 run 不再压缩（再压会让水位越过没摘要过的原文）
    continuations, empties, stream_retries = 0, 0, 0  # O7：续写（本 run 累计）、空回复提醒、流中途重试（本轮）
    repeats, outages = loop_guard.RepeatGuard(), loop_guard.UnavailableGuard()  # O6：本 run 的两种熔断计数
    breaker: tuple[str, str, int] | None = None  # O6：(stop_reason, 工具名, 次数)。非空 = 下一轮强制总结（复用收尾轮）

    while True:
        turn += 1
        # 最后一轮：仍带 tools 定义（历史里有 tool_use），但禁止调用
        last = turn >= max_turns or over_budget or breaker is not None
        end_reason = breaker[0] if breaker else ("budget" if over_budget else "max_turns")
        if breaker:
            why = context.UNAVAILABLE_FINAL_NOTE if breaker[0] == "tool_unavailable" else context.REPEAT_FINAL_NOTE
            notes = [why.format(name=breaker[1], n=breaker[2])]
        else:  # 复核①：最后一轮不能再叫它「换办法」「可以继续」，换成说明原因、要求直接回答的版本（你选的 B）
            notes = [(context.REPEAT_LAST_NOTE if last else context.REPEAT_POLLING_NOTE if polling
                      else context.REPEAT_NOTE).format(name=name, n=n) for name, n, polling in repeats.nudges()]
        final = None

        def shape(s, h, instruction=None):  # 每轮都重新拼：history 在变，时间在变；最后一轮的提示、重复提醒都进尾部，不动 system
            return context.build_request(user_name=user_name, summary=s, history=h,
                                         now=now(), final_turn=last, catalog=tools.deferred_catalog(),
                                         loaded_defs=tools.definitions(tools.context().loaded), persona=persona,
                                         notes=notes, instruction=instruction)

        def build(s, h):
            return shape(s, microcompact.project(h, cleared_until))  # O9：确定的投影

        def prompt(s, h, instruction):
            """P1-C1：摘要请求和正常请求同一套拼法、同一份 tools，只把尾部换成压缩指令。
            h 是 compact 交给 compactor 的那份（已投影过），不再投影：再投影一次字节会变，前缀就对不上了。"""
            sys_, msgs = shape(s, h, instruction)
            return {"system": sys_, "messages": msgs, "tools": tool_defs}

        async def compact(h, s, room, **kw):  # kw：C4 的 fallback 只在超窗恢复时给
            """O9 复核③：摘要那一级按投影量长度、做摘要（清掉的结果模型本来就看不到，不该再按原文切、再喂给摘要），
            返回的保留部分按 id 换回原文：循环里的 history 必须是原文，投影只在拼请求时做（占位符再投影一次字节会变）。"""
            if not cleared_until:
                return await compactor(h, s, room, measure, prompt=prompt, **kw)
            raw = {m["id"]: m for m in h if m.get("id") is not None}
            # C2：原文一起交过去，ref 段从原文生成（占位符里没有 ref）；project 一一对应，下标对得上
            res = await compactor(microcompact.project(h, cleared_until), s, room, measure, prompt=prompt, raw=h, **kw)
            if not res:
                return res
            return res[0], [{**raw[m["id"]], "usage": None} if m.get("id") in raw else m for m in res[1]]

        def compact_stats():  # C7：这次压缩走了哪条路、摘要请求自己的命中率（单独摘要器 / 降级没有 usage，记 None）
            st = getattr(compactor, "stats", None) or {}
            return {"mode": st.get("mode"), "summary_hit_rate": budget.hit_rate(st.get("usages") or [])}

        def measure(s, h):  # O4：整份请求的本地估算，采用规则和留额度都用它（名录、loaded_defs 都在里面）
            sys_, msgs = build(s, h)
            return budget.estimate_json([sys_, tool_defs, msgs])
        system, request_msgs = build(summary, history)
        # O9：超过触发线先走不调模型的一级：清旧工具结果（够一批才清），清完还超线才轮到下面的 LLM 摘要
        if not windowed and budget.request_tokens(history, system=system, tools=tool_defs,
                                                  messages=request_msgs) > limits["trigger"]:
            if w := microcompact.next_watermark(history, cleared_until):
                before = measure(summary, history)
                cleared_until = w
                history = compaction.strip_usage(history)  # 那些 usage 是清理前的大请求，打底会立刻又判超线
                yield {"type": "cleared", "until_id": w, "at_id": max(m.get("id") or 0 for m in history),
                       "tokens_before": before, "tokens_after": measure(summary, history)}
                system, request_msgs = build(summary, history)
        # 超过触发线就压到目标线；每个 run 最多压 MAX_COMPACTIONS 次，用完照发
        if (compactor and not windowed and compactions < MAX_COMPACTIONS and budget.request_tokens(
                history, system=system, tools=tool_defs, messages=request_msgs) > limits["trigger"]):
            compactions += 1
            # 原文的额度 = 目标线 - 摘要以外的固定部分（有摘要才出现的外壳和 loaded_defs 也扣掉）- 摘要预留
            room = limits["target"] - measure("-", []) - budget.SUMMARY_RESERVE
            before = measure(summary, history)
            result = await compact(history, summary, room)
            if result:
                summary, history = result
                yield {"type": "compacted", "reason": "trigger", "kept_messages": len(history),
                       "summary_chars": len(summary),
                       "tokens_before": before, "tokens_after": measure(summary, history), **compact_stats()}
                system, request_msgs = build(summary, history)
        emitted = False  # O7 水位：这次调用推出过文字没有
        try:
            async for ev in model_stream(
                system=system,
                messages=request_msgs,
                tools=tool_defs,
                tool_choice={"type": "none"} if last else {"type": "auto"},
            ):
                if ev["type"] == "text_delta":
                    emitted = True
                    yield ev
                elif ev["type"] == "final":
                    final = ev
        except ValueError:  # 工具入参 JSON 彻底坏了：没有完整的 tool_use 可回答，重发这一轮
            if emitted:  # 修前：先推的那段字重试时又推一遍，用户看到「我查我查一下」
                yield RETRACT
            json_retries += 1
            if json_retries > MAX_JSON_RETRIES:
                yield {"type": "run_end", "status": "failed", "stop_reason": "invalid_tool_json"}
                return
            turn -= 1
            continue
        except llm.LLMError as e:  # 发请求阶段 SDK 已经重试过了；流开始以后的失败（mid_stream）这里可以重试
            if emitted:  # 这次调用没有 final，推出去的字不会落库：先撤回，不管接下来重试还是失败
                yield RETRACT
            kind = llm.classify_error(e)
            if kind in RETRYABLE and e.mid_stream and stream_retries < MAX_STREAM_RETRIES:
                await asyncio.sleep(RETRY_DELAYS[min(stream_retries, len(RETRY_DELAYS) - 1)])
                stream_retries += 1
                turn -= 1
                continue
            if kind == "context_overflow" and turn not in recovered:
                recovered.add(turn)  # 每一轮只恢复一次
                before = measure(summary, history)
                # 先强制压缩（不占 MAX_COMPACTIONS），压不下来再临时截窗口（不摘要、不落库）
                room = limits["target"] - measure("-", []) - budget.SUMMARY_RESERVE
                use = compactor and not windowed
                result, reason = (await compact(history, summary, room, fallback=True) if use else None), "overflow"
                if result is None and (window := compaction.temporary_window(history)):
                    result, reason, windowed = (summary, window), "window", True
                if result:  # 请求变了才重发；没变的话服务端检查是确定性的，原样重发必然同样 400
                    summary, history = result
                    yield {"type": "compacted", "reason": reason, "kept_messages": len(history),
                           "summary_chars": len(summary or ""), "tokens_before": before,
                           "tokens_after": measure(summary, history),
                           **(compact_stats() if reason == "overflow" else {"mode": "window", "summary_hit_rate": None})}
                    turn -= 1
                    continue
            yield {"type": "run_end", "status": "failed", "stop_reason": kind, "error": str(e)}
            return
        json_retries, stream_retries = 0, 0

        # usage 跟着消息走：存库后，下个 run 的第一轮也能拿它打底估算（build_request 会剥掉它）
        assistant = {"role": "assistant", "content": final["content"], "usage": final.get("usage")}
        history.append(assistant)
        yield {"type": "message", "message": assistant}  # 无论成败都先存下来
        stop = final["stop_reason"]
        if context_limit and context_tokens(final.get("usage")) > context_limit:
            over_budget = True

        if stop == "pause_turn":  # 服务端工具暂停：原样传回让它接着做，算同一轮的续写
            pauses += 1
            if pauses > MAX_PAUSES:
                yield {"type": "run_end", "status": "failed", "stop_reason": "pause_turn_limit"}
                return
            turn -= 1
            continue
        pauses = 0

        # 执不执行看内容里有没有 tool_use，不看 stop_reason：
        # end_turn 却带 tool_use 照样执行；但被拒、被截断、收尾轮的一律不执行
        tool_uses = [b for b in final["content"] if b.get("type") == "tool_use"]
        if stop == "max_tokens":  # O7：截断续写。先补「未执行」（配对），额度用完就停；没用完就续写，不占轮数
            continuations += 1
            if tool_uses:  # 续写提示就是这条拒绝结果本身（让它重新调用、入参写短），不另加
                # 复核⑧：收尾轮（tool_choice none）不能叫它「重新调用」，换成收尾轮的说法
                not_run = _not_run(tool_uses, context.TRUNCATED_TOOL_FINAL_NOTE if last else context.TRUNCATED_TOOL_NOTE)
                history.append(not_run)
                yield {"type": "message", "message": not_run}
            if continuations > MAX_CONTINUATIONS:
                yield {"type": "run_end", "status": "failed", "stop_reason": "max_tokens",
                       "error": f"本 run 第 {continuations} 次被长度上限截断，续写额度（{MAX_CONTINUATIONS} 次）已用完"}
                return
            kind = "tool_use" if tool_uses else ("text" if context.text_of(final["content"]) else "thinking")
            if not tool_uses:
                note = context.loop_note("continue", context.CONTINUE_TEXT_NOTE if kind == "text"
                                         else context.CONTINUE_THINKING_NOTE)
                history.append(note)
                yield {"type": "message", "message": note}
            yield {"type": "continued", "kind": kind, "count": continuations}
            turn -= 1
            continue
        if stop == "end_turn" and not tool_uses and not context.text_of(final["content"]).strip():
            # 复核⑤：续写之后回空，说明前面那段已经写完了（回答在库里），按正常结束，不提醒、不判失败
            if context.note_kind(history[-2]) == "continue" and context.final_answer(history).strip():
                yield {"type": "run_end", "status": "completed", "stop_reason": end_reason if last else stop}
                return
            if empties < MAX_EMPTY_RETRIES:  # O7：空回复（常见于工具结果之后）提醒一次，不占轮数
                empties += 1
                note = context.loop_note("empty", context.EMPTY_NOTE)
                history.append(note)
                yield {"type": "message", "message": note}
                turn -= 1
                continue
            yield {"type": "run_end", "status": "failed", "stop_reason": "empty_response"}
            return
        if tool_uses and (stop in NOT_RUN_STOP or last):
            not_run = _not_run(tool_uses, NOT_RUN_STOP.get(stop, NOT_RUN_FINAL))
            history.append(not_run)
            yield {"type": "message", "message": not_run}
        if stop in NOT_RUN_STOP:
            yield {"type": "run_end", "status": "failed", "stop_reason": stop}
            return
        if not tool_uses:  # 正常结束；收尾轮报的是逼它收尾的原因
            # O6：原样重复判 failed（任务卡住了，你选的 G3）；依赖不可用判 completed（服务挂了，你选的 R3）
            if last and breaker and breaker[0] == "repeated_action":
                yield {"type": "run_end", "status": "failed", "stop_reason": end_reason,
                       "error": f"同一调用 {breaker[1]} 连续 {breaker[2]} 次，已强制总结"}
                return
            yield {"type": "run_end", "status": "completed", "stop_reason": end_reason if last else stop}
            return
        if last:  # 收尾轮违约：结果已补，再给一次纯文本机会（tools 不变、仍是 none，不占轮数）；再违约才 failed
            if retried_final:
                yield {"type": "run_end", "status": "failed", "stop_reason": "final_turn_violation"}
                return
            retried_final = True
            turn -= 1
            continue

        calls = [tools.prepare(u) for u in tool_uses]  # M5：检查 + 规则判定，一次算好
        # O10 复核③：只执行第一个「要确认」之前的调用（检查没过的照样当场给结果）。排在它后面、本来放行的先不执行：
        # 模型给的是「建目录（要确认）→ 往目录里写文件（放行）」时，写文件不能抢在建目录被批准之前跑
        first_ask = next((i for i, c in enumerate(calls) if c.decision == rules.ASK and c.error is None), len(calls))
        now_calls = [c for i, c in enumerate(calls) if c.error is not None or (i < first_ask)]
        for u, c in zip(tool_uses, calls):
            if c in now_calls:  # 要确认的不发 tool_start，它们的卡片由 approval_request 事件画
                yield {"type": "tool_start", "tool_use_id": u["id"], "name": u["name"], "input": u["input"]}
        results = await tools.execute_all(now_calls)  # 放行的立刻执行，读工具不陪着等
        for r in results:
            yield {"type": "tool_end", "tool_use_id": r["tool_use_id"],
                   "is_error": r["is_error"], "content": r["content"]}
        if len(now_calls) < len(calls):  # 有要确认的：整批交给外层落库，本 run 结束，不占会话锁
            done = {r["tool_use_id"]: r for r in results}
            yield {"type": "run_end", "status": "completed", "stop_reason": "awaiting_approval",
                   "approvals": [_approval_item(i, c, done.get(c.id)) for i, c in enumerate(calls)]}
            return
        tool_msg = {"role": "user", "content": results}  # 并行结果放同一条 user 消息
        history.append(tool_msg)
        yield {"type": "message", "message": tool_msg}
        # O6：整批执行完再计数（到硬阈值的那次也照常执行，你选的 G4-B）。指纹和工具名都用 call_tool 拆包后的
        repeats.observe(calls)
        outages.observe(calls, tools.UNAVAILABLE)
        hits = [("tool_unavailable", "unavailable", h) for h in [outages.tripped()] if h] + \
               [("repeated_action", "finalize", h) for h in [repeats.tripped()] if h]
        for _, action, (name, n) in hits:  # 尾部提示不落库，发事件留痕（runner 当普通事件存进 run_events）
            yield {"type": "loop_guard", "action": action, "tool": name, "count": n}
        if hits:  # 同一轮两种都到了：以不可用为准（你选的 R4-A），原因更具体
            breaker = (hits[0][0], *hits[0][2])
        else:
            for name, n, _ in repeats.nudges():
                yield {"type": "loop_guard", "action": "nudge", "tool": name, "count": n}
