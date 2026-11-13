"""M7：spawn_subagents 工具，只做接线。编排（并发、超时、汇总）在 subagent.py，跑一个子 run 在 runner.run_child。

流程：每个子任务先发 queued → 拿到并发名额后建子会话 + 子 run（落库），发 running（带 child_run_id）
→ 子任务一结束就发终态 → 全部结束后汇总成 tool_result。状态事件都写进父 run 的 run_events。
"""

import json

from app import context, db, subagent, tools

NAME = "spawn_subagents"


async def _status(parent: tools.ToolContext, tool_use_id: str, sub: dict, status: str, **extra) -> None:
    ev = {"type": "subagent_update", "tool_use_id": tool_use_id, "index": sub["index"], "title": sub["title"],
          "status": status, **extra}
    if parent.emit and parent.run_id:
        # 走父 run 的 EventWriter：几个子任务在不同 task 里同时写，它的锁保证「拿 seq → 推送」不交错，
        # 否则 27 先推、26 后到，前端按 seq <= lastSeq 去重会把 26 当重复丢掉
        await parent.emit(lambda: db.append_event(parent.run_id, ev))


async def _spawn(args: dict) -> str:
    from app import runner  # runner import 本模块来注册工具，这里延迟导入避免循环
    parent, tool_use_id = tools.context(), tools.call_id.get()
    subs = [{**t, "index": i} for i, t in enumerate(args["tasks"])]
    for s in subs:  # 真实模型实测：父按自己的习惯填了 ["call_tool", "tool_search"]，交集为空，子任务没工具只能编
        if s.get("tools") is not None and not tools.child_context(None, s["tools"]).only:
            raise ValueError(f"子任务「{s['title']}」的 tools 里没有可用的工具。子智能体直接拿到工具的完整定义，"
                             "不需要 tool_search / call_tool；请写真实工具名，或者不填 tools")
    for s in subs:
        await _status(parent, tool_use_id, s, "queued")

    async def run_child(sub: dict, cm) -> dict:
        conv, run_id = await db.start_child_run(parent.conversation_id, parent.run_id, tool_use_id,
                                                sub["title"], sub["task"], sub["index"])
        await _status(parent, tool_use_id, sub, "running", child_run_id=str(run_id))
        # 这时还在父上下文的拷贝里：child_context 从父能用的工具里缩小
        return await runner.run_child(conv, run_id, tools.child_context(conv, sub.get("tools"), run_id), cm)

    async def on_done(sub: dict, r: dict) -> None:
        await _status(parent, tool_use_id, sub, r["status"], started=r.get("started", False))

    results = await subagent.spawn(subs, run_child, on_done=on_done)
    return subagent.render(results) + _unknown_note(await db.spawn_children(db.pool(), parent.run_id, tool_use_id))


def _unknown_note(children: list[dict]) -> str:
    """子任务里结果未知的写操作单独列出：父模型看到这一段就不该重试，而是请用户核实。"""
    ops = [f"- 「{c['title']}」{u['tool']} {json.dumps(u['args'], ensure_ascii=False)}"
           for c in sorted(children, key=lambda c: c["child_index"]) for u in c["unknown"]]
    return ("\n\n⚠ 以下有副作用的调用结果未知（可能已经生效）：不要重试，请让用户先核实。\n" + "\n".join(ops)) if ops else ""


def _from_row(task: dict, row: dict | None) -> dict:
    """库里一个子 run → render 用的结果。没有行 = 还在排队就被停了。"""
    if row is None:
        return {"title": task["title"], "status": "not_started", "started": False}
    status, reason = row["status"], row["stop_reason"]
    if status == "failed" and reason in ("timeout", "interrupted"):
        status = reason
    elif status in ("running", "cancelling"):  # 理论上调用时子 run 都已收尾，兜底按中断算
        status = "interrupted"
    text = context.final_answer(row["msgs"] or [])  # O7 复核④：续写过的汇报拼回来
    # O6 复核⑥：带上 stop_reason，render 才能把 repeated_action / tool_unavailable 翻成人话
    return {"title": task["title"], "status": status, "started": True, "text": text, "stop_reason": reason,
            **({"error": reason} if status == "failed" else {})}


async def closing_blocks(conn, conversation_id, run_id) -> list[dict]:
    """M7 第 5 步：父 run 被停止或启动对账时，给历史里没收口的 spawn_subagents 补真 tool_result。
    按库里每个子 run 的状态汇总：完成的带结论，没完成的写清原因，下一轮父模型只需重做没完成的。"""
    msgs = await conn.fetch("SELECT run_id, role, content FROM messages WHERE conversation_id = $1 ORDER BY id",
                            conversation_id)
    answered = {b.get("tool_use_id") for m in msgs if m["role"] == "user" and isinstance(m["content"], list)
                for b in m["content"]}
    blocks = []
    for m in msgs:
        if m["run_id"] != run_id or m["role"] != "assistant":
            continue
        for u in m["content"]:
            if u.get("type") == "tool_use" and u["name"] == NAME and u["id"] not in answered:
                rows = {r["child_index"]: r for r in await db.spawn_children(conn, run_id, u["id"])}
                results = [_from_row(t, rows.get(i)) for i, t in enumerate(u["input"].get("tasks", []))]
                blocks.append({"type": "tool_result", "tool_use_id": u["id"], "is_error": False,
                               "content": subagent.render(results) + _unknown_note(list(rows.values()))})
    return blocks


tools.register(tools.Tool(
    name=NAME,
    description=("把互不依赖的几件事派给子智能体并行去做，全部结束后一次返回每个子任务的汇报。"
                 "子智能体看不到这段对话：task 里要写全背景、约束和交付格式。"
                 "子智能体直接拿到工具的完整定义（包括延迟工具），不需要 tool_search / call_tool，task 里也不用教它加载。"
                 "子智能体不能做需要用户确认的操作，也不能再派子智能体。适合分头查资料、对比；简单的事自己做。"),
    input_schema={
        "type": "object",
        "properties": {"tasks": {
            "type": "array", "minItems": 1, "maxItems": subagent.MAX_TASKS,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 30, "description": "几个字的任务名，给用户看"},
                    "task": {"type": "string", "minLength": 1, "maxLength": 4000, "description": "完整的任务说明"},
                    "tools": {"type": "array", "items": {"type": "string"},
                              "description": "可选：只给这几个工具（写真实工具名）；不填 = 你能用的全部（不能多于你自己的）"},
                },
                "required": ["title", "task"],
                "additionalProperties": False,
            }}},
        "required": ["tasks"],
        "additionalProperties": False,
    },
    handler=_spawn,
    permission=tools.PUBLIC,
    risk=tools.LOW,  # 子智能体里要确认的调用一律拒绝，deny 规则照常生效，所以派出去本身不需要人确认
    timeout=subagent.SUBAGENT_TIMEOUT + 30,  # 子任务的超时已包住排队 + 执行，这里只多留收尾写库的时间
))
