"""M8 第 3 步：一次 trial 跑完后，从库里收集事实（Trial）和指标。只读库，不信模型的自述。

指标口径（用户在对比表里看的）：
- 延迟 = 本题各个父 run 的 (finished_at - created_at) 之和：审批题里人决定的那段等待不算
- token = 父会话助手消息的 usage + 子 run 的 usage；input 不含缓存命中（DeepSeek，M3 实测），单列 cache_read
- 工具调用次数 = 父和子的全部 tool_use（call_tool 拆包后算一次）
- 费用按 DeepSeek 定价页的高峰价算（api-docs.deepseek.com/quick_start/pricing，2026-10-30 读），宁高勿低
- O3：cache_creation 按未命中价计。DeepSeek 只有命中 / 未命中两档、没有写入价；本地库 52 条真实回复里
  这一项全是 0（2026-11-03 查），现在不影响数字，换服务商时也不会少算
- O3：命中率 = cache_read / (input + cache_read + cache_creation)，按 token 加权（budget.hit_rate）；
  每个父 run 单独一行，它派出的子 run 的 usage 算在它头上
"""

import json
from uuid import UUID

from app import context
from app.budget import hit_rate
from app.eval_grade import Trial

# 美元 / 百万 token：(缓存命中输入, 未命中输入, 输出)，高峰价；非高峰是一半
PRICES = {"deepseek-flash": (0.006, 0.3, 1.2), "deepseek-v4-pro": (0.044, 1.32, 3.96)}


def _text(content) -> str:
    return "".join(b.get("text", "") for b in content if b.get("type") == "text") if isinstance(content, list) else str(content)


def extract_calls(messages: list[dict]) -> list[dict]:
    """助手消息里的 tool_use 配上后面 user 消息里的 tool_result；call_tool 拆成真名和真参数。"""
    calls, by_id = [], {}
    for m in messages:
        for b in m["content"] if isinstance(m["content"], list) else []:
            if b.get("type") == "tool_use":
                name, args = b["name"], b.get("input") or {}
                if name == "call_tool" and isinstance(args, dict):
                    name, args = args.get("tool_name", name), args.get("arguments", {})
                c = {"name": name, "input": args, "is_error": None, "result": None, "tool_use_id": b["id"]}
                calls.append(c)
                by_id[b["id"]] = c
            elif b.get("type") == "tool_result" and b.get("tool_use_id") in by_id:
                c = by_id[b["tool_use_id"]]
                c["is_error"] = bool(b.get("is_error")) or str(b.get("content", "")).startswith("[TOOL_ERROR]")
                c["result"] = _text(b.get("content")) if isinstance(b.get("content"), list) else b.get("content")
    for c in calls:  # 没配上结果的（挂起没执行、被中断）不算成功
        if c["is_error"] is None:
            c["is_error"] = True
    return calls


def add_usage(total: dict, u: dict | None) -> None:
    for k, v in (u or {}).items():
        total[k] = total.get(k, 0) + (v or 0)


def cost_usd(usage: dict, model: str) -> float:
    hit, miss, out = PRICES[model]
    return (usage.get("cache_read_input_tokens", 0) * hit
            + (usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)) * miss
            + usage.get("output_tokens", 0) * out) / 1e6


def _run_metrics(run, usages: list[dict], model: str) -> dict:
    total: dict = {}
    for u in usages:
        add_usage(total, u)
    return {"run_id": str(run["id"]), "stop_reason": run["stop_reason"],
            "cost_usd": round(cost_usd(total, model), 6), "hit_rate": hit_rate(usages)}


async def collect(pg, conversation_id: UUID, user_id: UUID, model: str) -> tuple[Trial, dict]:
    runs = await pg.fetch("SELECT id, stop_reason, created_at, finished_at FROM runs "
                          "WHERE conversation_id = $1 ORDER BY created_at", conversation_id)
    msgs = [dict(m) for m in await pg.fetch("SELECT role, content, usage, run_id FROM messages "
                                            "WHERE conversation_id = $1 ORDER BY id", conversation_id)]
    kids = await pg.fetch("SELECT id, parent_run_id, conversation_id, status, created_at, finished_at, usage FROM runs "
                          "WHERE parent_run_id = ANY($1::uuid[]) ORDER BY child_index", [r["id"] for r in runs])
    calls = extract_calls(msgs)
    usage: dict = {}
    per_run: dict = {r["id"]: [] for r in runs}  # O3：每个父 run 的 usage（子 run 算在派它的父 run 头上）
    for m in msgs:
        if m["role"] == "assistant" and m["usage"]:
            add_usage(usage, m["usage"])
            per_run.setdefault(m["run_id"], []).append(m["usage"])
    for k in kids:  # 子 run 的工具调用也算进来（子里调的 kb_search 也是忠实度评委的原文）
        calls += extract_calls([dict(m) for m in await pg.fetch(
            "SELECT role, content FROM messages WHERE conversation_id = $1 ORDER BY id", k["conversation_id"])])
        add_usage(usage, k["usage"])
        per_run.setdefault(k["parent_run_id"], []).append(k["usage"] or {})
    t0 = runs[0]["created_at"] if runs else None
    trial = Trial(
        answer=context.final_answer(msgs),  # O7：被截断续写过的回答要拼回来
        calls=calls,
        stop_reasons=[r["stop_reason"] for r in runs],
        ledger=[dict(r) for r in await pg.fetch("SELECT to_name, amount FROM ledger WHERE user_id = $1 ORDER BY id", user_id)],
        approvals=[{"tool_name": a["tool_name"], "args": json.loads(a["args_json"]), "status": a["status"]}
                   for a in await pg.fetch("SELECT tool_name, args_json, status FROM tool_approval "
                                           "WHERE conversation_id = $1 ORDER BY id", conversation_id)],
        children=[{"status": k["status"], "start": (k["created_at"] - t0).total_seconds(),
                   "end": ((k["finished_at"] or k["created_at"]) - t0).total_seconds()} for k in kids],
        continuations=sum(context.note_kind(m) == "continue" or context.is_truncation_reject(m) for m in msgs),
        compacted=bool(await pg.fetchval("SELECT compacted_at_id > 0 FROM conversations WHERE id = $1", conversation_id)),
    )
    metrics = {"latency_s": round(sum((r["finished_at"] - r["created_at"]).total_seconds()
                                      for r in runs if r["finished_at"]), 2),
               "usage": usage, "cost_usd": round(cost_usd(usage, model), 6), "tool_calls": len(calls),
               "run_ids": [str(r["id"]) for r in runs],  # 每个 run 一棵 trace，结果页点进去
               "runs": [_run_metrics(r, per_run.get(r["id"], []), model) for r in runs],
               # P1-C7：压缩后靠检索找回原文的次数（2608.16370：压缩时成功率不变、检索调用暴涨）和每次压缩的记录
               "recall_calls": sum(c["name"] == "recall" for c in calls),
               "read_output_calls": sum(c["name"] == "read_tool_output" for c in calls),
               "compactions": [r["data"] for r in await pg.fetch(
                   "SELECT data FROM run_events WHERE run_id = ANY($1::uuid[]) AND type = 'compacted' ORDER BY run_id, seq",
                   [r["id"] for r in runs])]}
    return trial, metrics
