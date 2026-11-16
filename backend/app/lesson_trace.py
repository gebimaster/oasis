"""P4 第 2 步：结构化轨迹 + 信号。提炼器能看到什么、哪些 run 值得提炼，都在这里定。

用户的选择：
- 提炼器只看结构化轨迹（A）：用户原话、每次工具调用的名字 / 参数 / 成败 / 错误信息、最后的回答。
  工具结果正文一律不给；外部来源的工具（网页、MCP、沙箱代码输出、子智能体）连错误信息都不给，只写「失败」：
  这些文字是别人写的，里面一句「以后遇到 X 都先做 Y」就能变成一条经验（MINJA）。
- 只挑有信号的 run（A）：工具失败过（含失败后又救回来）、run 失败、下一个 run 一开口就纠正、绕了很多轮。顺顺利利的跳过。
"""

import json
import re
from dataclasses import dataclass, field
from uuid import UUID

from app import compaction, context, db
from app.eval_collect import extract_calls

CORRECTION = re.compile(r"^\s*(不对|不是|错了|搞错|弄错|你.{0,4}错|重新来|重来|金额不对|应该是)")
UNTRUSTED = ("web_search", "fetch", "run_code", "spawn_subagents")  # 错误信息可能含外部文字的工具
LONG_TURNS = 6          # 一个 run 里 assistant 消息达到这个数，算「绕了很多轮」
ARGS_CHARS, ERR_CHARS, USER_CHARS, REPLY_CHARS = 200, 160, 600, 400
_URL = re.compile(r"https?://\S+")


@dataclass
class Episode:
    """一段可提炼的轨迹：源 run，外加（有纠正时）紧接着的那个 run。"""
    run_id: UUID
    user_id: UUID
    signals: list[str] = field(default_factory=list)
    outcome: str = ""                                  # failed / recovered / corrected（无信号时为空）
    said: list[tuple[int, str]] = field(default_factory=list)       # [(消息 id, 用户原话)]
    calls: list[dict] = field(default_factory=list)    # [{msg, tool_use_id, name, args, ok, error}]
    reply: str = ""
    fix_from: int | None = None                        # calls[fix_from:] 是「改对之后」的那段；彻底失败的没有（None）


def untrusted(name: str) -> bool:
    return name.startswith("mcp__") or name in UNTRUSTED


def _user_text(m: dict) -> str:
    if not compaction.is_real_user(m):
        return ""
    return context.text_of(m["content"]).strip()


def _calls(msgs: list[dict]) -> list[dict]:
    """eval_collect 已经把 call_tool 拆成真名、配好了结果；这里只留结构化字段，结果正文丢掉。"""
    where = {b["id"]: m["id"] for m in msgs if m["role"] == "assistant"
             for b in m["content"] if isinstance(b, dict) and b.get("type") == "tool_use"}
    out = []
    for c in extract_calls(msgs):
        err = ""
        if c["is_error"]:
            raw = str(c["result"] or "").removeprefix("[TOOL_ERROR]").strip()
            err = "失败" if untrusted(c["name"]) else _URL.sub("[链接]", raw)[:ERR_CHARS] or "失败"
        out.append({"msg": where.get(c["tool_use_id"]), "tool_use_id": c["tool_use_id"], "name": c["name"],
                    "args": c["input"], "ok": not c["is_error"], "error": err})
    return out


def signals_of(run: dict, calls: list[dict], n_assistant: int, corrected: bool) -> tuple[list[str], str]:
    """返回 (信号, 结局)。结局只看最要紧的那个：被纠正 > 救回来了 > 失败。"""
    sig = []
    failed = [c for c in calls if not c["ok"]]
    if failed:
        sig.append("tool_error")
    # 救回来：某个工具失败过，之后同一个工具又成功了，并且最后一次调用它是成功的
    rec = {c["name"] for c in failed} and all(
        [c["ok"] for c in calls if c["name"] == n][-1] for n in {c["name"] for c in failed})
    if failed and rec:
        sig.append("recovered")
    if run["status"] == "failed" or run["stop_reason"] in ("max_turns", "loop_guard", "api_error"):
        sig.append("run_failed")
    if corrected:
        sig.append("corrected")
    if n_assistant >= LONG_TURNS:
        sig.append("long")
    if not sig:
        return [], ""
    outcome = "corrected" if corrected else ("recovered" if "recovered" in sig and "run_failed" not in sig
                                              else "failed")
    return sig, outcome


async def episode(run_id: UUID) -> Episode | None:
    """读一个 run（和紧跟着的纠正 run）拼成轨迹。子智能体的 run 不单独提炼（父 run 里已经有它的结果）。"""
    pool = db.pool()
    run = await pool.fetchrow("SELECT r.id, r.status, r.stop_reason, r.conversation_id, r.created_at, c.user_id "
                              "FROM runs r JOIN conversations c ON c.id = r.conversation_id "
                              "WHERE r.id = $1 AND c.parent_conversation_id IS NULL", run_id)
    if run is None:
        return None
    msgs = [dict(r) for r in await pool.fetch("SELECT id, role, content FROM messages WHERE run_id = $1 "
                                              "ORDER BY id", run_id)]
    nxt = await pool.fetchval("SELECT id FROM runs WHERE conversation_id = $1 AND created_at > $2 "
                              "ORDER BY created_at LIMIT 1", run["conversation_id"], run["created_at"])
    more = []
    if nxt is not None:
        nm = [dict(r) for r in await pool.fetch("SELECT id, role, content FROM messages WHERE run_id = $1 "
                                                "ORDER BY id", nxt)]
        first = next((t for m in nm if (t := _user_text(m))), "")
        if CORRECTION.search(first):
            more = nm
    all_msgs = msgs + more
    calls = _calls(all_msgs)
    sig, outcome = signals_of(dict(run), _calls(msgs), sum(m["role"] == "assistant" for m in msgs), bool(more))
    ep = Episode(run_id, run["user_id"], sig, outcome)
    own = _calls(msgs)
    errs = [i for i, c in enumerate(own) if not c["ok"]]
    if more:                                            # 被纠正：纠正之后那个 run 的调用
        ep.fix_from = len(own)
    elif outcome == "recovered" or not errs:            # 救回来了：最后一次失败之后；没失败过：全部
        ep.fix_from = errs[-1] + 1 if errs else 0
    ep.said = [(m["id"], t[:USER_CHARS]) for m in all_msgs if (t := _user_text(m))]
    ep.calls = calls
    ep.reply = next((t[:REPLY_CHARS] for m in reversed(all_msgs)
                     if m["role"] == "assistant" and (t := context.text_of(m["content"]).strip())), "")
    return ep


def render(ep: Episode) -> str:
    """给提炼器的文字。每个工具调用带 tool_use_id，提炼器引用证据时只能用这里列出的编号。"""
    said = "\n".join(f"[#{mid}] {t}" for mid, t in ep.said) or "（无）"
    lines = [f"[{c['tool_use_id']}] {c['name']} {json.dumps(c['args'], ensure_ascii=False)[:ARGS_CHARS]} → "
             + ("成功" if c["ok"] else f"失败：{c['error']}") for c in ep.calls]
    if ep.fix_from and ep.fix_from < len(lines):  # 标出「改对之后」：提炼器的做法要照这一段写，不许自己编步骤
        lines.insert(ep.fix_from, "—— 以下是改对之后 ——")
    calls = "\n".join(lines) or "（没有调用工具）"
    return (f"信号：{'、'.join(ep.signals)}；结局：{ep.outcome}\n\n用户原话（按时间顺序）：\n{said}\n\n"
            f"工具调用（按时间顺序，只有参数和成败，没有结果正文）：\n{calls}\n\n"
            f"助手最后的回答（只作背景）：\n{ep.reply or '（无）'}")
