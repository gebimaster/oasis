"""P5 第 1 条：运行时尾部跟着消息落库、按段只写变了的。

核心不变量：同一个 run 里，上一次请求的 messages 是下一次请求的逐字前缀。
Claude 5 的 thinking 签名（指纹覆盖它之前的全部消息）、DeepSeek 请求边界的缓存单元都靠它。
"""

import asyncio
import copy
from datetime import datetime, timedelta, timezone

from app import context
from app.compaction import render
from app.context import build_request, is_tail, note_kind, text_of
from app.context import fingerprint as fp
from app.llm.registry import resolve
from app.llm.transform import replay
from app.web_tools import seen_from_history
from tests.test_agent_loop import CALL, THINK, USER, fake_model, net_error, run

NOW = datetime(2026, 11, 16, 10, 0, tzinfo=timezone(timedelta(hours=8)))


def build(history, **kw):
    return build_request(**(dict(user_name="小刘", summary=None, history=history, now=NOW, final_turn=False) | kw))


def tail_of(msgs):
    return msgs[-1]["content"][-1]


def strip(msgs):
    """API 真正收到的样子（_tail 被 replay 剥掉）。"""
    return replay(msgs, resolve("deepseek-flash"))


# ---------------- 拼装：按段比对 ----------------

def test_sections_written_once_then_only_when_changed():
    _, m1 = build([{"role": "user", "content": "你好"}], status="\n<status>S1</status>", memory="\nM", lessons="\nL")
    t1 = tail_of(m1)
    assert t1["_tail"] == {"status": fp("\n<status>S1</status>"), "memory": fp("\nM"), "lessons": fp("\nL")}  # 存指纹
    assert t1["text"].endswith("\n<status>S1</status>\nM\nL")
    # 下一轮：上一条已经带着尾部进了历史，三段都没变 → 只写时间
    hist = [m1[-1], {"role": "assistant", "content": [CALL]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "晴"}]}]
    _, m2 = build(hist, status="\n<status>S1</status>", memory="\nM", lessons="\nL")
    assert tail_of(m2)["_tail"] == {} and tail_of(m2)["text"].startswith("---\n当前时间")
    # 状态块变了（压缩过）→ 只重写状态块
    _, m3 = build(hist, status="\n<status>S2</status>", memory="\nM", lessons="\nL")
    assert tail_of(m3)["_tail"] == {"status": fp("\n<status>S2</status>")}
    assert "S2" in tail_of(m3)["text"] and "\nM" not in tail_of(m3)["text"]


def test_old_tail_summarized_away_is_rewritten():
    """压缩把带尾部的旧消息摘掉了：历史里看不见 → 重写（不能因为「以前写过」就省掉）。"""
    _, msgs = build([{"role": "user", "content": "继续"}], summary="摘要", memory="\nM")
    assert tail_of(msgs)["_tail"] == {"memory": fp("\nM")}


def test_retry_of_same_turn_adds_no_second_tail():
    """同一轮重试（JSON 坏了、流中途断了）：最后一条已经挂过尾部，原样重发，字节不变。"""
    _, m1 = build([{"role": "user", "content": "你好"}], memory="\nM")
    _, again = build([m1[-1]], memory="\nM", final_turn=True, notes=["\n提醒"])
    assert again[-1] == m1[-1]


def test_tail_marker_never_reaches_api():
    _, msgs = build([{"role": "user", "content": "你好"}])
    assert all("_tail" not in b for m in strip(msgs) for b in (m["content"] if isinstance(m["content"], list) else []))


# ---------------- 读「用户说的话」的地方都跳过尾部 ----------------

def test_readers_of_user_words_skip_tail():
    _, msgs = build([{"role": "user", "content": "你好"}], status="\n<status>web_fetch https://evil.example/x</status>")
    q = msgs[-1]
    assert text_of(q["content"]) == "你好"
    assert "evil" not in render([q])  # 摘要模型看不到尾部
    assert seen_from_history([q]) == set()  # 尾部里的链接不算用户贴的（否则 fetch 不用确认）


def test_loop_note_with_tail_is_still_a_loop_note():
    note = context.loop_note("empty", context.EMPTY_NOTE)
    _, msgs = build([{"role": "user", "content": "q"}, {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                     note])
    assert is_tail(msgs[-1]["content"][-1]) and note_kind(msgs[-1]) == "empty"


# ---------------- 循环：存回历史，前一次请求是后一次的前缀 ----------------

def test_each_request_is_prefix_of_next():
    stream, calls = fake_model([([THINK, CALL], "tool_use"), ([THINK, {**CALL, "id": "t2"}], "tool_use"),
                                ([{"type": "text", "text": "好了"}], "end_turn")])
    history = copy.deepcopy(USER)
    events = run(history, stream, memory_notes=["\n记忆提醒"], lesson_notes=["\n经验"])
    assert len(calls) == 3
    for a, b in zip(calls, calls[1:]):
        prev, nxt = strip(a["messages"]), strip(b["messages"])
        assert nxt[:len(prev)] == prev  # 上一次请求（含它的尾部）逐字出现在下一次里
    # 记忆、经验只在第一轮写了一次，后面两轮的尾部里没有重复
    tails = [c["messages"][-1]["content"][-1]["text"] for c in calls]
    assert "记忆提醒" in tails[0] and "经验" in tails[0]
    assert all("记忆提醒" not in t and "经验" not in t for t in tails[1:])
    # 每轮一个 tail 事件，挂的是同一个历史对象（runner 按 id 写库）
    saved_tails = [e["message"] for e in events if e["type"] == "tail"]
    assert len(saved_tails) == 3 and all(is_tail(m["content"][-1]) for m in saved_tails)


def test_retried_turn_resends_identical_request():
    """流中途断了重试：两次请求逐字相同（尾部不重算，时间不会变），只发一个 tail 事件。"""
    stream, calls = fake_model([net_error(), ([{"type": "text", "text": "好"}], "end_turn")])
    clock = iter([NOW + timedelta(minutes=i) for i in range(50)])
    events = run(copy.deepcopy(USER), stream, now=lambda: next(clock))
    assert len(calls) == 2 and calls[0]["messages"] == calls[1]["messages"]
    assert [e["type"] for e in events].count("tail") == 1



# ---------------- 落库以后：走库的三个读者（真 PG） ----------------

def test_db_readers_skip_stored_tail():
    """「记住」判断（last_user_words）、记忆抽取（run_input）、召回（search_messages）都不能读到尾部。
    尾部里有记忆回显：抽取读到会自我强化；「记住」读到会把模型自己的 memory_write 当成用户要求。"""
    from app import db, memory_extract, memory_tools, recall
    from test_db import with_fresh_db
    tail = {"type": "text", "text": "---\n当前时间\n请记住：用户喜欢 紫罗兰", "_tail": {"memory": "x"}}

    async def body():
        uid = await db.create_user("alice", "tok-a")
        old, cur = await db.create_conversation(uid), await db.create_conversation(uid)
        r = await db.start_run(old, [{"type": "text", "text": "北京天气"}])
        rows = await db.list_messages(old)
        await db.set_content(rows[-1]["id"], rows[-1]["content"] + [tail])  # 和 runner 一样：追加在末尾
        await db.insert_message(old, r, "assistant", [{"type": "tool_use", "id": "t1", "name": "w", "input": {}}])
        await db.insert_message(old, r, "user", [{"type": "tool_result", "tool_use_id": "t1", "content": "晴"}, tail])
        assert (await memory_tools.last_user_words(old))[1] == "北京天气"
        said, _ = await memory_extract.run_input(r)
        assert [t for _, t in said] == ["北京天气"]
        hits = await db.search_messages(uid, cur, ["紫罗兰"], limit=5, skip_tool=recall.NAME, skip_mark=recall.MARK)
        assert hits == []
    with_fresh_db(body)


# ---------------- 第 4 条：前缀换过的 thinking，Claude 上删掉 ----------------

THINK2 = {"type": "thinking", "thinking": "", "signature": "sig-2"}


def thinking_sigs(msgs, model):
    return [b["signature"] for m in replay(msgs, resolve(model)) if m["role"] == "assistant"
            for b in m["content"] if b["type"] == "thinking"]


def test_stale_thinking_dropped_for_claude_only():
    """floor=2：id 2 那条的 thinking 产生于压缩之前 → Claude 上删；id 4 的保留（只删最早的，不留空洞）。DeepSeek 都留。"""
    hist = [{"id": 1, "role": "user", "content": "q"}, {"id": 2, "role": "assistant", "content": [THINK, CALL]},
            {"id": 3, "role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "晴"}]},
            {"id": 4, "role": "assistant", "content": [THINK2, {"type": "text", "text": "晴"}]},
            {"id": 5, "role": "user", "content": "谢谢"}]
    _, msgs = build(context.mark_stale(hist, 2))
    assert thinking_sigs(msgs, "claude-opus-5-5") == ["sig-2"]
    assert thinking_sigs(msgs, "deepseek-flash") == ["sig-1", "sig-2"]
    assert "_stale" not in str(replay(msgs, resolve("deepseek-flash")))  # 标记不出站
    assert "_stale" not in str(hist)  # 原文不动


def test_compaction_in_run_raises_floor():
    """本 run 里压缩了：保留下来的那轮 thinking 之后在 Claude 上被删；压缩后新产生的照常带。"""
    from app import tools
    tools.current.set(tools.ToolContext(permissions=frozenset({"weather:read"})))
    stream, calls = fake_model([([THINK, CALL], "tool_use"), ([THINK2, {**CALL, "id": "t2"}], "tool_use"),
                                ([{"type": "text", "text": "好"}], "end_turn")])
    done = []

    async def compactor(history, summary, room, measure, **kw):  # 第 2 轮开头压一次：只留最后两条
        if len(history) < 3 or done:
            return None
        done.append(1)
        return "摘要", history[-2:]

    async def collect():
        n, out = 100, []
        async for ev in context_run(stream, compactor):
            if ev["type"] == "message":  # 像 runner 一样回填 id
                n += 1
                ev["message"]["id"] = n
            out.append(ev)
        return out
    events = asyncio.run(collect())
    assert any(e["type"] == "compacted" for e in events)
    assert thinking_sigs(calls[1]["messages"], "claude-opus-5-5") == []          # 保留的那轮（sig-1）删了
    assert thinking_sigs(calls[2]["messages"], "claude-opus-5-5") == ["sig-2"]   # 压缩后产生的留着
    assert thinking_sigs(calls[1]["messages"], "deepseek-flash") == ["sig-1"]    # DeepSeek 不动


def context_run(stream, compactor):
    from app.agent_loop import run_loop
    return run_loop(copy.deepcopy(USER), model_stream=stream, compactor=compactor,
                    limits={"trigger": 0, "target": 10**9})
