"""P3-C5：压缩后的状态块。从摘要里原样摘目标 / 进行中 / 下一步 / 未回答的请求 + 最近 5 个 ref + 使用提示，只进尾部。"""

import asyncio

from app import compaction, context
from app.agent_loop import run_loop
from app.compaction import REFS_HEADING, STATUS_HINT, status_note

SUMMARY = """## 目标
- 筹备 2026 年会
## 约束与偏好
- 预算 37 万
## 已定决定
- 场地在杭州
## 进度
### 已完成
- 订了酒店
### 进行中
- 联系 HR-2207 确认人数
### 受阻
（无）
## 下一步
1. 发邀请函
## 未回答的用户请求
（无）

""" + REFS_HEADING + "\n" + "\n".join(f"- t{i} {{}} → ref=out_{i:016x}" for i in range(8))


def test_picks_sections_refs_and_hint():
    note = status_note(SUMMARY)
    assert note.startswith("\n<status>") and note.endswith("</status>")
    for want in ("## 目标\n- 筹备 2026 年会", "### 进行中\n- 联系 HR-2207 确认人数", "## 下一步\n1. 发邀请函", STATUS_HINT):
        assert want in note
    for skip in ("预算 37 万", "场地在杭州", "订了酒店", "受阻", "未回答的用户请求"):  # 不摘的段、内容是（无）的段都不出现
        assert skip not in note
    assert "t2 {}" not in note and all(f"- t{i} {{}}" in note for i in range(3, 8))  # 只要最近 5 个 ref
    assert status_note(SUMMARY) == note and status_note(None) == "" and status_note("") == ""


def test_long_sections_are_cut_and_words_section_is_not_copied():
    long = "## 目标\n" + "\n".join(f"- 目标{i}" * 20 for i in range(30))
    s = compaction.with_refs(long, None, [{"role": "user", "content": "用户原话不该进状态块"}])
    note = status_note(s)
    assert "完整内容见上面的摘要" in note and "用户原话不该进状态块" not in note
    assert len(note) < compaction.STATUS_CHARS + 300


def test_loop_puts_status_in_the_tail_only():
    sent = []

    async def stream(**kw):
        sent.append(kw["messages"])
        yield {"type": "final", "content": [{"type": "text", "text": "好"}], "stop_reason": "end_turn", "usage": {}}

    async def go(summary):
        return [e async for e in run_loop([{"role": "user", "content": "接着做"}], model_stream=stream, summary=summary,
                                          limits={"trigger": 10**9, "target": 10**9})]
    asyncio.run(go(SUMMARY))
    msgs = sent[0]
    assert msgs[1]["content"].startswith("<summary>") and "<status>" not in msgs[1]["content"]  # 层3 不变
    tail = msgs[-1]["content"][-1]["text"]
    assert "当前时间" in tail and "<status>" in tail and "联系 HR-2207" in tail  # 拼在层4 尾部
    asyncio.run(go(None))
    assert "<status>" not in context.text_of(sent[1][-1]["content"])  # 没压缩过就没有
