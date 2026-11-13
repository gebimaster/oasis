"""P1-C4：模型摘要失败、请求已经超窗时，不调模型的确定性降级；以及最小进展守卫（省不到 10% 不压）。"""

import asyncio
from uuid import uuid4

import pytest

from app import compaction, context, llm
from app.agent_loop import run_loop
from app.compaction import FALLBACK_HEADING, REFS_HEADING, SummaryIncomplete, fallback_summary, make_compactor

REF = "out_0123456789abcdef"


def text(role, t):
    return {"role": role, "content": [{"type": "text", "text": t}]}


HISTORY = [
    {"role": "user", "content": "我叫小王，年会预算 37 万"},
    text("assistant", "我先查一下天气"),
    {"role": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "get_stock", "input": {"symbol": "AAPL"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "很长的行情 ref=" + REF}]},
    text("assistant", "苹果股价 200 美元"),                                 # 这一回合最后一条回答
    context.loop_note("continue", context.CONTINUE_TEXT_NOTE),              # 循环插的提示，不是用户原话
    {"role": "user", "content": "长" * 3000},                               # 粘贴了一整篇
    text("assistant", "收到"),
    {"role": "user", "content": "那就定苏州"},
]


def test_fallback_keeps_user_words_and_last_answers():
    old = "## 目标\n- 年会\n\n" + REFS_HEADING + "\n- recall {}"
    out = fallback_summary(old, HISTORY[:-1])
    assert out.startswith("## 目标\n- 年会\n\n" + FALLBACK_HEADING)          # 旧摘要正文原样留着，ref 段不进正文
    lines = out.split(FALLBACK_HEADING)[1].strip().splitlines()
    assert lines[0] == "- 用户：我叫小王，年会预算 37 万"
    assert lines[1] == "- 助手：苹果股价 200 美元"                          # 只留回合里最后一条回答
    assert lines[2].startswith("- 用户：" + "长" * compaction.USER_CHARS + "……（后面还有 1000 字")
    assert lines[3] == "- 助手：收到" and len(lines) == 4                     # 工具调用、结果、续写提示都没有
    assert "很长的行情" not in out and "loop_note" not in out


def measure(summary, history):
    return len(summary or "") + 10 * len(history)


@pytest.mark.parametrize("error", [SummaryIncomplete("太短"), llm.LLMError("boom", status_code=500)])
def test_falls_back_only_on_overflow_recovery(error):
    async def broken(old, msgs, **kw):
        raise error
    compact = make_compactor(uuid4(), 0, broken)
    big = lambda s, h: 10**6 if s is None else 1  # noqa: E731  降级前很大、降级后很小：只测走不走降级
    assert asyncio.run(compact(HISTORY, None, 50, big)) is None             # 平时：不压
    summary, kept = asyncio.run(compact(HISTORY, None, 50, big, fallback=True))  # 额度 50：切到最后的提问
    assert [m["content"] for m in kept] == ["那就定苏州"]                    # 当前提问原样保留
    assert FALLBACK_HEADING in summary and "- 用户：那就定苏州" not in summary
    assert summary.endswith(REFS_HEADING + '\n- get_stock {"symbol": "AAPL"} → ref=' + REF)  # ref 段照样由代码补


def test_min_progress_guard():
    """省不到 10% 不压。9 条没有工具调用的消息 → 切到最后一条：before = 90，after = 摘要字数 + 10。"""
    plain = [text("assistant", "答") if i % 2 else {"role": "user", "content": "问"} for i in range(9)]

    def compact_with(n):
        async def fake(old, msgs, **kw):
            return "摘" * n
        return asyncio.run(make_compactor(uuid4(), 0, fake)(plain, None, 5, measure))
    assert compact_with(72) is None          # 82 > 81：只省 8.9%
    assert compact_with(71) is not None      # 81：正好省 10%


def test_loop_asks_fallback_only_after_overflow():
    seen = []

    async def compactor(h, s, room, measure, **kw):
        seen.append(kw.get("fallback", False))
        return ("摘要", h[-1:]) if kw.get("fallback") else None

    calls = []

    async def stream(**kw):
        calls.append(1)
        if len(calls) == 1:
            raise llm.LLMError("This model's maximum context length is 65536 tokens", status_code=400)
        yield {"type": "final", "content": [{"type": "text", "text": "好"}], "stop_reason": "end_turn", "usage": {}}

    async def go():
        return [e async for e in run_loop([{"role": "user", "content": "问"}], model_stream=stream, compactor=compactor,
                                          limits={"trigger": 0, "target": 10**9})]
    events = asyncio.run(go())
    assert seen[:2] == [False, True]                                        # 超触发线：不降级；超窗恢复：降级
    assert events[-1]["status"] == "completed"
