"""P5 第 3 条：收尾轮的「软 none」。DeepSeek 发 auto（none 会拿掉工具定义、整段缓存失效），Claude/OpenAI 发 none；
违约后的第二次机会一律硬 none。"""

import asyncio
from types import SimpleNamespace

import pytest

from app import llm
from tests.test_agent_loop import CALL, TEXT, USER, fake_model, run


def sent_choice(monkeypatch, model_name, tool_choice) -> dict:
    """llm.stream_message 交给协议层的 tool_choice。"""
    got = {}

    async def stream_message(m, *, system, messages, tools, tool_choice):
        got["tc"] = tool_choice
        yield {"type": "final", "content": [], "stop_reason": "end_turn"}
    monkeypatch.setattr(llm, "_provider", lambda m: SimpleNamespace(stream_message=stream_message))
    monkeypatch.setenv("MODEL", model_name)

    async def go():
        async for _ in llm.stream_message(system=[], messages=[], tools=[], tool_choice=tool_choice):
            pass
    asyncio.run(go())
    return got["tc"]


SOFT = {"type": "none", "soft": True}


@pytest.mark.parametrize("name", ["deepseek-flash", "deepseek-flash@openai"])
def test_deepseek_soft_none_becomes_auto(monkeypatch, name):
    assert sent_choice(monkeypatch, name, SOFT) == {"type": "auto"}
    assert sent_choice(monkeypatch, name, {"type": "none"}) == {"type": "none"}  # 硬 none 照发


@pytest.mark.parametrize("name", ["claude-opus-5-5", "claude-sonnet-5-5", "some-other-model"])
def test_others_keep_none_and_soft_key_never_leaks(monkeypatch, name):
    assert sent_choice(monkeypatch, name, SOFT) == {"type": "none"}  # soft 不是 API 字段，剥掉


def test_loop_first_final_attempt_soft_retry_hard():
    """收尾轮违约（仍调工具）：第一次软、第二次硬；工具不执行。"""
    stream, calls = fake_model([([CALL], "tool_use"), TEXT])
    events = run(USER, stream, max_turns=1)
    assert [c["tool_choice"] for c in calls] == [SOFT, {"type": "none"}]
    assert "tool_start" not in [e["type"] for e in events]
    assert events[-1]["status"] == "completed"
