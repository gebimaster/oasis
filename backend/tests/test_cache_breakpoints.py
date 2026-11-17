"""P5 第 2 条：Anthropic 协议的缓存断点。在传输层截下 SDK 真正发出去的 JSON 来断言（未用真实 Claude 验证）。"""

import asyncio
import json

import anthropic
import httpx2
import pytest

from app import llm
from app.llm import anthropic_api
from app.llm.registry import resolve
from tests.test_llm_providers import _START, _sse

_END = [("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"})]
SYSTEM = [{"type": "text", "text": "平台"}, {"type": "text", "text": "工具说明"}, {"type": "text", "text": "人设"}]
MSGS = [{"role": "user", "content": "<session_context>"},
        {"role": "user", "content": [{"type": "text", "text": "你好"},
                                     {"type": "text", "text": "---\n当前时间", "_tail": {}}]}]


def sent_body(monkeypatch, model_name) -> dict:
    """按模型名走一遍 llm.stream_message，返回 SDK 发出的请求体。"""
    bodies = []

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self):
            for ev, data in _START + _END:
                yield _sse(ev, data)

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())
    client = anthropic.AsyncAnthropic(api_key="x", base_url="http://test", max_retries=0,
                                      http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(anthropic_api, "_get_client", lambda m: client)
    monkeypatch.setenv("MODEL", model_name)

    async def go():
        async for _ in llm.stream_message(system=SYSTEM, messages=MSGS, tools=[], tool_choice={"type": "auto"}):
            pass
    asyncio.run(go())
    return bodies[0]


@pytest.mark.parametrize("name", ["claude-opus-5-5", "claude-sonnet-5-5"])  # 登记过的、没登记的 claude-*
def test_claude_gets_two_breakpoints(monkeypatch, name):
    body = sent_body(monkeypatch, name)
    assert body["cache_control"] == {"type": "ephemeral"}                 # 顶层自动断点
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"}   # system 末尾
    assert json.dumps(body, ensure_ascii=False).count("cache_control") == 2  # 别处一个都没有（上限 4）
    assert "_tail" not in json.dumps(body)


def test_deepseek_request_bytes_unchanged(monkeypatch):
    """用户选 A：DeepSeek 不发 cache_control（没实测它是否真忽略）。"""
    body = sent_body(monkeypatch, "deepseek-flash")
    assert "cache_control" not in json.dumps(body)
    assert body["system"] == SYSTEM


def test_caller_system_not_mutated(monkeypatch):
    sent_body(monkeypatch, "claude-opus-5-5")
    assert all("cache_control" not in b for b in SYSTEM)  # 下一轮还要原样用它（也给 DeepSeek 用）


def test_registry_flags():
    assert resolve("claude-opus-5-5").compat.get("cache_breakpoints")
    assert resolve("claude-haiku-5-5").compat.get("cache_breakpoints")
    assert not resolve("deepseek-flash").compat.get("cache_breakpoints")
    assert not resolve("some-other-model").compat.get("cache_breakpoints")
