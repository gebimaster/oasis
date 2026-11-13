"""Anthropic Messages 协议（含 DeepSeek 的 /anthropic 端点）。内部格式就是它，所以几乎不用翻译：
发出前按 transform.replay 筛 thinking，收到后给 thinking 打来源标签，SDK 异常包成 LLMError。"""

import os
import time
from typing import AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from .errors import _TRANSPORT, LLMError
from .registry import Model
from .transform import replay, tag_thinking

_clients: dict[tuple, AsyncAnthropic] = {}


def _get_client(model: Model) -> AsyncAnthropic:
    key = (model.base_url, model.api_key())
    if key not in _clients:  # 延迟创建，保证 .env 已加载；None 时 SDK 自己读环境变量；自带 2 次重试（429/5xx/断网）
        _clients[key] = AsyncAnthropic(api_key=key[1], base_url=key[0])
    return _clients[key]


async def complete_with_stop(model: Model, *, system: str, prompt: str, max_tokens: int) -> tuple[str, str | None]:
    try:
        resp = await _get_client(model).messages.create(
            model=model.id, max_tokens=max_tokens, system=system, messages=[{"role": "user", "content": prompt}])
    except anthropic.APIError as e:
        raise LLMError.wrap(e, model.api) from e
    return "".join(b.text for b in resp.content if b.type == "text").strip(), resp.stop_reason


async def stream_message(model: Model, *, system: list[dict], messages: list[dict], tools: list[dict],
                         tool_choice: dict) -> AsyncIterator[dict]:
    max_tokens = int(os.getenv("MAX_TOKENS", "64000"))
    t0, first = time.monotonic(), None  # O3 首字时间：发请求 → 第一个内容增量（thinking / 正文 / 工具入参都算）
    # O7：拿到响应头以后（进了 async with）SDK 就不再重试，之后的失败标 mid_stream 交给循环决定。
    # 复核②：初版在收到第一个事件时才置位，第一个 SSE 事件就是 error（过载）时，SDK 和循环都不重试
    started = False
    try:
        async with _get_client(model).messages.stream(
            model=model.id, max_tokens=max_tokens, system=system,
            messages=replay(messages, model), tools=tools, tool_choice=tool_choice,
        ) as stream:
            started = True
            async for event in stream:  # 入参 JSON 彻底解析失败时，这里会抛 ValueError（不包，循环靠它重发）
                if first is None and event.type == "content_block_delta":
                    first = time.monotonic()
                if event.type == "text":
                    yield {"type": "text_delta", "text": event.text}
            final = await stream.get_final_message()
    except anthropic.APIError as e:  # 流里的 error 事件也在这里（SDK 包成 APIStatusError，状态码是响应头的 200）
        raise LLMError.wrap(e, model.api, mid_stream=started) from e
    except _TRANSPORT as e:  # O7：流中途断网，SDK 把 httpx2 的原始异常原样抛出（_streaming.py:248-251），以前一路冒成 crashed
        raise LLMError.wrap(e, model.api, mid_stream=True) from e
    if final.stop_reason is None:  # O7 复核③：流干净地结束了、却没有 message_delta/stop（网关提前关连接）。
        # SDK 照样给出半截 final：纯文本会被当成 completed，tool_use 的 partial 入参会被执行。同 openai 的 finish_reason 检查
        raise LLMError("stream ended without stop_reason", provider=model.api, mid_stream=True, network=True)
    # 转成纯 dict：既能直接存 JSONB，也能原样作为下一轮的 assistant content 传回
    content = [b.model_dump(mode="json", exclude_none=True) for b in final.content]
    yield {
        "type": "final",
        "content": tag_thinking(content, model),
        "stop_reason": final.stop_reason,
        # M3 预算用：只留四个数，DeepSeek 的 input_tokens 不含缓存命中部分
        "usage": {k: getattr(final.usage, k, None) or 0 for k in
                  ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")},
        "first_token_ms": round((first - t0) * 1000) if first is not None else None,
    }
