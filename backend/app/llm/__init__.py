"""模型适配层：循环只认这里的普通 dict，不接触任何 SDK 类型。

对外接口和改造前完全一样：
- stream_message(...) 产出 {"type": "text_delta", "text"} 和 {"type": "final", "content", "stop_reason", "usage", "first_token_ms"}
- complete / complete_with_stop：非流式单轮（评委、压缩摘要用）
- classify_error：错误分类唯一入口；LLMError：所有 provider 抛的统一异常

内部标准格式是 Anthropic 块（tool_use / tool_result / thinking）。按模型名查注册表决定协议，
每个协议一个文件负责进出翻译（仿 pi-ai：anthropic-messages.ts / openai-completions.ts）。
"""

from typing import AsyncIterator

from . import anthropic_api, openai_chat, transform
from .errors import LLMError, classify_error
from .registry import ANTHROPIC, MODELS, OPENAI_CHAT, Model, resolve

_PROVIDERS = {ANTHROPIC: anthropic_api, OPENAI_CHAT: openai_chat}

__all__ = ["LLMError", "MODELS", "Model", "classify_error", "complete", "complete_with_stop", "resolve",
           "stream_message"]


def _provider(model: Model):
    try:
        return _PROVIDERS[model.api]
    except KeyError:
        raise LLMError(f"unsupported api: {model.api}") from None


async def complete(*, system: str, prompt: str, max_tokens: int = 4000, model: str | None = None) -> str:
    """非流式单轮调用（M8 评委用）。DeepSeek 会先出 thinking，只取正文。
    max_tokens 要给 thinking 留余量：实测 300 时会被 thinking 吃光、没有正文。model 不传就用被测的 MODEL。"""
    return (await complete_with_stop(system=system, prompt=prompt, max_tokens=max_tokens, model=model))[0]


async def complete_with_stop(*, system: str, prompt: str, max_tokens: int = 4000,
                             model: str | None = None) -> tuple[str, str | None]:
    """同 complete，另外返回 stop_reason（已归一成 Anthropic 口径）：O4 压缩摘要要知道它写完没有。"""
    m = resolve(model)
    return await _provider(m).complete_with_stop(m, system=system, prompt=prompt, max_tokens=max_tokens)


async def stream_message(*, system: list[dict], messages: list[dict], tools: list[dict],
                         tool_choice: dict) -> AsyncIterator[dict]:
    m = resolve()
    async for ev in _provider(m).stream_message(m, system=system, messages=messages, tools=tools,
                                                tool_choice=tool_choice):
        # O6 第二轮复核①：清洗 tool 入参里的孤立代理。只有 OpenAI Chat 协议能走到这里带着孤立代理；
        # Anthropic 协议下 SDK 的 jiter 先拒收、抛 ValueError（见 transform.scrub_tool_inputs）
        if ev["type"] == "final":
            ev["content"] = transform.scrub_tool_inputs(ev["content"])
        yield ev
