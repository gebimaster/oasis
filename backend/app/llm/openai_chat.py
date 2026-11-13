"""OpenAI Chat Completions 协议（DeepSeek 的 https://api.deepseek.com 就是这种）。
内部格式是 Anthropic 块，进出各翻译一次（对照 pi-ai api/openai-completions.ts 的 convertMessages / 流循环）：
  tool_use 块        ⇄ assistant.tool_calls[].function.arguments（JSON 字符串）
  tool_result 块     ⇄ 单独一条 role:"tool" 消息
  thinking 块        ⇄ reasoning_content 字段（DeepSeek）
  input_schema       → {type:function, function:{parameters}}
"""

import json
import os
import time
from typing import AsyncIterator

import openai
from openai import AsyncOpenAI

from .errors import _TRANSPORT, LLMError
from .registry import Model
from .transform import replay, tag_thinking

# finish_reason → 内部（Anthropic）的 stop_reason，循环只认后者
_STOP = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "function_call": "tool_use",
         "content_filter": "refusal"}
_clients: dict[tuple, AsyncOpenAI] = {}


def _get_client(model: Model) -> AsyncOpenAI:
    key = (model.base_url, model.api_key())
    if key not in _clients:  # SDK 自带 2 次重试（429/5xx/断网），和 Anthropic 一致
        _clients[key] = AsyncOpenAI(api_key=key[1], base_url=key[0])
    return _clients[key]


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content or [] if b.get("type") == "text")


def to_messages(system, messages: list[dict], model: Model) -> list[dict]:
    """出站：Anthropic 消息 → Chat 消息。system 块拼成一条 system（cache_control 等扩展字段自然丢掉）。"""
    out = [{"role": "system", "content": s}] if (s := _text(system)) else []
    for m in replay(messages, model):
        c = m["content"]
        if m["role"] == "user":
            blocks = [{"type": "text", "text": c}] if isinstance(c, str) else c
            for b in blocks:  # tool 消息必须紧跟在 tool_calls 后面，所以先放
                if b.get("type") == "tool_result":
                    body = _text(b.get("content")) or "(no tool output)"
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                "content": f"[tool error] {body}" if b.get("is_error") else body})
            if text := _text(blocks):
                out.append({"role": "user", "content": text})
            continue
        msg = {"role": "assistant", "content": _text(c) or None}  # 发字符串，不发块数组（pi-ai 的注释：会被照抄）
        reasoning = "".join(b.get("thinking", "") for b in c if b.get("type") == "thinking")
        if reasoning or model.compat.get("reasoning_content_required"):
            msg["reasoning_content"] = reasoning  # DeepSeek 要求 assistant 带这个字段，没有就给空串
        calls = [{"id": b["id"], "type": "function",
                  "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}}
                 for b in c if b.get("type") == "tool_use"]
        if calls:
            msg["tool_calls"] = calls
        if msg["content"] or calls:  # 既没正文也没工具调用的 assistant 服务端不收
            out.append(msg)
    return out


def to_tools(tools: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t["input_schema"]}} for t in tools]


def to_tool_choice(tc: dict):
    t = tc.get("type")
    if t in ("auto", "none"):
        return t
    if t == "any":
        return "required"
    return {"type": "function", "function": {"name": tc["name"]}}  # {"type": "tool", "name": ...}


def to_usage(u) -> dict:
    """入站：usage 换成 Anthropic 口径（input 不含缓存命中）。缓存字段名各家不同，同 pi-ai parseChunkUsage。"""
    if u is None:
        return {k: 0 for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")}
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", None) if details else None) or getattr(u, "prompt_cache_hit_tokens", 0) or 0
    return {"input_tokens": max(0, (u.prompt_tokens or 0) - cached), "output_tokens": u.completion_tokens or 0,
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}


async def complete_with_stop(model: Model, *, system: str, prompt: str, max_tokens: int) -> tuple[str, str | None]:
    try:
        resp = await _get_client(model).chat.completions.create(
            model=model.id, max_tokens=max_tokens, extra_body=model.compat.get("extra_body"),
            messages=to_messages(system, [{"role": "user", "content": prompt}], model))
    except openai.APIError as e:
        raise LLMError.wrap(e, model.api) from e
    choice = resp.choices[0]
    return (choice.message.content or "").strip(), _STOP.get(choice.finish_reason, choice.finish_reason)


async def stream_message(model: Model, *, system: list[dict], messages: list[dict], tools: list[dict],
                         tool_choice: dict) -> AsyncIterator[dict]:
    kw = dict(model=model.id, max_tokens=int(os.getenv("MAX_TOKENS", "64000")), stream=True,
              stream_options={"include_usage": True}, messages=to_messages(system, messages, model))
    if tools:  # 没有工具时不能带 tool_choice
        kw.update(tools=to_tools(tools), tool_choice=to_tool_choice(tool_choice))
    if extra := model.compat.get("extra_body"):  # 厂商私有参数（如 DeepSeek 的 thinking 开关）
        kw["extra_body"] = extra
    thinking, text, calls, finish, usage = "", "", {}, None, None
    t0, first = time.monotonic(), None
    started = False  # O7：同 anthropic_api
    try:
        stream = await _get_client(model).chat.completions.create(**kw)
        started = True  # 复核②：拿到响应就算，不等第一个 chunk
        async for chunk in stream:
            usage = chunk.usage or usage  # include_usage：最后一个 chunk 只带 usage、choices 为空
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish = choice.finish_reason or finish
            d = choice.delta
            if d is None:
                continue
            r = getattr(d, "reasoning_content", None)  # SDK 没声明这个字段，靠 extra 字段读
            if first is None and (r or d.content or d.tool_calls):
                first = time.monotonic()
            thinking += r or ""
            if d.content:
                text += d.content
                yield {"type": "text_delta", "text": d.content}
            for tc in d.tool_calls or []:  # 按 index 拼：id/name 只在第一片出现，arguments 分片到达
                slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                slot["id"] = tc.id or slot["id"]
                if tc.function:
                    slot["name"] = tc.function.name or slot["name"]
                    slot["args"] += tc.function.arguments or ""
    except openai.APIError as e:
        raise LLMError.wrap(e, model.api, mid_stream=started) from e
    except _TRANSPORT as e:  # O7：流中途断网
        raise LLMError.wrap(e, model.api, mid_stream=True) from e
    if finish is None:  # 同 pi-ai：没有 finish_reason 说明流被截断了，不能当成正常结束。O7：算断网，可重试
        raise LLMError("stream ended without finish_reason", provider=model.api, mid_stream=True, network=True)
    content = [{"type": "thinking", "thinking": thinking}] if thinking else []
    if text:
        content.append({"type": "text", "text": text})
    for i in sorted(calls):
        try:
            args = json.loads(calls[i]["args"] or "{}")  # 坏 JSON 抛 ValueError，循环靠它重发这一轮（同 Anthropic SDK）
        except ValueError:
            if finish != "length":
                raise
            args = {}  # O7：被长度上限截断的入参。不是模型写坏了，重发也一样；循环对 max_tokens 的调用一律不执行
        if not isinstance(args, dict):
            raise ValueError(f"tool arguments is not an object: {calls[i]['args'][:200]}")
        content.append({"type": "tool_use", "id": calls[i]["id"], "name": calls[i]["name"], "input": args})
    yield {"type": "final", "content": tag_thinking(content, model), "stop_reason": _STOP.get(finish, finish),
           "usage": to_usage(usage), "first_token_ms": round((first - t0) * 1000) if first is not None else None}
