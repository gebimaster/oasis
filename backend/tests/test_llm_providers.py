"""多协议适配层：OpenAI Chat 双向翻译、thinking 跨模型接力、错误归一、按模型名路由。全部用假客户端，不连网。"""

import asyncio
from types import SimpleNamespace as NS

import httpx2
import openai
import pytest

from app import llm
from app.llm import openai_chat, transform
from app.llm.registry import ANTHROPIC, MODELS

OAI = MODELS["deepseek-flash@openai"]
ANT = MODELS["deepseek-flash"]

# 一段完整的多轮历史（内部 Anthropic 格式）：Anthropic 协议产生的 thinking 没有标签（改造前的存量）
HISTORY = [
    {"role": "user", "content": "北京天气？"},
    {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "旧模型想", "signature": "sig"},
        {"type": "thinking", "thinking": "新模型想", "_src": OAI.source},
        {"type": "text", "text": "我查一下"},
        {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "北京"}}]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": [{"type": "text", "text": "晴 20 度"}]},
        {"type": "text", "text": "（提醒：简短回答）"}]},
]


def test_to_messages_translates_blocks():
    out = openai_chat.to_messages([{"type": "text", "text": "规则", "cache_control": {"type": "ephemeral"}}],
                                  HISTORY, OAI)
    assert out[0] == {"role": "system", "content": "规则"}
    assert out[1] == {"role": "user", "content": "北京天气？"}
    a = out[2]
    assert a["content"] == "我查一下" and a["reasoning_content"] == "新模型想"  # 只回传自己产生的 thinking
    assert a["tool_calls"] == [{"id": "call_1", "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}]
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": "晴 20 度"}  # tool 紧跟 tool_calls
    assert out[4] == {"role": "user", "content": "（提醒：简短回答）"}


def test_reasoning_content_required_and_error_result():
    hist = [{"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "x", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "坏了", "is_error": True}]}]
    out = openai_chat.to_messages([], hist, OAI)
    assert out[0]["reasoning_content"] == "" and out[0]["content"] is None  # DeepSeek 要求有这个字段
    assert out[1]["content"] == "[tool error] 坏了"


def test_replay_for_anthropic_drops_foreign_thinking_and_strips_tag():
    out = transform.replay(HISTORY, ANT)
    blocks = out[1]["content"]
    assert [b.get("thinking") for b in blocks if b["type"] == "thinking"] == ["旧模型想"]  # 存量保留，OpenAI 产生的丢
    assert all("_src" not in b for b in blocks)
    assert "_src" in HISTORY[1]["content"][1]  # 不改调用方的 history


def test_replay_skips_assistant_left_empty():
    hist = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "x", "_src": "other:m"}]}]
    assert transform.replay(hist, OAI) == []


def test_tool_choice_mapping():
    assert openai_chat.to_tool_choice({"type": "auto"}) == "auto"
    assert openai_chat.to_tool_choice({"type": "none"}) == "none"
    assert openai_chat.to_tool_choice({"type": "any"}) == "required"
    assert openai_chat.to_tool_choice({"type": "tool", "name": "f"}) == {"type": "function", "function": {"name": "f"}}


def chunk(content=None, reasoning=None, tool_calls=None, finish=None, usage=None, choices=True):
    delta = NS(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return NS(usage=usage, choices=[NS(delta=delta, finish_reason=finish)] if choices else [])


def tc(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def fake_client(monkeypatch, chunks=None, exc=None):
    calls = []

    async def create(**kw):
        calls.append(kw)
        if exc:
            raise exc

        async def gen():
            for c in chunks:
                yield c
        return gen()
    monkeypatch.setattr(openai_chat, "_get_client", lambda m: NS(chat=NS(completions=NS(create=create))))
    return calls


def run_stream(tools=(), tool_choice=None):
    async def go():
        return [ev async for ev in llm.stream_message(system=[{"type": "text", "text": "s"}], messages=HISTORY,
                                                      tools=list(tools), tool_choice=tool_choice or {"type": "auto"})]
    return asyncio.run(go())


TOOL = {"name": "get_weather", "description": "天气", "input_schema": {"type": "object", "properties": {}}}


def test_stream_assembles_final(monkeypatch):
    monkeypatch.setenv("MODEL", "deepseek-flash@openai")
    calls = fake_client(monkeypatch, [
        chunk(reasoning="先想"), chunk(content="好的"), chunk(content="，查"),
        chunk(tool_calls=[tc(0, "call_9", "get_weather", '{"ci')]), chunk(tool_calls=[tc(0, args='ty": "上海"}')]),
        chunk(finish="tool_calls"),
        chunk(choices=False, usage=NS(prompt_tokens=100, completion_tokens=7, prompt_tokens_details=None,
                                      prompt_cache_hit_tokens=60))])
    evs = run_stream([TOOL], {"type": "none"})
    assert [e["text"] for e in evs if e["type"] == "text_delta"] == ["好的", "，查"]
    final = evs[-1]
    assert final["content"] == [
        {"type": "thinking", "thinking": "先想", "_src": OAI.source},
        {"type": "text", "text": "好的，查"},
        {"type": "tool_use", "id": "call_9", "name": "get_weather", "input": {"city": "上海"}}]
    assert final["stop_reason"] == "tool_use"
    assert final["usage"] == {"input_tokens": 40, "output_tokens": 7, "cache_read_input_tokens": 60,
                              "cache_creation_input_tokens": 0}
    kw = calls[0]
    assert kw["tool_choice"] == "none" and kw["tools"][0]["function"]["parameters"] == TOOL["input_schema"]
    assert kw["model"] == "deepseek-flash" and kw["stream_options"] == {"include_usage": True}
    assert kw["extra_body"] == {"thinking": {"type": "enabled"}}  # 实测 OpenAI 端点默认不思考，要显式打开


def test_no_tools_means_no_tool_choice(monkeypatch):
    monkeypatch.setenv("MODEL", "deepseek-flash@openai")
    calls = fake_client(monkeypatch, [chunk(content="hi", finish="stop")])
    assert run_stream()[-1]["stop_reason"] == "end_turn"
    assert "tool_choice" not in calls[0] and "tools" not in calls[0]


def test_bad_tool_json_raises_value_error(monkeypatch):
    monkeypatch.setenv("MODEL", "deepseek-flash@openai")
    fake_client(monkeypatch, [chunk(tool_calls=[tc(0, "c", "f", '{"a": ')]), chunk(finish="tool_calls")])
    with pytest.raises(ValueError):  # 循环靠 ValueError 重发这一轮
        run_stream([TOOL])


def test_missing_finish_reason_is_error(monkeypatch):
    monkeypatch.setenv("MODEL", "deepseek-flash@openai")
    fake_client(monkeypatch, [chunk(content="写到一半")])
    with pytest.raises(llm.LLMError):
        run_stream()


def oai_error(cls, status, body):
    req = httpx2.Request("POST", "https://api.deepseek.com/chat/completions")
    return cls(f"Error code: {status} - {body}", response=httpx2.Response(status, request=req), body=body)


@pytest.mark.parametrize("cls,status,body,kind", [
    (openai.BadRequestError, 400, {"error": {"message": "This model's maximum context length is 1048576 tokens.",
                                             "type": "invalid_request_error", "code": "invalid_request_error"}},
     "context_overflow"),
    (openai.BadRequestError, 400, {"error": {"message": "tool_call_id not found", "code": "invalid_request_error"}},
     "api_error"),
    (openai.RateLimitError, 429, {"error": {"message": "slow"}}, "rate_limited"),
    (openai.InternalServerError, 500, {"error": {"message": "x"}}, "server_error"),
])
def test_openai_errors_wrapped_and_classified(monkeypatch, cls, status, body, kind):
    monkeypatch.setenv("MODEL", "deepseek-flash@openai")
    fake_client(monkeypatch, exc=oai_error(cls, status, body))
    with pytest.raises(llm.LLMError) as ei:
        run_stream()
    assert ei.value.status_code == status and llm.classify_error(ei.value) == kind


def test_complete_routes_by_model_name(monkeypatch):
    """评委和被测模型可以走不同协议：model 参数决定，不看 MODEL。"""
    monkeypatch.setenv("MODEL", "deepseek-flash")

    async def create(**kw):
        assert kw["messages"][0] == {"role": "system", "content": "评委"}
        return NS(choices=[NS(message=NS(content=" 8 分 "), finish_reason="length")])
    monkeypatch.setattr(openai_chat, "_get_client", lambda m: NS(chat=NS(completions=NS(create=create))))
    out = asyncio.run(llm.complete_with_stop(system="评委", prompt="打分", model="deepseek-flash@openai"))
    assert out == ("8 分", "max_tokens")


def test_unknown_model_falls_back_to_anthropic():
    m = llm.resolve("some-new-claude")
    assert m.api == ANTHROPIC and m.id == "some-new-claude"



def test_lone_surrogate_in_tool_input_is_replaced(monkeypatch):
    """O6 第二轮复核①（你选 B）：模型返回的 tool 入参里有 JSON 合法的孤立代理字符（"\\ud800"），解析后是孤立代理，
    Postgres jsonb 拒收，run 会 internal_error。llm.stream_message 出口换成 U+FFFD，键和嵌套值都换；成对的不动。
    只测 @openai：这条路才走得到清洗。Anthropic 协议下 SDK 的 jiter 先拒收孤立代理转义（第三轮复核 A，只改了文档）。"""
    import json as _json
    monkeypatch.setenv("MODEL", "deepseek-flash@openai")
    args = ('{"q": "a\\ud800b", "\\udc00k": ["\\udfff", "\\ud83d\\ude00"], '
            '"o": {"x": "\\ud800"}, "l": [{"y": "\\udfff"}]}')
    # 上面是 JSON 转义：前几个都是孤立代理（值、键、list 里的值、dict 套 dict、list 套 dict），"ud83d ude00" 那一对是 😀，不该动
    fake_client(monkeypatch, [chunk(tool_calls=[tc(0, "call_1", "get_weather", args)]), chunk(finish="tool_calls"),
                              chunk(choices=False, usage=None)])
    final = run_stream([TOOL])[-1]
    got = next(b for b in final["content"] if b["type"] == "tool_use")["input"]
    assert got == {"q": "a�b", "�k": ["�", "😀"], "o": {"x": "�"}, "l": [{"y": "�"}]}
    _json.dumps(got, ensure_ascii=False).encode("utf-8")  # 不转义直接按严格 utf-8 编码：还有孤立代理就会抛


# ---------------- O7：流中途出错、被长度截断的入参 ----------------
# Anthropic 协议用真 SDK + httpx2 的 MockTransport，只假传输层：SDK 自己的 _streaming 代码照常跑。
# 测试必须抛 httpx2 的异常：抛 httpx.ReadError 的话分类器写成只认 httpx 也会变绿，线上照样接不住。

def _sse(ev, data):
    import json
    return f"event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


_START = [("message_start", {"type": "message_start", "message": {
              "id": "m", "type": "message", "role": "assistant", "model": "x", "content": [], "stop_reason": None,
              "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}}),
          ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
          ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": "北京"}})]


def anthropic_client(monkeypatch, events, then=None, connect_error=False):
    import anthropic
    from app.llm import anthropic_api

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self):
            for ev, data in events:
                yield _sse(ev, data)
            if then:
                raise then

    def handler(request):
        if connect_error:
            raise httpx2.ConnectError("refused")
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())
    client = anthropic.AsyncAnthropic(api_key="x", base_url="http://test", max_retries=0,
                                      http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(anthropic_api, "_get_client", lambda m: client)


def _collect(evs):
    async def go():
        async for ev in llm.stream_message(system=[{"type": "text", "text": "s"}],
                                           messages=[{"role": "user", "content": "hi"}], tools=[],
                                           tool_choice={"type": "auto"}):
            evs.append(ev)
    asyncio.run(go())


def test_anthropic_mid_stream_disconnect_becomes_retryable_llm_error(monkeypatch):
    monkeypatch.setenv("MODEL", "deepseek-flash")
    anthropic_client(monkeypatch, _START, then=httpx2.ReadError("connection reset"))
    evs = []
    with pytest.raises(llm.LLMError) as info:  # 修前：httpx2.ReadError 原样冒出去，runner 记成 crashed/internal_error
        _collect(evs)
    assert [e["text"] for e in evs] == ["北京"]  # 先推出了字：循环要靠这一点决定撤回
    assert info.value.mid_stream and llm.classify_error(info.value) == "network"


def test_anthropic_in_stream_error_event_is_classified(monkeypatch):
    """复核②：第一个 SSE 事件就是 error（200 响应头已经到了，SDK 不会重试）也要标 mid_stream，循环才会重试。
    初版在收到第一个事件时才置位，这种情况两边都不重试；前面带 _START 的写法测不出来。"""
    monkeypatch.setenv("MODEL", "deepseek-flash")
    anthropic_client(monkeypatch, [("error", {"type": "error", "error": {
        "type": "overloaded_error", "message": "Overloaded"}})])
    with pytest.raises(llm.LLMError) as info:
        _collect([])
    assert info.value.mid_stream and llm.classify_error(info.value) == "server_error"


def test_anthropic_connect_error_is_not_mid_stream(monkeypatch):
    """发请求阶段的失败：SDK 自己重试（这里 max_retries=0），包成 APIConnectionError；循环不再叠加重试。"""
    monkeypatch.setenv("MODEL", "deepseek-flash")
    anthropic_client(monkeypatch, [], connect_error=True)
    with pytest.raises(llm.LLMError) as info:
        _collect([])
    assert not info.value.mid_stream and llm.classify_error(info.value) == "network"


def test_openai_length_truncated_tool_args_become_empty_input(monkeypatch):
    """finish_reason=length 时入参 JSON 没写完：不是模型写坏了，不走 invalid_tool_json 重发，交给循环按截断处理。"""
    monkeypatch.setenv("MODEL", "deepseek-flash@openai")
    fake_client(monkeypatch, [chunk(tool_calls=[tc(0, "c", "get_weather", '{"city": "北')]), chunk(finish="length")])
    final = run_stream([TOOL])[-1]
    assert final["stop_reason"] == "max_tokens"
    assert final["content"][-1] == {"type": "tool_use", "id": "c", "name": "get_weather", "input": {}}


def test_openai_missing_finish_reason_is_retryable(monkeypatch):
    monkeypatch.setenv("MODEL", "deepseek-flash@openai")
    fake_client(monkeypatch, [chunk(content="写到一半")])
    with pytest.raises(llm.LLMError) as info:
        run_stream()
    assert info.value.mid_stream and llm.classify_error(info.value) == "network"



def test_anthropic_stream_without_stop_reason_is_a_disconnect(monkeypatch):
    """复核③：流干净结束、却没有 message_delta/message_stop。SDK 照样给半截 final（stop_reason=None）：
    纯文本会被当成 completed，tool_use 的 partial 入参会被执行。当成断网，交给循环重试。"""
    monkeypatch.setenv("MODEL", "deepseek-flash")
    anthropic_client(monkeypatch, _START)
    evs = []
    with pytest.raises(llm.LLMError) as info:
        _collect(evs)
    assert [e["type"] for e in evs] == ["text_delta"]  # 没有 final
    assert info.value.mid_stream and llm.classify_error(info.value) == "network"
