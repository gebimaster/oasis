"""统一错误类型 + 错误分类（唯一入口）。

各 provider 把自家 SDK 的异常包成 LLMError 再抛（raise LLMError.wrap(e) from e），
循环和压缩只 except LLMError，不再 import 任何 SDK。
"""

import httpx

try:  # O7：anthropic SDK 2.x 底层用的是 httpx2（和 httpx 是两个包），流中途断网抛的是 httpx2 的异常
    import httpx2
    _TRANSPORT: tuple[type, ...] = (httpx.TransportError, httpx2.TransportError)
except ImportError:  # pragma: no cover
    _TRANSPORT = (httpx.TransportError,)

_OVERFLOW_CODES = {"context_length_exceeded", "prompt_too_long", "request_too_large"}
# 前四个是 O4 实测/Anthropic 文案；后两个取自 pi-ai utils/overflow.ts（OpenAI 官方、Groq/DeepSeek 风格）
_OVERFLOW_TEXT = ("maximum context length", "prompt is too long", "prompt_too_long", "context_length_exceeded",
                  "exceeds the context window", "reduce the length of the messages")
# O7：流里的 error 事件（Anthropic 文档的错误类型）。HTTP 头已经是 200，状态码帮不上忙，只能看 body 里的 type
# OpenAI 兼容端点流内出错时 SDK 抛不带状态码的 APIError，body 是 error 对象：type/code 用的是 OpenAI 的名字（复核⑨）
_SERVER_TYPES = {"overloaded_error", "api_error", "server_error", "service_unavailable"}
_RATE_TYPES = {"rate_limit_error", "rate_limit_exceeded"}


class LLMError(Exception):
    """模型调用失败。status_code / body 照搬 SDK 异常，classify_error 只看这几个字段和文本。
    O7：mid_stream=True 表示流已经开始以后才出错。这类错误 SDK 不会重试（它只重试拿到响应头之前的失败），
    循环可以自己重试；发请求阶段的失败 SDK 已经重试过 2 次，循环不再叠加。
    network=True 表示连接层的问题（没有 HTTP 状态可看）。"""

    def __init__(self, message: str, *, status_code: int | None = None, body=None, provider: str = "",
                 mid_stream: bool = False, network: bool = False):
        super().__init__(message)
        self.status_code, self.body, self.provider = status_code, body, provider
        self.mid_stream, self.network = mid_stream, network

    @classmethod
    def wrap(cls, exc: BaseException, provider: str = "", *, mid_stream: bool = False) -> "LLMError":
        return cls(str(exc) or type(exc).__name__, status_code=getattr(exc, "status_code", None),
                   body=getattr(exc, "body", None), provider=provider, mid_stream=mid_stream)


def classify_error(exc: BaseException) -> str:
    """O4 错误分类，只此一处。顺序：结构化错误码 → 超窗（413 或文本）→ 连接层异常 → 429 → 5xx → 其余 api_error，
    沿 __cause__ 链查（外层可能被包过）。
    DeepSeek 超窗实测：400、OpenAI 格式的 body、code 是笼统的 invalid_request_error，只有 message 认得出。
    400 不都是超窗（孤儿 tool_result 也是 400），所以不能只看状态码。
    O7 加了 network（断网、读超时）和流内 error 事件的类型；「要不要重试」不在这里判断，由循环结合 mid_stream 决定。"""
    chain, e = [], exc
    while e is not None and e not in chain:
        chain.append(e)
        e = e.__cause__
    for e in chain:
        body = getattr(e, "body", None)
        err = body.get("error", body) if isinstance(body, dict) else None
        if not isinstance(err, dict):
            continue
        codes = {str(err.get(k) or "").lower() for k in ("code", "type")}
        if codes & _OVERFLOW_CODES:
            return "context_overflow"
        if codes & _RATE_TYPES:
            return "rate_limited"
        if codes & _SERVER_TYPES:
            return "server_error"
    statuses = [s for e in chain if isinstance(s := getattr(e, "status_code", None), int)]
    if 413 in statuses or any(m in str(e).lower() for e in chain for m in _OVERFLOW_TEXT):
        return "context_overflow"
    if any(isinstance(e, _TRANSPORT) or getattr(e, "network", False) for e in chain):
        return "network"
    if 429 in statuses:
        return "rate_limited"
    if any(s >= 500 for s in statuses):
        return "server_error"
    return "api_error"
