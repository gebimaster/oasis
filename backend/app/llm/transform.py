"""跨模型接力（仿 pi-ai api/transform-messages.ts）：内部标准格式是 Anthropic 块，这里只管 thinking。

thinking 的签名只对产生它的模型有效，转成文本又会被新模型当成自己说过的话，所以：
- 收到回复时给 thinking 块打来源标签 _src（存库后跟着消息走）；
- 发请求前，只保留来源 == 当前模型的 thinking，其余丢掉；_src 本身不是 API 字段，一律剥掉。
改造前存下的 thinking 没有标签，都是 Anthropic 协议产生的，按「Anthropic 协议才保留」处理。
"""

from .. import surrogates
from .registry import ANTHROPIC, Model

THINKING = {"thinking", "redacted_thinking"}
SRC = "_src"


def scrub_tool_inputs(content: list[dict]) -> list[dict]:
    """O6 第二轮复核①（用户选 B）：tool 入参里的孤立代理换成 U+FFFD（规则见 app/surrogates.py）。
    只对 OpenAI Chat 协议生效：那边用 json.loads 解析入参，孤立代理会解析出来，走到这里被洗掉。
    Anthropic 协议（默认的 deepseek-flash、claude-*）下，SDK 用 jiter 拼 input_json_delta，jiter 拒收孤立代理转义、
    抛 ValueError，final 根本产不出来，这里一次也执行不到；agent_loop 当成坏 JSON 重发 2 次，第 3 次 invalid_tool_json。
    第三轮复核 A 用户选只改文档，没接管 SDK 的累积。存库那一侧由 db 的 jsonb 编码器兜底。"""
    for b in content:
        if b.get("type") == "tool_use" and "input" in b:
            b["input"] = surrogates.scrub(b["input"])
    return content


def tag_thinking(content: list[dict], model: Model) -> list[dict]:
    for b in content:
        if b.get("type") in THINKING:
            b[SRC] = model.source
    return content


def _keep(block: dict, model: Model) -> bool:
    if block.get("type") not in THINKING:
        return True
    src = block.get(SRC)
    return src == model.source if src else model.api == ANTHROPIC


def replay(messages: list[dict], model: Model) -> list[dict]:
    """返回新列表，不改调用方的 history（它还要存库、还要给下一轮用）。"""
    out = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "assistant" and isinstance(content, list):
            content = [{k: v for k, v in b.items() if k != SRC} for b in content if _keep(b, model)]
            if not content:  # 只剩 thinking 被丢光的空 assistant：服务端不收空消息
                continue
            m = {**m, "content": content}
        out.append(m)
    return out
