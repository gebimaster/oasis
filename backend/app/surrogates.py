"""孤立代理字符的清洗，O6 复核时加的。

JSON 里合法的 "\\ud800" 转义，用 Python json.loads 解析出来就是孤立代理码点；成对的转义会合成一个字符，
所以 str 里剩下的代理码点一定是孤立的。这种字符写进 Postgres jsonb 会被拒收（json.dumps 写出 \\ud800，
Postgres 不认孤立代理转义），严格的 utf-8 也编码不了。这里统一换成 U+FFFD。

两处在用：
- db 的 jsonb 编码器（第三轮复核 B，用户选甲）：所有 jsonb 列写库前都洗，工具结果、用户消息、事件、执行和审批记录都兜住；
- llm 出口的 tool 入参（第二轮复核①）：只对 OpenAI Chat 协议有用，Anthropic 协议下 SDK 先拒收了，见 llm/__init__.py。
text 列（压缩摘要、runs.error、spans.error）不经过 jsonb 编码器，没有兜住（已知不足）。
"""

import re

LONE = re.compile("[\ud800-\udfff]")
REPLACEMENT = "�"


def scrub(v):
    """递归替换 str、dict 的键和值、list 元素里的孤立代理；其余类型原样返回。"""
    if isinstance(v, str):
        return LONE.sub(REPLACEMENT, v)
    if isinstance(v, dict):
        return {scrub(k): scrub(x) for k, x in v.items()}
    if isinstance(v, list):
        return [scrub(x) for x in v]
    return v
