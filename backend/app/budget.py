"""token 预算（M3 第 2 步）：算触发线/目标线，估当前请求有多大。

计数策略 = 「最近一条带 usage 的 assistant 消息」打底 + 它之后新增消息的本地估算：
- usage 是服务端算的，准，而且免费（不像 count_tokens 每次要 0.5~1s 网络往返）；
- usage 跟着 assistant 消息存进库，新 run 的第一轮也有底可用；
- 本地估算只覆盖新增的一小段，宁大勿小：多估只是早一点压缩，少估会撑爆窗口。
"""

import json
import os

from app import llm

USAGE_RATIO = 0.8   # 窗口 > 256K 时取 80%
TARGET_RATIO = 0.7  # 压缩目标线，比触发线低，留回差（window.py:28）
SUMMARY_RESERVE = 2000  # 压缩时给新摘要预留的 token


def compute_budget(window: int, output_reserve: int) -> dict:
    """trigger：超过就压缩；target：压到这条线以下。两条线都已扣掉输出预留。"""
    trigger = int(window * USAGE_RATIO) - output_reserve
    target = int(window * TARGET_RATIO) - output_reserve
    if target <= 0:
        raise ValueError(f"窗口 {window} 扣掉输出预留 {output_reserve} 后没有输入空间")
    return {"trigger": trigger, "target": target}


def context_window() -> int:
    """当前模型的窗口：显式设了 CONTEXT_WINDOW 就用它（测试调小才能触发压缩），否则查模型表（换模型自动跟着变）。
    表里没有的模型按默认 200K 算，宁小勿大：估小了只是早压缩，估大了要等服务端报超窗。"""
    if v := os.getenv("CONTEXT_WINDOW"):
        return int(v)
    return llm.resolve().context_window


def current_budget() -> dict:
    return compute_budget(context_window(), int(os.getenv("MAX_TOKENS", "64000")))


def estimate_text(s: str) -> int:
    """本地估算（经验系数）。对 DeepSeek 中文偏大，正合宁大勿小。"""
    cjk = sum(1 for ch in s if "一" <= ch <= "鿿")
    return int(cjk * 1.05 + (len(s) - cjk) / 3.6) + 1


def estimate_json(obj) -> int:
    return estimate_text(json.dumps(obj, ensure_ascii=False))


def usage_total(u: dict) -> int:
    """这次请求的完整输入 + 输出。DeepSeek 的 input_tokens 不含缓存命中（实测），要把三项加起来；
    输出会变成下一轮历史里的 assistant 消息，所以也算进去。"""
    return sum(u.get(k) or 0 for k in
               ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"))


def hit_rate(usages: list[dict]) -> float | None:
    """O3 命中率口径：cache_read / (input + cache_read + cache_creation)。DeepSeek 的 input 不含命中，三项才是完整输入。
    多轮（一个 run、一道题）先各自求和再相除：按 token 加权，小轮不会把大轮的命中率拉偏。没有输入时返回 None。"""
    read = sum(u.get("cache_read_input_tokens") or 0 for u in usages)
    total = sum(u.get(k) or 0 for u in usages
                for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    return round(read / total, 4) if total else None


def request_tokens(history: list[dict], *, system, tools, summary: str | None = None,
                   messages: list[dict] | None = None) -> int:
    """估算下一次请求的输入 token。history 里的 assistant 消息可带 usage 字段（压缩后会被清掉）。
    messages 是拼好的整份请求消息（O4）：给了就用它兜底，名录、loaded_defs、尾部提示都算进去。"""
    for i in range(len(history) - 1, -1, -1):
        u = history[i].get("usage")
        if u:  # 那次请求已包含 system/tools/会话上下文/摘要，只需补上之后新增的
            return usage_total(u) + estimate_json([m["content"] for m in history[i + 1:]])
    # 没有可用 usage（新会话或刚压缩完）：整份本地估
    if messages is not None:
        return estimate_json([system, tools, messages])
    return estimate_json([system, tools, summary, [m["content"] for m in history]])
