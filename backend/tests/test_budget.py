"""token 预算验证（M3 第 2 步）：两条线、回差、usage 打底 + 新增部分估算。"""

import pytest

from app.budget import compute_budget, estimate_text, hit_rate, request_tokens, usage_total

USAGE = {"input_tokens": 123, "cache_read_input_tokens": 768, "cache_creation_input_tokens": 0, "output_tokens": 34}


def test_budget_lines_with_hysteresis():
    b = compute_budget(1_000_000, 8_000)
    assert b == {"trigger": 792_000, "target": 692_000}  # 两条线都扣输出预留
    assert b["trigger"] - b["target"] == 100_000          # 回差 = 窗口的 10%：压一次能撑好几轮
    with pytest.raises(ValueError):
        compute_budget(10_000, 8_000)                     # 输出预留吃光了输入空间


def test_usage_total_includes_cache_hits_and_output():
    assert usage_total(USAGE) == 123 + 768 + 34           # 只看 input_tokens 会少算 768
    assert usage_total({"input_tokens": 5, "cache_read_input_tokens": None}) == 5  # 字段缺失/None 当 0


def test_estimate_errs_high_for_chinese():
    # 实测 DeepSeek：1600 个汉字（含少量请求开销）= 891 token；估算必须不低于它
    assert estimate_text("规则" * 800) >= 891
    assert estimate_text("") == 1


def test_request_tokens_uses_latest_usage_plus_new_messages():
    big = "很长的工具结果" * 1000
    history = [{"role": "user", "content": "q1"},
               {"role": "assistant", "content": [{"type": "text", "text": "a1"}], "usage": {"input_tokens": 999_999}},
               {"role": "user", "content": "q2"},
               {"role": "assistant", "content": [{"type": "text", "text": "a2"}], "usage": USAGE},
               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": big}]}]
    n = request_tokens(history, system=[], tools=[])
    assert usage_total(USAGE) + estimate_text(big) <= n < usage_total(USAGE) + estimate_text(big) + 50
    # 只取最近一条 usage，不会用到更早那条 999_999


def test_request_tokens_without_usage_estimates_everything():
    history = [{"role": "user", "content": "你好"}]
    system = [{"type": "text", "text": "规则" * 100}]
    assert request_tokens(history, system=system, tools=[]) > estimate_text("规则" * 100)


def test_request_tokens_without_usage_counts_whole_request():
    """O4：没有 usage 时按拼好的整份请求估。会话上下文里的延迟工具名录不在 history 里，以前会漏掉。"""
    history = [{"role": "user", "content": "你好"}]
    catalog = {"role": "user", "content": "<deferred_tools>" + "工具名录" * 500}
    n = request_tokens(history, system=[], tools=[], messages=[catalog] + history)
    assert n > estimate_text("工具名录" * 500)


def test_hit_rate_sums_before_dividing():
    """O3 命中率口径：cache_read / (input + cache_read + cache_creation)，多轮先各自求和再相除（按 token 加权）。"""
    small = {"input_tokens": 100, "cache_read_input_tokens": 0, "output_tokens": 50}       # 本轮 0%
    big = {"input_tokens": 100, "cache_read_input_tokens": 9800, "cache_creation_input_tokens": 100}
    assert hit_rate([big]) == 0.98
    assert hit_rate([small, big]) == round(9800 / 10100, 4)  # 按轮平均会得到 0.49
    assert hit_rate([{"output_tokens": 5}]) is None and hit_rate([]) is None  # 没有输入：不是 0%，是没法算


def test_context_window_follows_model_registry(monkeypatch):
    """窗口跟着 MODEL 走（模型表），显式 CONTEXT_WINDOW 优先；表里没有的模型宁小勿大。"""
    from app.budget import context_window, current_budget
    monkeypatch.delenv("CONTEXT_WINDOW", raising=False)
    monkeypatch.setenv("MAX_TOKENS", "64000")
    monkeypatch.setenv("MODEL", "deepseek-flash@openai")
    assert context_window() == 1_000_000
    assert current_budget() == compute_budget(1_000_000, 64000)
    monkeypatch.setenv("MODEL", "some-unknown-model")
    assert context_window() == 200_000  # 以前一律当 1M，换成小窗口模型要等服务端报超窗
    monkeypatch.setenv("CONTEXT_WINDOW", "14000")
    assert context_window() == 14000
