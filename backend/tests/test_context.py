"""上下文分层验证（M3 第 1 步）：顺序、system 稳定、尾部只在副本上、status 不外泄。"""

import copy
from datetime import datetime, timedelta, timezone

from app.context import FINAL_TURN_NOTE, INTERRUPTED_NOTE, build_request

NOW = datetime(2026, 10, 28, 21, 5, tzinfo=timezone(timedelta(hours=8)))
TOOL_USE = {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}}
TOOL_RESULT = {"type": "tool_result", "tool_use_id": "t1", "content": "晴"}


def build(history, **kw):
    args = dict(user_name="小刘", summary=None, history=history, now=NOW, final_turn=False) | kw
    return build_request(**args)


def test_layers_in_stability_order():
    system, msgs = build([{"role": "user", "content": "你好"}], summary="之前聊了天气")
    assert [b["text"][:6] for b in system] == ["你运行在 o", "需要时调用工", "你是一个有帮"]  # 平台 → 工具 → 人设
    assert "用户名：小刘" in msgs[0]["content"]           # 层2 会话上下文
    assert msgs[1]["content"].startswith("<summary>")    # 层3 摘要在原文之前
    assert msgs[2]["content"][0] == {"type": "text", "text": "你好"}
    assert "当前时间：2026-10-28 21:05 (UTC+0800)" in msgs[2]["content"][-1]["text"]  # 层4 在最末尾


def test_final_turn_only_changes_tail_not_system():
    history = [{"role": "user", "content": "你好"}]
    sys_a, msgs_a = build(history)
    sys_b, msgs_b = build(history, final_turn=True)
    assert sys_a == sys_b                     # 最后一轮 system 不变：前缀缓存不失效
    assert msgs_a[:-1] == msgs_b[:-1]         # 只有最后一条不同
    assert FINAL_TURN_NOTE in msgs_b[-1]["content"][-1]["text"]


def test_tail_goes_after_tool_result_and_history_untouched():
    history = [{"role": "user", "content": "北京天气"},
               {"role": "assistant", "content": [TOOL_USE]},
               {"role": "user", "content": [TOOL_RESULT]}]
    before = copy.deepcopy(history)
    _, msgs = build(history)
    assert msgs[-1]["content"][0] == TOOL_RESULT         # tool_result 仍在最前
    assert msgs[-1]["content"][1]["text"].startswith("---\n当前时间")
    assert history == before                              # 尾部只拼在副本上，不污染要落库的历史


def test_pause_turn_last_assistant_gets_no_tail():
    history = [{"role": "user", "content": "搜一下"},
               {"role": "assistant", "content": [{"type": "text", "text": "正在"}]}]
    _, msgs = build(history)
    assert msgs[-1] == history[-1]  # 让模型原样续写


def test_interrupted_marked_and_status_stripped():
    history = [{"role": "user", "content": "讲个故事", "status": "completed"},
               {"role": "assistant", "content": [{"type": "text", "text": "从前有座"}], "status": "interrupted"},
               {"role": "user", "content": "继续", "status": "completed"}]
    _, msgs = build(history)
    assert all(set(m) == {"role", "content"} for m in msgs)  # status 不发给 API
    assert msgs[2]["content"] == [{"type": "text", "text": "从前有座"}, {"type": "text", "text": INTERRUPTED_NOTE}]
    assert history[1]["content"] == [{"type": "text", "text": "从前有座"}]  # 原文不改
