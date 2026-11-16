"""P3-C6：清理范围。上一回合也保最近 3 条、记忆工具可清（占位符留编号）、水位以下的旧 thinking 去掉。"""

from app import microcompact

PAGE = "这是一大段读回来的文件正文。" * 300
THINK = "让我仔细想一想这个问题的各个方面。" * 200


def q(mid, text):
    return {"id": mid, "role": "user", "content": text}


def rnd(i, mid, name="read_file", content=PAGE):
    return [{"id": mid, "role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": name, "input": {"n": i}}]},
            {"id": mid + 1, "role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": content}]}]


def results(h):
    return [b["content"] for m in h if isinstance(m["content"], list) for b in m["content"] if b.get("type") == "tool_result"]


def test_previous_turn_keeps_its_last_three():
    # 第一回合读了 5 个文件，第二回合又调了 3 次：原规则只保全局最近 3 条，第一回合 5 条全清
    h = [q(1, "读这些文件")] + sum((rnd(i, 10 + 2 * i) for i in range(5)), []) + [q(30, "再搜一下")]
    h += sum((rnd(i, 40 + 2 * i) for i in range(5, 8)), [])
    w = microcompact.next_watermark(h, 0)
    assert w == 13  # 只清第一回合的前 2 条，它最近的 3 条还在
    out = results(microcompact.project(h, w))
    assert all(x.startswith("[已清理") for x in out[:2]) and out[2:] == [PAGE] * 6


def test_current_turn_without_results_protects_the_whole_turn():
    h = [q(1, "读文件")] + sum((rnd(i, 10 + 2 * i) for i in range(8)), []) + [q(40, "第二问")]
    assert microcompact.next_watermark(h, 0) == 19  # 上一回合 8 条清前 5 条，和 O9 一样


def test_never_protects_less_than_the_global_last_three():
    # 本回合 1 条结果、上一回合没调工具：全局最近 3 条里有两条在更早的回合，照 O9 原规则也不清
    h = [q(1, "读文件")] + sum((rnd(i, 10 + 2 * i) for i in range(6)), []) + [q(30, "聊聊"),
         {"id": 31, "role": "assistant", "content": "好"}, q(32, "再读一个")] + rnd(9, 40)
    assert microcompact.next_watermark(h, 0) == 17  # 只清前 4 条，第一回合最后 2 条还在


def test_memory_placeholders_keep_ids_and_write_is_not_cleared():
    search = "找到 2 条长期记忆（…）。\n- [#12] 用户对花生过敏（2026-11-13）\n- [#38] 用户是素食者（2026-11-13）"
    p = microcompact.placeholder("memory_search", search)
    assert "#12、#38" in p and "花生" not in p
    opened = "记忆 #38（fact，user_stated，状态 superseded…）：用户是素食者\n已被 #45 取代。\n来源原文：\n--- 消息#7 user\n我在 #99 群里说过"
    p = microcompact.placeholder("memory_open", opened)
    assert "#38、#45" in p and "#7" not in p and "#99" not in p  # 证据原文里的 # 不算
    assert "memory_write" not in microcompact.CLEARABLE and {"memory_search", "memory_open"} <= microcompact.CLEARABLE


def test_old_thinking_is_dropped_behind_the_watermark():
    def ans(mid, text, think=THINK):
        return {"id": mid, "role": "assistant", "content": [{"type": "thinking", "thinking": think}, {"type": "text", "text": text}]}
    only = {"id": 4, "role": "assistant", "content": [{"type": "thinking", "thinking": THINK}]}  # 截断续写那种
    h = [q(1, "问一"), ans(2, "答一"), q(3, "问二"), only, ans(5, "答二"), q(6, "问三"), ans(7, "答三")]
    w = microcompact.next_watermark(h, 0)  # 纯聊天、没有工具结果：只靠 thinking 也能抬水位
    assert w == 5  # 本回合（问三）整回合不动
    out = microcompact.project(h, w)
    assert out[1]["content"] == [{"type": "text", "text": "答一"}] and out[4]["content"] == [{"type": "text", "text": "答二"}]
    assert out[3] == only           # 只剩 thinking 的不能变成空 assistant，原样留
    assert out[6] == h[6]           # 水位之上的不动
    assert h[1]["content"][0]["type"] == "thinking"  # 原 history 不动
    assert microcompact.project(h, w) == out  # 投影确定


def test_thinking_inside_a_long_turn_follows_the_watermark():
    h = [q(1, "对比这些文件")]
    for i in range(6):
        a, r = rnd(i, 10 + 2 * i)
        a["content"].insert(0, {"type": "thinking", "thinking": "想想"})
        h += [a, r]
    w = microcompact.next_watermark(h, 0)
    # 保护区从最近 3 条结果里最早那条（id 17）开始；发起它的 assistant（id 16）在它前面，thinking 也去掉，结果本身不动
    assert w == 16
    out = microcompact.project(h, w)
    assert [b["type"] for b in out[1]["content"]] == ["tool_use"]           # 水位以下的工具回合去掉 thinking
    assert [b["type"] for b in out[7]["content"]] == ["tool_use"] and results(out)[3] == PAGE
    assert [b["type"] for b in out[9]["content"]] == ["thinking", "tool_use"]  # 保护区里的原样
