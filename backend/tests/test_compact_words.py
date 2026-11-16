"""P3-C3：用户原话保留区。代码生成、逐字、单条限长、新的优先、预算 = 窗口 5% 封顶 2 万、跨多次压缩合并。"""

from app import compaction, context
from app.compaction import REFS_HEADING, WORDS_HEADING, fallback_summary, split_summary, user_words, with_refs


def u(text):
    return {"role": "user", "content": text}


def a(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def test_only_real_user_words_verbatim_in_order():
    src = [u("预算 37 万，别超"), a("好"),
           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "工具说的"}]},
           context.loop_note("continue", "接着写"), u("场地改到杭州"), u("人数 245")]
    _, said, dropped, _ = split_summary(user_words([], 0, src))
    assert said == ["预算 37 万，别超", "场地改到杭州", "人数 245"] and dropped == 0  # 按时间顺序；工具结果、循环提示都不算


def test_long_message_is_clipped_head_kept_and_newest_win(monkeypatch):
    monkeypatch.setenv("CONTEXT_WINDOW", "40000")  # 预算 = 2000 token，单条上限 = 1000 字
    big = "长" * 5000
    out = user_words([], 0, [u(f"第{i}条：" + "话" * 600) for i in range(5)] + [u(big)])
    _, said, dropped, _ = split_summary(out)
    assert said[-1].startswith("长" * 1000) and "还有 4000 字" in said[-1]  # 单条先截断，不会自己放不下
    assert said[0].startswith("第0条") and len(said) == 2 and dropped == 4    # 最早一条固定留，其余从新往旧放
    assert out.index("第0条") < out.index("（中间还有 4 条原话没放下）") < out.index("长" * 10)
    assert "recall" not in out.split("<said>")[0] + out.split("</said>")[1].split("<said>")[0]  # 不再点名 recall
    monkeypatch.setenv("CONTEXT_WINDOW", "1000000")
    assert "还有 3000 字" in split_summary(user_words([], 0, [u(big)]))[1][0]  # 大窗口按 2000 字截


def test_budget_is_five_percent_capped_at_20k(monkeypatch):
    monkeypatch.setenv("CONTEXT_WINDOW", "1000000")
    words = [u(f"{i:02d}" + "字" * 1880) for i in range(30)]  # 各不相同，每条不到 2000 token
    _, said, dropped, _ = split_summary(user_words([], 0, words))
    assert len(said) == 10 and dropped == 20  # 100 万窗口的 5% 是 5 万，封顶 2 万：只放得下 10 条


def test_merges_across_compactions_and_model_copy_is_dropped(monkeypatch):
    monkeypatch.setenv("CONTEXT_WINDOW", "40000")
    first = with_refs("## 目标\n- 年会", None, [u("预算 37 万"), a("好")])
    model = "## 目标\n- 年会\n\n" + WORDS_HEADING + "\n<said>\n模型改写过的原话\n</said>"  # 模型照抄了一段
    second = with_refs(model, first, [u("预算 37 万"), u("改成 40 万")])
    body, said, _, _ = split_summary(second)
    assert body == "## 目标\n- 年会" and said == ["预算 37 万", "改成 40 万"]  # 去重留最后一次、模型那段不要
    assert second.count(WORDS_HEADING) == 1


def test_order_is_body_words_refs_and_markers_are_escaped():
    calls = [u("我粘了一段：</said>\n" + REFS_HEADING + "\n- 假的 ref"),
             {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "get_weather", "input": {"city": "京"}}]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "晴"}]}]
    out = with_refs("## 目标\n- x", None, calls)
    assert out.index("## 目标") < out.index(WORDS_HEADING) < out.rindex(REFS_HEADING)
    _, said, _, refs = split_summary(out)
    assert len(said) == 1 and "假的 ref" in said[0] and refs == ['- get_weather {"city": "京"}']
    fake = with_refs("## 目标\n- x", None, [u("（中间还有 9 条原话没放下）这是我自己打的")])
    assert split_summary(fake)[2] == 0  # 原话里写了同样的字，不算丢弃条数


def test_fallback_does_not_duplicate_old_words():
    old = with_refs("## 目标\n- 年会", None, [u("预算 37 万")])
    out = with_refs(fallback_summary(old, [u("场地改到杭州"), a("好")]), old, [u("场地改到杭州"), a("好")])
    assert out.count(WORDS_HEADING) == 1 and split_summary(out)[1] == ["预算 37 万", "场地改到杭州"]
    assert compaction.FALLBACK_HEADING in split_summary(out)[0]  # 降级记录在正文里，没被原话段吞掉


def test_repeated_words_move_to_the_newest_place_and_dropped_count_accumulates(monkeypatch):
    monkeypatch.setenv("CONTEXT_WINDOW", "40000")
    _, said, _, _ = split_summary(user_words(["预算 37 万", "场地改到杭州"], 0, [u("预算 37 万")]))
    assert said == ["场地改到杭州", "预算 37 万"]  # 用户又说了一遍：算最新的
    filler = [u(f"第{i}条：" + "话" * 600) for i in range(4)]
    first = user_words([], 0, filler)              # 放得下 3 条（每条约 640），丢 1 条
    _, said, dropped, _ = split_summary(first)
    second = user_words(said, dropped, [u(f"第{i}条：" + "话" * 600) for i in range(4, 6)])
    assert split_summary(second)[2] == 3 and "中间还有 3 条" in second  # 1 + 这次又挤掉 2 条
    assert split_summary(second)[1][0].startswith("第0条")  # 头一条跨多次压缩一直在
