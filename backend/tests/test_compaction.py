"""压缩验证（M3 第 3 步）：切点不切开配对、原文不改、水位正确、过期 usage 不打底、并发只有一个成功。"""

import asyncio
from uuid import uuid4

from app import budget, compaction, db, microcompact, runner, tools
from app.agent_loop import run_loop
from app.compaction import choose_cut, make_compactor
from test_db import _conversation, with_fresh_db

BIG = "很长的工具结果" * 300  # 约 2200 token


def call(i):
    # O6：每次换一个城市。原来 8 次都是 {"city": "北京"}，原样重复到第 5 次会被熔断（repeated_action）
    return {"type": "tool_use", "id": f"t{i}", "name": "get_weather", "input": {"city": f"城市{i}"}}


def result(i, text=BIG):
    return {"type": "tool_result", "tool_use_id": f"t{i}", "content": text}


def turn(i, text=BIG):
    """一轮：真提问 → assistant 调工具 → 工具结果 → assistant 回答。"""
    return [{"role": "user", "content": [{"type": "text", "text": f"第{i}个问题"}]},
            {"role": "assistant", "content": [call(i)]},
            {"role": "user", "content": [result(i, text)]},
            {"role": "assistant", "content": [{"type": "text", "text": f"第{i}个回答"}]}]


def test_cut_only_at_real_user_question():
    history = turn(1) + turn(2, "短") + turn(3, "短")
    k = choose_cut(history, limit=500)  # 第 1 轮的大结果放不下，第 2 轮起放得下
    assert k == 4 and history[k]["content"][0]["text"] == "第2个问题"
    # 额度很大时，「不是 tool_use」的规则会切在 k=2（装 tool_result 的 user）；正确规则跳到下一个真提问
    assert choose_cut(turn(1, "短") + turn(2, "短"), limit=10_000) == 4
    # choose_cut 只找「真提问」切点，找不到返回 None。O4 起 None 不再等于「不压」：
    # 压缩器会改切当前提问之后的工具轮次（本 run 内临时投影），见 test_single_round_keeps_question_and_last_group
    assert choose_cut(turn(1, "短"), limit=10_000) is None  # 只有一轮：k=0 不算，没有真提问切点
    assert choose_cut(history, limit=10) is None             # 最后一轮本身就超：真提问切点放不下


def fake_model(texts):
    calls = []

    async def stream(**kw):
        calls.append(kw)
        yield {"type": "final", "content": [{"type": "text", "text": texts[len(calls) - 1]}],
               "stop_reason": "end_turn"}
    return stream, calls


def test_compaction_end_to_end(monkeypatch):
    async def body():
        _, cid = await _conversation()
        for i in range(1, 4):  # 先存 3 轮带大工具结果的历史
            run_id = await db.start_run(cid, turn(i)[0]["content"])
            for m in turn(i)[1:]:
                await db.insert_message(cid, run_id, m["role"], m["content"])
            await db.finish_run(run_id, "completed", "end_turn")
        before = await db.list_messages(cid)

        seen = []
        async def fake_summarize(old, msgs, **kw):
            seen.append(msgs)
            return "摘要：用户问过北京天气"
        monkeypatch.setattr(compaction, "summarize", fake_summarize)

        limits = {"trigger": 4000, "target": 3500}  # 3 轮约 6600 token：超触发线；只放得下最后一轮多一点
        run_id = await db.start_run(cid, [{"type": "text", "text": "第4个问题"}])
        stream, calls = fake_model(["第4个回答", "第5个回答"])
        await runner._run(cid, run_id, model_stream=stream, limits=limits)

        cp = await db.get_compaction(cid)
        after = await db.list_messages(cid)
        assert after[:len(before)] == before                        # 原文一行没改
        kept_first = next(m for m in after if m["id"] == cp["keep_from_id"])
        assert compaction.is_real_user(kept_first)                  # 水位落在真提问上
        assert cp["revision"] == 1 and cp["summary"].startswith("摘要")
        assert len(seen) == 1 and seen[0][0]["content"][0]["text"] == "第1个问题"  # 摘要从头开始
        sent = calls[0]["messages"]
        assert sent[1]["content"].startswith("<summary>")           # 层3：摘要在原文前
        uses = {b["id"] for m in sent if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"}
        results = {b["tool_use_id"] for m in sent if isinstance(m["content"], list)
                   for b in m["content"] if b.get("type") == "tool_result"}
        assert results <= uses                                      # 没有孤儿 tool_result
        rows = await db.pool().fetch("SELECT data FROM run_events WHERE run_id = $1 AND type = 'compacted'", run_id)
        assert len(rows) == 1                                       # 压缩事件记进了事件日志

        # 第二个 run：从水位加载，前面换成摘要
        run2 = await db.start_run(cid, [{"type": "text", "text": "第5个问题"}])
        await runner._run(cid, run2, model_stream=stream, limits={"trigger": 10**9, "target": 10**9})
        # C2：摘要末尾是代码生成的 ref 段，落库后下个 run 原样带上
        assert calls[1]["messages"][1]["content"].startswith(
            "<summary>\n摘要：用户问过北京天气\n\n## 关键标识与 ref\n- get_weather {\"city\": \"城市1\"}\n")
        assert calls[1]["messages"][2]["content"][0]["text"] == kept_first["content"][0]["text"]
    with_fresh_db(body)


def test_stale_usage_ignored_after_compaction(monkeypatch):
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "q"}])
        await db.insert_message(cid, run_id, "assistant", [{"type": "text", "text": "a"}],
                                usage={"input_tokens": 790_000})          # 压缩前那份大请求的 usage
        await db.finish_run(run_id, "completed", "end_turn")
        first_id = (await db.list_messages(cid))[0]["id"]
        assert await db.commit_compaction(cid, "s", first_id, expected_revision=0)

        calls = []
        async def stream(**kw):
            calls.append(kw)
            yield {"type": "final", "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}

        async def explode(*_, **kw):
            raise AssertionError("不该再次压缩")
        monkeypatch.setattr(compaction, "summarize", explode)
        run2 = await db.start_run(cid, [{"type": "text", "text": "q2"}])
        await runner._run(cid, run2, model_stream=stream, limits={"trigger": 100_000, "target": 90_000})
        # 用 790_000 打底会超触发线 → 压缩 → explode → run 失败、模型一次都没调到
        assert len(calls) == 1 and (await db.get_run(run2))["status"] == "completed"
    with_fresh_db(body)


def test_concurrent_compactions_only_one_wins():
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "第1个问题"}])
        for m in turn(1)[1:] + turn(2, "短"):  # usage 是压缩前那份大请求的：赢家、输家返回的都要剥掉
            await db.insert_message(cid, run_id, m["role"], m["content"], usage={"input_tokens": 10**6})
        history = await db.list_messages(cid)

        def slow(name):
            async def summarize(old, msgs):
                await asyncio.sleep(0.2)  # 两边都在「调 LLM」时，revision 还都是 0
                return name
            return summarize
        a = make_compactor(cid, 0, slow("A 的摘要"))
        b = make_compactor(cid, 0, slow("B 的摘要"))
        ra, rb = await asyncio.gather(a(history, None, 500, measure), b(history, None, 500, measure))

        cp = await db.get_compaction(cid)
        assert cp["revision"] == 1                       # 只提交成功了一次
        assert ra[0] == rb[0] == cp["summary"]           # 输的一方改用赢家的摘要
        assert [m["id"] for m in ra[1]] == [m["id"] for m in rb[1]]
        assert all(m["usage"] is None for m in ra[1] + rb[1])
    with_fresh_db(body)


def test_cas_loser_rejects_broken_pairs():
    """O4 复核：抢输时按赢家水位重排历史，没 id 的消息（孤儿修复补的结果）会被无条件留下；
    它前面的调用被赢家的水位切掉，就成了孤儿 tool_result（API 400）。重排后配对不完整就这次不压，照发。"""
    from app.agent_loop import repair_orphan_tool_uses

    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "很长的问题" * 600}])
        await db.insert_message(cid, run_id, "assistant", [call(1)])  # 跑到一半中断：t1 没有结果
        for m in turn(2, "短"):
            await db.insert_message(cid, run_id, m["role"], m["content"])
        history = repair_orphan_tool_uses(await db.list_messages(cid))  # 补出来的结果没有 id
        assert history[2].get("id") is None and history[3]["content"] == turn(2)[0]["content"]

        async def summarize(old, msgs):  # 我们在调 LLM 的时候，别人抢先提交了同一个水位
            assert await db.commit_compaction(cid, "赢家的摘要", history[3]["id"], 0)
            return "我们的摘要"
        assert await make_compactor(cid, 0, summarize)(history, None, 500, measure) is None
    with_fresh_db(body)


# ---------------- O4 1a：单轮内切点、采用规则 ----------------

def single_round(n=8):
    """一个真提问连调 n 次工具，每次都是大结果：没有 k>0 的真提问可切。"""
    h = [{"role": "user", "content": [{"type": "text", "text": f"帮我对比 {n} 个城市的天气"}]}]
    for i in range(1, n + 1):
        h += [{"role": "assistant", "content": [call(i)]}, {"role": "user", "content": [result(i)]}]
    return h


def measure(summary, history):
    """单元测试里的整份请求估算：只量摘要和原文（循环里用的是 build 出来的完整请求）。"""
    return budget.estimate_json([summary, [m["content"] for m in history]])


def test_single_round_keeps_question_and_last_group():
    history = single_round()
    seen = []

    async def fake_summarize(old, msgs, **kw):
        seen.append(msgs)
        return "摘要：已查 7 个城市"
    compact = make_compactor(uuid4(), 0, fake_summarize)  # 随便一个会话 id：这条路径不碰数据库
    summary, kept = asyncio.run(compact(history, None, 500, measure))
    assert kept[0]["content"] == history[0]["content"]                               # 真提问逐字保留
    assert [m["content"] for m in kept[1:]] == [m["content"] for m in history[-2:]]  # 最近一组原样
    assert compaction.pairs_ok(kept)
    assert seen == [history[:-2]]                       # 摘要输入从 Q 开始，到倒数第二组为止
    assert measure(summary, kept) < measure(None, history)
    # 本地投影（游标没 id）也要剥掉过期 usage，否则下一轮拿它打底又立刻触发
    stale = [{**m, "usage": {"input_tokens": 10**6}} for m in history]
    assert all(m["usage"] is None for m in asyncio.run(compact(stale, None, 500, measure))[1])


def test_fit_source_counts_old_summary(monkeypatch):
    """O4 复核补：摘要源 = 旧摘要 + 对话，推迟尾部时旧摘要也占额度。"""
    h = turn(1, "短") + turn(2, "短") + turn(3, "短")
    need = budget.estimate_text("") + sum(budget.estimate_text(compaction.render([m])) for m in h[:8])  # 空串也算 1
    monkeypatch.setattr(compaction, "summary_budget", lambda: need)  # 刚好放得下前两轮
    assert compaction.fit_source(None, h, 8) == 8
    assert compaction.fit_source("旧摘要" * 200, h, 8) < 8


def test_accepts_requires_pairs_and_verbatim_protected():
    history = single_round(3)
    protected = [history[0]] + history[-2:]
    assert compaction.accepts(protected, protected)
    assert not compaction.accepts([history[0], history[2]] + history[-2:], protected)  # R1 前面没有 A1
    changed = [{**history[0], "content": "被改过的提问"}] + history[-2:]
    assert not compaction.accepts(changed, protected)
    # O4 复核补反例：保护区都在、配对也对，但顺序变了（Q 跑到最后）
    assert not compaction.accepts(history[-2:] + [history[0]], protected)
    # pairs_ok 第二条：调用的下一条必须有它的结果（A1 后面直接跟了提问）
    assert not compaction.pairs_ok([history[0], history[1], history[0]])


def test_current_question_shared_by_compact_and_window():
    """O4 复核：压缩和截窗口用同一个「找当前提问」。最后一条就是新提问时，不能再带上更早、已答完的问题。"""
    q_cur = {"role": "user", "content": [{"type": "text", "text": "新问题"}]}
    history = turn(1, "短") + [q_cur]
    assert compaction.temporary_window(history) == [q_cur]
    assert compaction.current_question(history) == (4, 4)  # 保护区本身从当前提问开始
    rounds = single_round(3)
    assert compaction.current_question(rounds) == (5, 0)  # 保护区是 A3 R3，当前提问在它前面
    assert compaction.temporary_window(rounds) == [rounds[0]] + rounds[-2:]


def test_single_round_compaction_in_loop():
    """循环里：请求排成 会话上下文 → 摘要 → Q → A8 → R8，压缩事件带前后 token 数。"""
    history = single_round()

    async def fake_summarize(old, msgs, **kw):
        return "摘要：已查 7 个城市"
    stream, calls = fake_model(["对比完了"])

    async def collect():
        return [ev async for ev in run_loop(history, model_stream=stream, limits={"trigger": 5000, "target": 4000},
                                            compactor=make_compactor(uuid4(), 0, fake_summarize))]
    events = asyncio.run(collect())
    sent = calls[0]["messages"]
    assert isinstance(sent[1]["content"], str) and sent[1]["content"].startswith("<summary>\n摘要：已查 7 个城市")
    assert [m["content"] for m in sent[2:4]] == [history[0]["content"], history[-2]["content"]]
    assert sent[4]["content"][0] == result(8)           # 尾部时间提示拼在它后面
    ev = next(e for e in events if e["type"] == "compacted")
    assert ev["tokens_after"] < ev["tokens_before"]
    assert events[-1]["status"] == "completed"


def test_long_summary_not_committed(monkeypatch):
    """摘要比原文还长：整份请求没变小，不采用，也不写库（判断在提交之前）。"""
    async def body():
        _, cid = await _conversation()
        for i in range(1, 4):
            run_id = await db.start_run(cid, turn(i)[0]["content"])
            for m in turn(i)[1:]:
                await db.insert_message(cid, run_id, m["role"], m["content"])
            await db.finish_run(run_id, "completed", "end_turn")

        async def long_summarize(old, msgs, **kw):
            return "很长的摘要" * 2000  # 1 万字，比被压掉的 3 轮原文（约 6600 token）还长
        monkeypatch.setattr(compaction, "summarize", long_summarize)
        run_id = await db.start_run(cid, [{"type": "text", "text": "第4个问题"}])
        stream, calls = fake_model(["第4个回答"])
        await runner._run(cid, run_id, model_stream=stream, limits={"trigger": 4000, "target": 3500})
        cp = await db.get_compaction(cid)
        assert cp["revision"] == 0 and cp["summary"] is None
        assert not any(isinstance(m["content"], str) and m["content"].startswith("<summary>")
                       for m in calls[0]["messages"])  # 照原样发
        assert (await db.get_run(run_id))["status"] == "completed"
    with_fresh_db(body)


TARGET = 9000
DEFS = [{"name": f"big_tool_{i}", "description": "很长的工具说明" * 25,
         "input_schema": {"type": "object", "properties": {}}} for i in range(20)]


def test_room_counts_loaded_defs(monkeypatch):
    """压缩后才出现的 loaded_defs（跟在摘要后面）要从额度里扣掉，否则压完的整份请求仍高于目标线。"""
    async def body():
        _, cid = await _conversation()
        for i in range(1, 6):
            run_id = await db.start_run(cid, turn(i)[0]["content"])
            for m in turn(i)[1:]:
                await db.insert_message(cid, run_id, m["role"], m["content"])
            await db.finish_run(run_id, "completed", "end_turn")

        async def fake_summarize(old, msgs, **kw):
            return "摘要"
        monkeypatch.setattr(compaction, "summarize", fake_summarize)
        monkeypatch.setattr(tools, "definitions", lambda names: DEFS)
        # O9：历史里有可清的天气结果，清旧结果那一级会先把请求压到线下、摘要不触发。这里只测摘要那一级的额度
        monkeypatch.setattr(microcompact, "MIN_SAVING", 10**9)
        limits = {"trigger": TARGET + 1000, "target": TARGET}
        run_id = await db.start_run(cid, [{"type": "text", "text": "第6个问题"}])
        stream, calls = fake_model(["第6个回答"])
        await runner._run(cid, run_id, model_stream=stream, limits=limits)
        sent = calls[0]
        assert sent["messages"][1]["content"].startswith("<summary>")
        assert budget.estimate_json([sent["system"], sent["messages"], sent["tools"]]) <= TARGET
    with_fresh_db(body)


# ---------------- O4 1b：单轮内压缩落库 ----------------

def test_single_round_compaction_persists_and_restores(monkeypatch):
    """一个提问连调 8 次工具、本 run 内压缩：水位落库（Q 的副本进 replay_prefix，游标指向本 run 新产生的 assistant），
    下个 run 加载成 摘要 → Q 副本 → 游标之后的原文，不用再摘要一次。"""
    async def body():
        _, cid = await _conversation()
        q1 = [{"type": "text", "text": "帮我对比 8 个城市的天气"}]

        async def fake_summarize(old, msgs, **kw):
            return "摘要：已查前几个城市"
        monkeypatch.setattr(compaction, "summarize", fake_summarize)

        async def big_results(calls):  # 每个工具都返回大结果
            return [{"type": "tool_result", "tool_use_id": c.id, "content": BIG, "is_error": False} for c in calls]
        monkeypatch.setattr(tools, "execute_all", big_results)
        calls = []

        async def stream(**kw):
            calls.append(kw)
            n = len(calls)
            content = [call(n)] if n <= 8 else [{"type": "text", "text": "对比完了" if n == 9 else "好的"}]
            yield {"type": "final", "content": content, "stop_reason": "tool_use" if n <= 8 else "end_turn"}

        run1 = await db.start_run(cid, q1)
        await runner._run(cid, run1, model_stream=stream, limits={"trigger": 8000, "target": 6000})
        cp = await db.get_compaction(cid)
        rows = {m["id"]: m for m in await db.list_messages(cid)}
        assert cp["revision"] >= 1
        assert cp["replay_prefix"] == [{"role": "user", "content": q1}]       # Q 原样存了一份
        cursor = rows[cp["keep_from_id"]]
        assert cursor["role"] == "assistant" and cursor["run_id"] == run1    # 游标指向本 run 新产生的消息
        assert (await db.get_run(run1))["status"] == "completed"

        async def explode(*_, **kw):
            raise AssertionError("水位已经落库，下个 run 不该再摘要")
        monkeypatch.setattr(compaction, "summarize", explode)
        run2 = await db.start_run(cid, [{"type": "text", "text": "哪个最热？"}])
        n_before = len(calls)
        await runner._run(cid, run2, model_stream=stream, limits={"trigger": 10**9, "target": 10**9})
        sent = calls[n_before]["messages"]
        assert sent[1]["content"].startswith("<summary>\n" + cp["summary"])
        assert sent[2] == {"role": "user", "content": q1}                    # 摘要 → Q 副本 → 游标之后
        assert sent[3]["content"] == cursor["content"]
        assert compaction.pairs_ok(sent[2:])
        assert (await db.get_run(run2))["status"] == "completed"

        # 第 3 个 run 超线、切在真提问上：整份覆盖，上一次的副本清掉
        monkeypatch.setattr(compaction, "summarize", fake_summarize)
        run3 = await db.start_run(cid, [{"type": "text", "text": "那最冷的呢？"}])
        await runner._run(cid, run3, model_stream=stream, limits={"trigger": 3000, "target": 2500})
        cp3 = await db.get_compaction(cid)
        assert cp3["revision"] > cp["revision"] and cp3["replay_prefix"] == []
        assert compaction.is_real_user((await db.list_messages(cid, from_id=cp3["keep_from_id"]))[0])
    with_fresh_db(body)


# ---------------- O4 第 3 步：摘要请求自身的预算、失败降级 ----------------

def test_summary_source_deferred_when_over_budget(monkeypatch):
    """摘要源超上限：只摘要放得下的前缀，推迟的尾部原样保留；切点不拆配对（R4 前面的 A4 一起推迟）。"""
    monkeypatch.setenv("CONTEXT_WINDOW", "14000")  # 摘要源上限约 6900 token，8 组大结果放不下
    limit = compaction.summary_budget()
    # 窗口算的是输入 + max_tokens（实测）：先扣摘要输出和摘要器 system，再按目标线比例留余量
    system = budget.estimate_text(compaction.SUMMARIZER_SYSTEM)
    assert limit == int((14000 - compaction.SUMMARY_MAX_TOKENS - system) * budget.TARGET_RATIO)
    history = single_round()
    seen = []

    async def fake_summarize(old, msgs, **kw):
        seen.append(msgs)
        return "摘要：已查前 3 个城市"
    summary, kept = asyncio.run(make_compactor(uuid4(), 0, fake_summarize)(history, None, 500, measure))
    source = seen[0]
    assert budget.estimate_text(compaction.render(source)) <= limit < budget.estimate_text(compaction.render(history[:-2]))
    assert source[0] == history[0] and source[-1]["role"] == "user"        # 从 Q 开始，停在一组结果之后
    assert [m["content"] for m in kept] == [m["content"] for m in [history[0]] + history[len(source):]]  # 尾部原样推迟
    assert compaction.pairs_ok(kept)


def test_snap_cut_keeps_groups_and_their_question():
    h = turn(1) + turn(2)  # 0 Q1, 1 A1(调用), 2 R1, 3 A1(回答), 4 Q2, 5 A2(调用), 6 R2, 7 A2(回答)
    assert compaction.snap_cut(h, 6) == 4  # 落在结果上 → 退到 A2 → A2 前面是它的提问，一起留下
    assert compaction.snap_cut(h, 3) == 3  # 落在回答上：前面是结果，不用再退
    assert compaction.snap_cut(h, 1) == 0  # 退到头：一条都放不下


def test_summary_overflow_retries_with_shorter_source():
    """摘要请求自己超窗：缩短摘要源重试，最多 MAX_SUMMARY_RETRIES 次；一直超窗就放弃（返回 None，照发）。"""
    from test_agent_loop import ptl
    history = single_round()
    sizes = []

    def flaky(fails):
        async def summarize(old, msgs):
            sizes.append(len(msgs))
            if len(sizes) <= fails:
                raise ptl()
            return "摘要"
        return summarize
    result = asyncio.run(make_compactor(uuid4(), 0, flaky(2))(history, None, 500, measure))
    assert result is not None and len(sizes) == 3 and sizes == sorted(sizes, reverse=True) and sizes[0] > sizes[-1]
    assert compaction.pairs_ok(result[1])
    sizes.clear()
    assert asyncio.run(make_compactor(uuid4(), 0, flaky(99))(history, None, 500, measure)) is None
    assert len(sizes) == compaction.MAX_SUMMARY_RETRIES + 1


def test_summary_failure_does_not_crash_run(monkeypatch):
    """摘要调用失败（5xx）：这次不压，本轮请求照发，run 正常完成，不记 crashed。"""
    from test_agent_loop import api_error
    import anthropic

    async def body():
        _, cid = await _conversation()
        for i in range(1, 4):
            run_id = await db.start_run(cid, turn(i)[0]["content"])
            for m in turn(i)[1:]:
                await db.insert_message(cid, run_id, m["role"], m["content"])
            await db.finish_run(run_id, "completed", "end_turn")

        attempts = []

        async def broken(old, msgs, **kw):
            attempts.append(1)
            raise api_error(anthropic.InternalServerError, 500, {"error": {"message": "summarizer down"}})
        monkeypatch.setattr(compaction, "summarize", broken)
        run_id = await db.start_run(cid, [{"type": "text", "text": "第4个问题"}])
        stream, calls = fake_model(["第4个回答"])
        await runner._run(cid, run_id, model_stream=stream, limits={"trigger": 4000, "target": 3500})
        assert (await db.get_run(run_id))["status"] == "completed" and len(calls) == 1
        assert (await db.get_compaction(cid))["revision"] == 0
        assert len(attempts) == 1  # 5xx 不是摘要源太长：缩短了也没用，不重试（SDK 已经重试过）
    with_fresh_db(body)


def test_incomplete_summary_not_adopted(monkeypatch):
    """O4 复核：摘要空的、被截断的、短得离谱的都算摘要失败，这次不压；
    否则新摘要顶掉旧摘要、水位越过原文，旧摘要和那段原文一起丢。"""
    from types import SimpleNamespace
    from app import llm

    def client(text, stop):
        async def create(**kw):
            return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking="想"),
                                            SimpleNamespace(type="text", text=text)], stop_reason=stop)
        return SimpleNamespace(messages=SimpleNamespace(create=create))

    def compact_with(text, stop):
        monkeypatch.setenv("MODEL", "claude-opus-5-5")  # 固定走 Anthropic 协议，不受本机 .env 影响
        monkeypatch.setattr(llm.anthropic_api, "_get_client", lambda m: client(text, stop))
        return asyncio.run(make_compactor(uuid4(), 0)(single_round(), "旧摘要：用户叫小王", 500, measure))

    assert compact_with("", "max_tokens") is None                     # thinking 吃光额度，没有正文
    assert compact_with("用户叫小王，要订周五去", "max_tokens") is None  # 写到一半被截断
    assert compact_with("好", "end_turn") is None                     # 写完了但短得不像摘要
    summary, _ = compact_with("用户叫小王，已查 7 个城市的天气，还差最后一个", "end_turn")
    assert summary.startswith("用户叫小王")                            # 对照：正常写完的照样采用
