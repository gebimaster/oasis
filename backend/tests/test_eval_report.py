"""M8 第 3 步：收集和汇总。钉住「call_tool 拆包」「没结果的调用不算成功」「pass@1 按题平均」「环境故障不进分母」。"""

from app.eval_collect import cost_usd, extract_calls
from app.eval_report import report, summarize


def test_extract_calls_unwraps_call_tool_and_pairs_results():
    msgs = [{"role": "assistant", "content": [
                {"type": "tool_use", "id": "a", "name": "call_tool",
                 "input": {"tool_name": "get_stock", "arguments": {"symbol": "AAPL"}}},
                {"type": "tool_use", "id": "b", "name": "transfer", "input": {"to": "张三", "amount": 500}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "AAPL 231.5"}]}]
    a, b = extract_calls(msgs)
    assert (a["name"], a["input"], a["is_error"], a["result"]) == ("get_stock", {"symbol": "AAPL"}, False, "AAPL 231.5")
    assert b["is_error"] is True  # 挂起没执行：不能算「成功调用了转账」


def test_tool_error_prefix_counts_as_error():  # DeepSeek 会忽略 is_error，工具结果靠前缀标错（M1）
    msgs = [{"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "get_stock", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "[TOOL_ERROR] unknown tool"}]}]
    assert extract_calls(msgs)[0]["is_error"] is True


def test_cost_uses_cache_hit_price_for_cache_reads():
    u = {"input_tokens": 1_000_000, "cache_read_input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert round(cost_usd(u, "deepseek-flash"), 3) == 1.506  # 0.3 + 0.006 + 1.2


def test_cost_prices_cache_creation_as_miss():
    """写进缓存的那部分是一次没命中的完整计算；DeepSeek 只有命中 / 未命中两档，按未命中价（宁高勿低）。"""
    assert round(cost_usd({"cache_creation_input_tokens": 1_000_000}, "deepseek-flash"), 3) == 0.3


def rec(case, passed, cat="tool", **kw):
    return {"config": "A", "case": case, "category": cat, "passed": passed, "latency_s": 1.0, "trial": 0,
            "usage": {"input_tokens": 10}, "cost_usd": 0.001, "tool_calls": 1} | kw


def test_summarize_averages_per_case_and_excludes_env_errors():
    rs = [rec("x", True), rec("x", True), rec("x", True),          # 3 次都过
          rec("y", True), rec("y", False), rec("y", None, env_error="TimeoutError")]  # 1/2，故障那次不算
    s = summarize(rs)
    assert s["pass@1"] == 0.75 and s["pass^k"] == 0.5 and s["unstable_cases"] == 1
    assert s["env_errors"] == 1 and s["judge_errors"] == 0 and s["n_trials"] == 6
    assert s["tokens_avg"] == 10  # 故障那次没有 usage，不进平均


def test_report_marks_categories_missing_in_a_config_as_not_applicable():
    snap = {"time": "t", "model": "m", "judge_model": "j", "trials": 1, "dataset_sha": "s", "n_cases": 2,
            "latency_note": "n", "configs": {"A": {"name": "基线"}, "B": {"name": "向量"}}}
    md = report(snap, [rec("c1", True, cat="compaction"), rec("k1", True, cat="kb_in"),
                       rec("k1", False, cat="kb_in") | {"config": "B"}])
    assert "| compaction | 1.0 | 不适用 |" in md and "| kb_in | 1.0 | 0.0 |" in md


def test_report_lists_cost_and_hit_rate_per_run():
    snap = {"time": "t", "model": "m", "judge_model": "j", "trials": 1, "dataset_sha": "s", "n_cases": 1,
            "latency_note": "n", "configs": {"A": {"name": "基线"}}}
    runs = [{"run_id": "aaaaaaaa-1111", "stop_reason": "awaiting_approval", "cost_usd": 0.0012, "hit_rate": 0.5},
            {"run_id": "bbbbbbbb-2222", "stop_reason": "end_turn", "cost_usd": 0.0003, "hit_rate": None}]
    r = rec("pay", True, runs=runs, usage={"input_tokens": 100, "cache_read_input_tokens": 300})
    s = summarize([r])
    assert s["hit_rate"] == 0.75  # 汇总也按 token 加权
    md = report(snap, [r])
    assert "| 缓存命中率 | 0.75 |" in md
    assert "| A | pay | 0 | aaaaaaaa | awaiting_approval | 0.0012 | 0.5 |" in md
    assert "| A | pay | 0 | bbbbbbbb | end_turn | 0.0003 | 无输入 |" in md


def test_collect_splits_cost_and_hit_rate_per_run():
    """O3：每个父 run 一行；子 run 的 usage 算在派它的父 run 上。需要 docker compose 里的 PG。"""
    from app import db
    from app.eval_collect import collect
    from test_db import with_fresh_db

    async def body():
        uid = await db.create_user("u", "t-u")
        cid = await db.create_conversation(uid)
        pg = db.pool()
        rids = []
        for stop, u in (("awaiting_approval", {"input_tokens": 100, "cache_read_input_tokens": 300}),
                        ("end_turn", {"input_tokens": 50})):
            rid = await db.start_run(cid, [{"type": "text", "text": "q"}])
            await pg.execute("INSERT INTO messages (conversation_id, run_id, role, content, usage) VALUES ($1, $2, 'assistant', $3, $4)",
                             cid, rid, [{"type": "text", "text": "a"}], u)
            await pg.execute("UPDATE runs SET status = 'completed', stop_reason = $2, finished_at = now() WHERE id = $1", rid, stop)
            rids.append(rid)
        kid_cid = await db.create_conversation(uid)
        await pg.execute("INSERT INTO runs (conversation_id, status, parent_run_id, child_index, usage, finished_at) "
                         "VALUES ($1, 'completed', $2, 0, $3, now())", kid_cid, rids[0], {"input_tokens": 100})
        _, m = await collect(pg, cid, uid, "deepseek-flash")
        first, second = m["runs"]
        assert (first["run_id"], first["stop_reason"], first["hit_rate"]) == (str(rids[0]), "awaiting_approval", 0.6)
        assert first["cost_usd"] == round((300 * 0.006 + 200 * 0.3) / 1e6, 6)  # 子 run 的 100 未命中算在这里
        assert (second["stop_reason"], second["hit_rate"]) == ("end_turn", 0.0)
    with_fresh_db(body)


def test_o8_citation_summary_and_grid():
    """O8：引用正确率按引用数加权；四格表把「不忠实」拆成编造（引用无错）和归因错（有引错）。"""
    from app.eval_report import citation_summary
    c = lambda ok, wrong, cov, n=None: {"ok": ok, "wrong": wrong, "coverage": cov, "n": n or ok + wrong, "unchecked": 0}
    rec = lambda passed, cit: {"grades": [{"check": "answer_has", "passed": True},
                                          {"check": "faithful", "passed": passed, "citations": cit}]}
    s = citation_summary([rec(True, c(3, 0, 1.0)), rec(False, c(1, 1, 0.5)), rec(False, c(2, 0, None)),
                          rec(None, c(0, 4, 0.0))])
    assert s["citation_correctness"] == round(6 / 7, 3)   # 评委出错的那条（0 对 4 错）不算：(3+1+2)/(3+1+2 + 0+1+0)
    assert s["citation_coverage"] == 0.75                 # (1.0+0.5)/2，None 和评委出错的不进
    assert s["faithful_x_citation"] == {"忠实·引用无错": 1, "忠实·有引错": 0, "忠实·无可判引用": 0,
                                        "不忠实·引用无错": 1, "不忠实·有引错": 1, "不忠实·无可判引用": 0}
    assert citation_summary([{"grades": [{"check": "answer_has", "passed": True}]}])["citation_correctness"] is None


def test_o8_answers_without_checkable_citations_are_their_own_bucket():
    """复核⑤：没引用（或全是判不了的）不能算进「引用无错」；评委出错的两处都不算。"""
    from app.eval_report import citation_summary
    c = lambda ok, wrong, unchecked=0: {"ok": ok, "wrong": wrong, "coverage": None, "n": ok + wrong + unchecked,
                                        "unchecked": unchecked}
    rec = lambda passed, cit: {"grades": [{"check": "faithful", "passed": passed, "citations": cit}]}
    s = citation_summary([rec(False, c(0, 0)), rec(False, c(0, 0, 2)), rec(None, c(0, 5)), rec(True, c(1, 0))])
    assert s["faithful_x_citation"]["不忠实·无可判引用"] == 2 and s["faithful_x_citation"]["不忠实·引用无错"] == 0
    assert s["citation_correctness"] == 1.0 and s["citations_total"] == "3/2"
