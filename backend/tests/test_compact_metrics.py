"""P1-C7：每次压缩记走法和摘要请求的命中率；评测收集检索调用次数和压缩记录；报告做压缩 / 不压缩的成对对照。"""

import asyncio
from uuid import uuid4

from app import compaction, llm
from app.agent_loop import run_loop
from app.compaction import SummaryIncomplete, make_compactor
from app.eval_report import compaction_pairs, report

TEXT = [{"type": "text", "text": "## 目标\n- 年会"}]
TOOL = [{"type": "tool_use", "id": "x", "name": "recall", "input": {}}]
HIT = {"input_tokens": 100, "cache_read_input_tokens": 900}


def fake_llm(monkeypatch, replies):
    async def stream(**kw):
        content, stop = replies.pop(0)
        yield {"type": "final", "content": content, "stop_reason": stop, "usage": HIT}
    monkeypatch.setattr(llm, "stream_message", stream)

    async def alone(**kw):
        return "单独摘要器写的摘要", "end_turn"
    monkeypatch.setattr(llm, "complete_with_stop", alone)


def prompt(s, h, note):
    return {"system": [], "messages": [{"role": "user", "content": note}], "tools": []}


def test_summarize_records_which_path(monkeypatch):
    for replies, mode, n in (([(TEXT, "end_turn")], "prefix", 1),
                             ([(TOOL, "tool_use"), (TEXT, "end_turn")], "retry", 2),
                             ([(TOOL, "tool_use"), (TOOL, "tool_use")], "alone", 2)):
        fake_llm(monkeypatch, replies)
        stats = {}
        asyncio.run(compaction.summarize(None, [{"role": "user", "content": "问"}], prompt=prompt, stats=stats))
        assert stats["mode"] == mode and stats["usages"] == [HIT] * n


H = [{"role": "user", "content": "我叫小王"}, {"role": "assistant", "content": [{"type": "text", "text": "好"}]},
     {"role": "user", "content": "定苏州"}]


def test_compactor_marks_fallback():
    async def broken(old, msgs, **kw):
        raise SummaryIncomplete("短")
    compact = make_compactor(uuid4(), 0, broken)
    asyncio.run(compact(H, None, 5, lambda s, h: 10**6 if s is None else 1, fallback=True))
    assert compact.stats["mode"] == "fallback"


def test_compacted_event_carries_mode_and_summary_hit_rate():
    async def compactor(h, s, room, measure, **kw):
        compactor.stats = {"mode": "retry", "usages": [HIT, {"input_tokens": 1000}]}
        return "摘要", h[-1:]

    async def stream(**kw):
        yield {"type": "final", "content": TEXT, "stop_reason": "end_turn", "usage": {}}

    async def go():
        return [e async for e in run_loop(H, model_stream=stream, compactor=compactor,
                                          limits={"trigger": 0, "target": 10**9})]
    ev = next(e for e in asyncio.run(go()) if e["type"] == "compacted")
    assert ev["mode"] == "retry" and ev["summary_hit_rate"] == 0.45  # 900 / 2000，按 token 加权


def rec(variant, case, trial, passed, **kw):
    return {"config": "A" if variant == "compact" else "A-对照", "variant": variant, "case": case,
            "category": "compaction", "trial": trial, "passed": passed, "latency_s": 1.0, "cost_usd": 0.001,
            "tool_calls": 1, "usage": {"input_tokens": 100}, "answer": f"{variant}{case}{trial}"} | kw


EV = {"mode": "prefix", "tokens_before": 1000, "tokens_after": 250, "summary_hit_rate": 0.9}
RECS = [rec("compact", "plate", 0, False, recall_calls=3, compactions=[EV]),
        rec("compact", "plate", 1, True, recall_calls=1, compactions=[EV | {"mode": "fallback", "tokens_after": 750,
                                                                             "summary_hit_rate": None}]),
        rec("control", "plate", 0, True, recall_calls=0), rec("control", "plate", 1, True, recall_calls=0),
        rec("compact", "tax", 0, False), rec("control", "tax", 0, False)]  # 两边都失败：题本身难，不算压缩的锅


def test_pairs_list_only_compact_fail_control_pass():
    p = compaction_pairs(RECS)
    assert [(x["case"], x["trial"]) for x in p["regressions"]] == [("plate", 0)]
    assert p["regressions"][0]["compact"] == "compactplate0" and p["regressions"][0]["control"] == "controlplate0"
    assert p["compact"]["recall_avg"] == 2.0 and p["control"]["recall_avg"] == 0.0  # 检索调用的差别看得见
    assert p["compact"]["pass@1"] == 0.333 and p["control"]["pass@1"] == 0.667
    assert p["modes"] == {"prefix": 1, "fallback": 1} and p["ratio_avg"] == 0.5 and p["summary_hit_avg"] == 0.9


def test_report_has_pair_section_and_keeps_control_out_of_main_table():
    snap = {"time": "t", "model": "m", "judge_model": "j", "trials": 2, "dataset_sha": "s", "n_cases": 2,
            "latency_note": "n", "configs": {"A": {"name": "基线"}}}
    md = report(snap, RECS)
    assert "## 压缩对照（P1-C7）" in md and "| 平均 recall 调用 / 题 | 2.0 | 0.0 |" in md
    assert "- plate#0：压缩「compactplate0」｜对照「controlplate0」" in md
    assert "| compaction | 0.25 |" in md  # 主表 A 只算会压缩的那组：plate 0.5、tax 0 → 0.25
    assert compaction_pairs([r for r in RECS if r["variant"] == "compact"]) is None  # 没跑对照就不出这一节


def test_collect_counts_retrieval_and_reads_compaction_events():
    """需要 docker compose 里的 PG。"""
    from app import db
    from app.eval_collect import collect
    from test_db import with_fresh_db

    async def body():
        uid = await db.create_user("u", "t-u")
        cid = await db.create_conversation(uid)
        pg = db.pool()
        rid = await db.start_run(cid, [{"type": "text", "text": "q"}])
        calls = [{"type": "tool_use", "id": f"c{i}", "name": n, "input": {}}
                 for i, n in enumerate(["recall", "recall", "read_tool_output", "get_weather"])]
        await pg.execute("INSERT INTO messages (conversation_id, run_id, role, content) VALUES ($1, $2, 'assistant', $3)",
                         cid, rid, calls)
        await pg.execute("INSERT INTO run_events (run_id, seq, type, data) VALUES ($1, 99, 'compacted', $2)", rid, EV)
        await pg.execute("UPDATE runs SET status = 'completed', stop_reason = 'end_turn', finished_at = now() WHERE id = $1", rid)
        _, m = await collect(pg, cid, uid, "deepseek-flash")
        assert (m["recall_calls"], m["read_output_calls"]) == (2, 1)
        assert m["compactions"] == [EV]
    with_fresh_db(body)
