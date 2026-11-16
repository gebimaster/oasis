"""P4 第 3 步：提炼器的形状校验和扫描记账。用假模型，不花钱。"""

import json

from app import db, lesson_distill as ld, lesson_store as ls, memory_store as ms
from test_lesson_trace import _conv, _run, res, text, use
from test_db import with_fresh_db


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


def fake(reply):
    seen = []

    async def complete(*, system, prompt, max_tokens):
        seen.append(prompt)
        if isinstance(reply, Exception):
            raise reply
        return (reply if isinstance(reply, str) else json.dumps({"lessons": reply}, ensure_ascii=False)), "end_turn"
    complete.seen = seen
    return complete


GOOD = {"precondition": "调用 search_flights 时", "behavior": "城市用三字码（北京 PEK）", "expected": "一次查到航班",
        "verify": [{"name": "tool_errors_max", "args": {"name": "search_flights", "max": 0}}],
        "tools": ["search_flights"], "rationale": "可能是工具只认三字码", "evidence": ["t1", "t2"]}


async def _flight_run():
    uid, cid = await _conv()
    rid = await _run(cid, "查北京到成都的航班", [
        ([use("t1", "search_flights", {"origin": "北京"})], [res("t1", "[TOOL_ERROR] INVALID_CITY", True)]),
        ([use("t2", "search_flights", {"origin": "PEK"})], [res("t2", "CA4113")]),
        ([text("CA4113")], None)])
    return uid, cid, rid


def test_shape_drops_bad_lessons_and_maps_evidence():
    async def body():
        ms.set_embedder(FakeEmbedder())
        uid, _, rid = await _flight_run()
        msg_id = (await db.pool().fetch("SELECT id FROM messages WHERE run_id = $1 AND role = 'user' "
                                        "ORDER BY id LIMIT 1", rid))[0]["id"]
        bad = [{**GOOD, "evidence": ["toolu_编的"]},                 # 证据不在轨迹里
               {**GOOD, "behavior": ""},                             # 字段空
               {**GOOD, "verify": "调一次就成功"},                    # 验证方式不是检查列表
               {**GOOD, "behavior": "x" * 301}]                      # 太长
        model = fake([{**GOOD, "evidence": ["t1", f"#{msg_id}"]}, *bad])
        out = await ld.scan(rid, complete=model)
        assert "工具调用（按时间顺序" in model.seen[0] and len(out["candidates"]) == 1
        assert out["dropped"] == 4 and out["signals"] == ["tool_error", "recovered"]  # 只看前 2 条，其余算丢掉
        lid = out["candidates"][0]
        got = await ls.get(uid, lid)
        assert got["outcome"] == "recovered" and got["status"] == "candidate" and got["tools"] == ["search_flights"]
        ev = await ls.evidence(uid, lid)
        assert [(e["message_id"] == msg_id, e["tool_use_id"]) for e in ev][-1] == (True, "") or \
            {e["tool_use_id"] for e in ev} == {"t1", ""}
        assert await db.pool().fetchval("SELECT embedding IS NOT NULL FROM memory_lessons WHERE id = $1", lid)
        assert await ld.scan(rid, complete=model) is None and len(model.seen) == 1   # 扫过的不再扫、不再花钱
    with_fresh_db(body)


def test_smooth_run_skipped_and_failure_recorded():
    async def body():
        _, cid = await _conv()
        r1 = await _run(cid, "北京天气", [([use("g", "get_weather", {"city": "北京"})], [res("g", "晴")]),
                                          ([text("晴")], None)])
        model = fake([GOOD])
        assert await ld.scan(r1, complete=model) == {"signals": [], "candidates": []} and model.seen == []
        assert await db.pool().fetchval("SELECT status FROM lesson_scans WHERE run_id = $1", r1) == "skipped"
        r2 = await _run(cid, "查北京到成都的航班", [
            ([use("t9", "search_flights", {"origin": "北京"})], [res("t9", "[TOOL_ERROR] INVALID_CITY", True)]),
            ([text("查不到")], None)])
        out = await ld.scan(r2, complete=fake("不是 JSON"))
        assert out["error"] in ("ValueError", "JSONDecodeError")
        assert await db.pool().fetchval("SELECT status FROM lesson_scans WHERE run_id = $1", r2) == "failed"
        assert await db.pool().fetchval("SELECT count(*) FROM memory_lessons") == 0
    with_fresh_db(body)


def test_at_most_two_lessons_per_episode():
    async def body():
        ms.set_embedder(FakeEmbedder())
        _, _, rid = await _flight_run()
        out = await ld.scan(rid, complete=fake([GOOD, {**GOOD, "behavior": "二"}, {**GOOD, "behavior": "三"}]))
        assert len(out["candidates"]) == 2 and out["dropped"] == 1
    with_fresh_db(body)
