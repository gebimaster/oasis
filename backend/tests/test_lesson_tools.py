"""P4 第 5 步：经验的读取路径。挑选（关键词 / 向量 / 限量）、尾部注入不动前缀、memory_open 下钻、run 结束判违反。"""

from app import db, lesson_store as ls, lesson_tools as lt, memory_store as ms, tools
from test_db import with_fresh_db
from test_events import _run_and_collect
from test_lesson_trace import _conv, _run, res, text, use

H = {"precondition": "用户用公司中文名问美股股价时", "behavior": "先把公司名换成股票代码再调 get_stock",
     "expected": "get_stock 一次成功", "rationale": "可能是工具只认代码",
     "verify": [{"name": "tool_errors_max", "args": {"name": "get_stock", "max": 0}}], "tools": ["get_stock"],
     "outcome": "recovered"}


class Emb:
    """按文字里有没有「股价」给两种向量：同类的余弦相似度 1，不同类 0。"""
    async def embed(self, texts):
        return [[1.0] + [0.0] * 1023 if "股价" in t else [0.0, 1.0] + [0.0] * 1022 for t in texts]


class Broken:
    async def embed(self, texts):
        raise RuntimeError("down")


async def _lesson(uid, rid, mid, status="active", **kw):
    lid = await ls.add_candidate(uid, rid, {**H, **kw}, [(mid, "s1")])
    await ls.index(lid, {**H, **kw}["precondition"])
    if status != "candidate":
        await ls.decide(lid, "l1", {"ok": status == "active", "summary": "收益 1、回归 0"}, status)
    return lid


async def _source(name="alice"):
    uid, cid = await _conv(name)
    rid = await _run(cid, "苹果股价多少", [
        ([use("s1", "get_stock", {"symbol": "苹果"})], [res("s1", "[TOOL_ERROR] 没有 苹果 的行情", True)]),
        ([use("s2", "get_stock", {"symbol": "AAPL"})], [res("s2", "AAPL 231.5")]), ([text("231.5")], None)])
    mid = await db.pool().fetchval("SELECT min(id) FROM messages WHERE run_id = $1 AND role = 'assistant'", rid)
    return uid, cid, rid, mid


def test_select_by_vector_keyword_and_status():
    async def body():
        ms.set_embedder(Emb())
        uid, _, rid, mid = await _source()
        a = await _lesson(uid, rid, mid)
        await _lesson(uid, rid, mid, status="rejected", behavior="被拒的")
        await _lesson(uid, rid, mid, status="candidate", behavior="还没判的")
        assert [x["id"] for x in await lt.select(uid, "特斯拉股价现在多少")] == [a]   # 向量这一路；只要 active
        assert await lt.select(uid, "北京天气怎么样") == []                          # 无关的不注入
        bob, *_ = await _source("bob")
        assert await lt.select(bob, "特斯拉股价现在多少") == []                       # 别人的经验不给
        ms.set_embedder(Broken())                                                    # 向量挂了：退回关键词
        assert [x["id"] for x in await lt.select(uid, "用中文名问美股的股价")] == [a]   # 命中「美股」「股价」等
        assert await lt.select(uid, "股价") == []                                    # 只命中 1 个词不够
    with_fresh_db(body)


def test_select_caps_count_and_tokens():
    async def body():
        ms.set_embedder(Emb())
        uid, _, rid, mid = await _source()
        ids = [await _lesson(uid, rid, mid, behavior=f"第 {i} 种做法：" + "说明" * 30) for i in range(5)]
        got = await lt.select(uid, "股价")
        assert 1 <= len(got) <= lt.MAX_LESSONS and {x["id"] for x in got} <= set(ids)
        from app import budget
        assert budget.estimate_text(lt.render(got)) <= lt.MAX_TOKENS
        failed = await _lesson(uid, rid, mid, outcome="failed", behavior="短")
        assert "（来自失败的 run）" in lt.line(await ls.get(uid, failed))
    with_fresh_db(body)


def test_tail_injection_keeps_prefix_and_records_uses(monkeypatch):
    async def body():
        monkeypatch.setenv("LESSONS", "1")
        ms.set_embedder(Emb())
        uid, cid, rid, mid = await _source()
        lid = await _lesson(uid, rid, mid)
        prefixes, tails = [], []

        async def model(**req):
            prefixes.append((req["system"], req["messages"][0]["content"], [t["name"] for t in req["tools"]]))
            tails.append(req["messages"][-1]["content"][-1]["text"])
            yield {"type": "final", "content": [text("好")], "stop_reason": "end_turn"}
        r1 = await db.start_run(cid, [text("特斯拉股价多少")])
        await _run_and_collect(cid, r1, model)
        r2 = await db.start_run(cid, [text("北京天气怎么样")])
        await _run_and_collect(cid, r2, model)
        assert f"[L{lid}]" in tails[0] and "<lessons>" in tails[0] and "<lessons>" not in tails[1]
        assert prefixes[0] == prefixes[1]                    # system、层 2、tools 两次逐字一样：缓存保住
        assert (await ls.get(uid, lid))["uses"] == 1
        assert await db.pool().fetchval("SELECT violated FROM lesson_uses WHERE run_id = $1", r1) is False
    with_fresh_db(body)


def test_after_run_counts_violation():
    async def body():
        uid, cid, rid, mid = await _source()
        lid = await _lesson(uid, rid, mid)
        r = await _run(cid, "特斯拉股价", [
            ([use("x1", "get_stock", {"symbol": "特斯拉"})], [res("x1", "[TOOL_ERROR] 没有", True)]),
            ([text("查不到")], None)])
        await ls.record_uses(r, [lid])
        import os
        os.environ["LESSONS"] = "1"
        try:
            assert await lt.after_run(r) == []                                      # 违反 1 次还不隔离
        finally:
            os.environ["LESSONS"] = "0"
        got = await ls.get(uid, lid)
        assert got["violations"] == 1 and got["status"] == "active"
    with_fresh_db(body)


def test_memory_open_drills_down_by_layers():
    async def body():
        ms.set_embedder(Emb())
        uid, _, rid, mid = await _source()
        lid = await _lesson(uid, rid, mid)
        bob, *_ = await _source("bob")
        open_tool = tools._REGISTRY["memory_open"].handler
        tools.current.set(tools.ToolContext(user_id=uid, conversation_id=uid))
        out = await open_tool({"lesson": lid})
        for part in ("前提：", "原因摘要（提炼时写的推测", "源 run 的轨迹（过去的数据", "失败：没有 苹果 的行情",
                     f"消息#{mid}/s1", "收益 1、回归 0"):
            assert part in out, part
        assert "AAPL 231.5" not in out                                             # 结果正文不给
        assert "请给 id" in await open_tool({})
        tools.current.set(tools.ToolContext(user_id=bob, conversation_id=bob))
        assert await open_tool({"lesson": lid}) == f"没有编号为 L{lid} 的经验。"      # 别人的看不到
        copy = await db.pool().fetchval(                                            # L1 那样复制给检验用户
            "INSERT INTO memory_lessons (user_id, precondition, behavior, expected, verify, tools, rationale, outcome, "
            "source_run, status) SELECT $1, precondition, behavior, expected, verify, tools, rationale, outcome, "
            "source_run, 'active' FROM memory_lessons WHERE id = $2 RETURNING id", bob, lid)
        got = await open_tool({"lesson": copy})
        assert "前提：" in got and "源 run 的轨迹" not in got and "苹果" not in got  # 原主人的轨迹不给
    with_fresh_db(body)


class Angle:
    """按文字给定余弦：前提文字里带「航班」的经验和问题的相似度可控。"""
    TABLE = {"航班经验": 0.60, "报销经验": 0.57, "天气经验": 0.45}

    async def embed(self, texts):
        import math
        out = []
        for t in texts:
            c = next((v for k, v in self.TABLE.items() if k in t), 1.0)  # 问题本身 = 基准方向
            out.append([c, math.sqrt(max(0.0, 1 - c * c))] + [0.0] * 1022)
        return out


def test_select_relative_margin():
    async def body():
        ms.set_embedder(Angle())
        uid, _, rid, mid = await _source()
        a = await _lesson(uid, rid, mid, precondition="航班经验", behavior="甲")
        b = await _lesson(uid, rid, mid, precondition="报销经验", behavior="乙")
        await _lesson(uid, rid, mid, precondition="天气经验", behavior="丙")   # 过了 0.4，但离最高分 0.15
        got = [x["id"] for x in await lt.select(uid, "随便问一句")]
        assert got == [a, b]                                                   # 只留离最高分 0.05 以内的
    with_fresh_db(body)
