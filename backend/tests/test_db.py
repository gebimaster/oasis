"""数据库层验证：需要先 `docker compose up -d`。每个测试用一个全新的 oasis_test 库。"""

import asyncio

import asyncpg
import pytest

from app import db

ADMIN_URL = "postgresql://oasis:oasis@127.0.0.1:5433/oasis"
TEST_URL = "postgresql://oasis:oasis@127.0.0.1:5433/oasis_test"


def with_fresh_db(test_body):
    """重建测试库 → 连接（自动建表）→ 跑测试 → 关闭。"""
    async def main():
        admin = await asyncpg.connect(ADMIN_URL)
        await admin.execute("DROP DATABASE IF EXISTS oasis_test WITH (FORCE)")
        await admin.execute("CREATE DATABASE oasis_test")
        await admin.close()
        await db.connect(TEST_URL)
        try:
            await test_body()
        finally:
            await db.close()
    asyncio.run(main())


async def _conversation():
    uid = await db.create_user("alice", "token-alice")
    return uid, await db.create_conversation(uid)


def test_messages_roundtrip_in_order():
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "北京天气？"}])
        thinking = {"type": "thinking", "thinking": "", "signature": "sig"}
        await db.insert_message(cid, run_id, "assistant", [thinking, {"type": "tool_use", "id": "t1"}])
        await db.insert_message(cid, run_id, "user", [{"type": "tool_result", "tool_use_id": "t1"}])
        msgs = await db.list_messages(cid)
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[1]["content"][0] == thinking  # JSONB 原样取回（含签名）
    with_fresh_db(body)


def test_one_running_run_per_conversation():
    async def body():
        _, cid = await _conversation()
        run1 = await db.start_run(cid, [{"type": "text", "text": "第一句"}])
        with pytest.raises(db.RunConflict):
            await db.start_run(cid, [{"type": "text", "text": "第二句"}])
        assert len(await db.list_messages(cid)) == 1  # 被拒的那句没落库（同一事务回滚）

        await db.finish_run(run1, "completed", "end_turn")
        await db.start_run(cid, [{"type": "text", "text": "第三句"}])  # 上一个结束后可以再发
    with_fresh_db(body)


def test_concurrent_start_only_one_wins():
    async def body():
        _, cid = await _conversation()
        results = await asyncio.gather(
            *(db.start_run(cid, [{"type": "text", "text": f"并发{i}"}]) for i in range(5)),
            return_exceptions=True)
        assert sum(not isinstance(r, Exception) for r in results) == 1
        assert all(isinstance(r, db.RunConflict) for r in results if isinstance(r, Exception))
    with_fresh_db(body)


def test_recover_and_finish_only_once():
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "hi"}])
        await db.pool().execute("UPDATE runs SET lease_until = now() - interval '1 second'")  # O1：模拟属主崩溃、租约过期
        assert await db.recover_expired_runs() == 1
        await db.finish_run(run_id, "completed", "end_turn")  # 已结束的不会被覆盖
        run = await db.get_run(run_id)
        assert (run["status"], run["stop_reason"]) == ("failed", "interrupted")
    with_fresh_db(body)


def test_recover_saves_partial_assistant_as_interrupted():
    async def body():
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "讲个故事"}])
        await db.append_event(run_id, {"type": "text_delta", "text": "已存"})
        await db.insert_message(cid, run_id, "assistant", [{"type": "text", "text": "已存"}])
        await db.insert_message(cid, run_id, "user", [{"type": "tool_result", "tool_use_id": "t1"}])
        for t in ("从前", "有座山"):  # 崩溃前只推送了、没落库的半截
            await db.append_event(run_id, {"type": "text_delta", "text": t})
        await db.pool().execute("UPDATE runs SET lease_until = now() - interval '1 second'")  # O1：模拟属主崩溃、租约过期
        assert await db.recover_expired_runs() == 1
        last = (await db.list_messages(cid))[-1]
        assert (last["role"], last["status"]) == ("assistant", "interrupted")
        assert last["content"] == [{"type": "text", "text": "从前有座山"}]  # 只拼最后一个 message_saved 之后的
        assert (await db.list_messages(cid))[0]["status"] == "completed"   # 其余默认 completed
        assert (await db.conversation_user(cid))["name"] == "alice"
    with_fresh_db(body)


def test_ownership():
    async def body():
        alice, cid = await _conversation()
        bob = await db.create_user("bob", "token-bob")
        assert await db.user_id_by_token("token-bob") == bob
        assert await db.owns_conversation(alice, cid)
        assert not await db.owns_conversation(bob, cid)
    with_fresh_db(body)



def test_jsonb_encoder_scrubs_lone_surrogates_everywhere():
    """O6 第三轮复核 B（用户选甲）：根因和修复都在真库上钉住。
    根因：不经过编码器、直接把 json.dumps 写出的 \\ud800 转义交给 Postgres，jsonb 拒收。
    修复：db 的 jsonb 编码器统一换成 U+FFFD。消息（tool_use 入参和外部 API 带回来的 tool_result）、
    事件、工具执行记录这三类写库入口都能存进去、取回来就是替换后的内容。"""
    import json
    bs = chr(92)
    raw = json.loads('{"q": "a' + bs + 'ud800b", "' + bs + 'udc00k": [{"y": "' + bs + 'udfff"}], '
                     '"o": {"x": "' + bs + 'ud83d"}}')  # JSON 合法的转义，解析出来是孤立代理（键、list 里的 dict、dict 里的 dict）
    clean = {"q": "a�b", "�k": [{"y": "�"}], "o": {"x": "�"}}

    async def body():
        conn = await asyncpg.connect(TEST_URL)  # 不带 db 的编码器
        try:
            with pytest.raises(asyncpg.PostgresError):
                await conn.fetchval("SELECT $1::text::jsonb", json.dumps(raw))
        finally:
            await conn.close()
        _, cid = await _conversation()
        run_id = await db.start_run(cid, [{"type": "text", "text": "查"}])
        await db.insert_message(cid, run_id, "assistant", [{"type": "tool_use", "id": "t1", "name": "x", "input": raw}])
        await db.insert_message(cid, run_id, "user", [{"type": "tool_result", "tool_use_id": "t1", "content": raw["q"]}])
        saved = await db.list_messages(cid)
        assert saved[-2]["content"][0]["input"] == clean
        assert saved[-1]["content"][0]["content"] == clean["q"]
        ev = await db.append_event(run_id, {"type": "tool_end", "result": raw})
        assert (await db.list_events(run_id, ev["seq"] - 1))[0]["result"] == clean
        ex = await db.start_execution(run_id, "t1", "fetch", raw)
        await db.finish_execution(ex, "succeeded", raw)
        row = await db.pool().fetchrow("SELECT args, result FROM tool_executions WHERE id = $1", ex)
        assert row["args"] == clean and row["result"] == clean
    with_fresh_db(body)
