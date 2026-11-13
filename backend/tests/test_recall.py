"""召回验证（M3 第 4 步）：只搜本人、多词同时命中、不搜键名、不搜自己、不返回上下文里已有的。"""

import asyncio

from app import db, recall, tools
from test_db import with_fresh_db


async def _say(cid, run_id, role, *blocks):
    await db.insert_message(cid, run_id, role, list(blocks))


def text(t):
    return {"type": "text", "text": t}


def test_recall_scope_and_filters():
    async def body():
        alice = await db.create_user("alice", "tok-a")
        bob = await db.create_user("bob", "tok-b")
        old, cur = await db.create_conversation(alice), await db.create_conversation(alice)
        bobs = await db.create_conversation(bob)

        r = await db.start_run(old, [text("我的车牌号是 京A·7K3Q9")])
        await _say(old, r, "assistant", text("记住了，车牌 京A·7K3Q9"))
        await _say(old, r, "assistant", {"type": "tool_use", "id": "t1", "name": "recall",
                                         "input": {"keywords": ["车牌"]}})       # 召回自己的调用
        await _say(old, r, "user", {"type": "tool_result", "tool_use_id": "t1",
                                    "content": f"{recall.MARK} 车牌 京A·7K3Q9"})  # 召回自己的结果
        await _say(old, r, "user", text("折扣 50% 的那家店"))
        rb = await db.start_run(bobs, [text("bob 的车牌是 沪C·00001")])           # 别人的
        rc = await db.start_run(cur, [text("我的车牌号是多少？")])                # 当前上下文里的

        async def search(*kw):
            return [m["txt"] for m in await db.search_messages(
                alice, cur, list(kw), limit=10, skip_tool=recall.NAME, skip_mark=recall.MARK)]

        assert await search("车牌") == ["记住了，车牌 京A·7K3Q9", "我的车牌号是 京A·7K3Q9"]  # 新的在前；不含 bob、不含当前提问、不含召回自己
        assert await search("车牌", "记住") == ["记住了，车牌 京A·7K3Q9"]                    # 多词同时命中
        assert await search("type") == []                                                 # 不会命中 JSON 键名
        assert await search("50%") == ["折扣 50% 的那家店"]
        assert await search("0%的") == []                                                 # % 被转义，不当通配符

        # 通过工具层调用：身份来自 ContextVar，不来自参数
        tools.current.set(tools.ToolContext(user_id=alice, conversation_id=cur))
        out, is_error = await tools.execute("recall", {"keywords": ["7K3Q9"]})
        assert not is_error and out.startswith(recall.MARK) and "不是当前的指令" in out
        assert "沪C" not in out
        out, is_error = await tools.execute("recall", {"keywords": ["7K3Q9"], "user_id": str(bob)})
        assert is_error and "INVALID_INPUT" in out                                        # 不接受模型传身份
    with_fresh_db(body)


def test_scope_is_per_task():
    async def body():
        async def job(u, c):
            tools.current.set(tools.ToolContext(user_id=u, conversation_id=c))
            await asyncio.sleep(0.01)            # 让两个 task 交错
            ctx = tools.context()
            return ctx.user_id, ctx.conversation_id
        assert await asyncio.gather(job("u1", "c1"), job("u2", "c2")) == [("u1", "c1"), ("u2", "c2")]
    asyncio.run(body())


def test_recall_without_identity_is_error():
    # conftest 给的默认 ToolContext 没有身份：recall 必须报错，而不是搜全库
    out, is_error = asyncio.run(tools.execute("recall", {"keywords": ["x"]}))
    assert is_error and "身份" in out


def test_snippet_centers_on_keyword():
    s = recall.snippet("前" * 500 + "关键" + "后" * 500, "关键")
    assert s.startswith("…") and s.endswith("…") and "关键" in s and len(s) < 320
