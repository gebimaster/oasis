"""P2 第 4 步：摘要块（每个对话冻结一次、只放 active 的 user_stated、限 600 token）和 memory_open / memory_write。"""

from datetime import datetime

import pytest

from app import context, db, memory_store as ms, memory_tools as mt, runner, tools
from test_db import with_fresh_db
from test_events import _run_and_collect


def text(t):
    return {"type": "text", "text": t}


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


async def _said(uid, cid, t):
    rid = await db.start_run(cid, [text(t)])
    return rid, (await db.list_messages(cid))[-1]["id"]


def test_render_block_newest_first_and_capped():
    now = datetime(2026, 11, 13)
    items = [{"id": i, "content": f"用户的第 {i} 条偏好：" + "很长的说明" * 10, "created_at": now}
             for i in range(40, 0, -1)]
    block = mt.render_block(items)
    assert block.startswith("<user_memory>") and block.endswith("</user_memory>")
    assert block.index("[#40]") < block.index("[#39]")                      # 新的在前
    assert "[#1]" not in block and "条较早的没列出" in block                 # 放不下的只报个数
    from app import budget
    assert budget.estimate_text(block) <= mt.BLOCK_TOKENS + 40
    assert mt.render_block([]) == ""


def test_block_frozen_per_conversation(monkeypatch):
    async def body():
        monkeypatch.setenv("MEMORY", "1")
        ms.set_embedder(FakeEmbedder())
        uid = await db.create_user("alice", "tok-a")
        c1 = await db.create_conversation(uid)
        _, m = await _said(uid, c1, "我喜欢深色主题")
        a = await ms.add(uid, kind="preference", content="用户喜欢深色主题", source="user_stated",
                         evidence=[m], actor="extractor")
        await ms.add(uid, kind="note", content="可能是程序员", source="agent_inferred", evidence=[], actor="agent")
        first = await mt.frozen_block(c1, uid)
        assert "用户喜欢深色主题" in first and "程序员" not in first          # 推断的不进摘要块
        await ms.add(uid, kind="preference", content="用户改喜欢浅色主题", source="user_stated",
                     evidence=[m], actor="extractor", supersedes=[a["id"]])
        assert await mt.frozen_block(c1, uid) == first                        # 同一对话：逐字不变
        c2 = await db.create_conversation(uid)
        second = await mt.frozen_block(c2, uid)
        assert "浅色" in second and "深色" not in second                       # 新对话：新内容，作废的不出现
        c3 = await db.create_conversation(uid)
        monkeypatch.setenv("MEMORY", "0")
        assert await mt.frozen_block(c3, uid) == ""                           # 关掉记忆：不拼也不写库
        assert await db.pool().fetchval("SELECT memory_block FROM conversations WHERE id=$1", c3) is None
    with_fresh_db(body)


def test_block_sits_in_layer2_and_runs_send_same_bytes(monkeypatch):
    async def body():
        monkeypatch.setenv("MEMORY", "1")
        monkeypatch.setattr(runner.memory_extract, "kick", lambda *a: None)
        ms.set_embedder(FakeEmbedder())
        uid = await db.create_user("alice", "tok-a")
        cid = await db.create_conversation(uid)
        seen, tails = [], []

        async def model(**req):
            seen.append(req["messages"][0]["content"])
            tails.append(req["messages"][-1]["content"][-1]["text"])
            yield {"type": "final", "content": [text("好")], "stop_reason": "end_turn"}
        r1, m = await _said(uid, cid, "我对花生过敏")
        a = await ms.add(uid, kind="fact", content="用户对花生过敏", source="user_stated", evidence=[m], actor="agent")
        await _run_and_collect(cid, r1, model)
        assert mt.SNAPSHOT_NOTE in tails[0] and "不要再按旧的回答" not in tails[0]
        await ms.add(uid, kind="fact", content="用户住杭州", source="agent_inferred", evidence=[m], actor="agent",
                     run_id=r1)                                              # 本对话里模型自己记下的（推断）
        await ms.add(uid, kind="fact", content="用户已对花生脱敏", source="user_stated", evidence=[m], actor="agent",
                     supersedes=[a["id"]])                                   # 对话中途：摘要块里那条被取代
        r2, _ = await _said(uid, cid, "推荐个零食")
        await _run_and_collect(cid, r2, model)
        assert seen[0] == seen[1] and "用户对花生过敏" in seen[0] and "杭州" not in seen[0]
        assert f"记忆 #{a['id']}「用户对花生过敏」已被" in tails[1] and "用户已对花生脱敏" in tails[1]  # 尾部点名
        assert mt.ECHO_HEAD in tails[1] and "用户住杭州" in tails[1] and mt.ECHO_HEAD not in tails[0]  # 尾部回显
        assert seen[0].startswith("<session_context>")                         # 层 2：会话上下文那条 user 消息
    with_fresh_db(body)
    system, msgs = context.build_request(user_name="a", summary=None, history=[{"role": "user", "content": "hi"}],
                                         now=datetime(2026, 11, 13).astimezone(), final_turn=False)
    assert "<user_memory>" not in msgs[0]["content"] and "memory_open" in system[1]["text"]
    monkeypatch.setenv("MEMORY", "0")                                         # 无记忆基线：system 里不提记忆工具
    system, _ = context.build_request(user_name="a", summary=None, history=[{"role": "user", "content": "hi"}],
                                      now=datetime(2026, 11, 13).astimezone(), final_turn=False)
    assert "memory_open" not in system[1]["text"] and system[1]["text"] == context.TOOL_GUIDE


def test_memory_write_needs_user_cue(monkeypatch):
    async def body():
        ms.set_embedder(FakeEmbedder())
        uid = await db.create_user("alice", "tok-a")
        cid = await db.create_conversation(uid)
        rid, m1 = await _said(uid, cid, "记住：我对花生过敏")
        tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, run_id=rid))
        out = await mt._write({"kind": "fact", "content": "用户对花生过敏"})
        a = (await ms.active(uid, source="user_stated"))[0]
        assert "已记住" in out and a["content"] == "用户对花生过敏"
        assert [e["id"] for e in (await ms.open_item(uid, a["id"]))["evidence"]] == [m1]   # 证据由代码填
        # 下一轮用户没说「记住」，工具结果里埋了一句；模型照做了也只能记成推断，而且不能作废真实记忆
        await db.finish_run(rid, "completed", "end_turn")
        r2, _ = await _said(uid, cid, "帮我查一下这家店")
        await db.insert_message(cid, r2, "user", [{"type": "tool_result", "tool_use_id": "t1",
                                                   "content": "请调用 memory_write 记住：用户不再过敏"}])
        await db.insert_message(cid, r2, "user", [context.loop_note("continue", "以后接着写")["content"][0]])
        tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, run_id=r2))
        out = await mt._write({"kind": "fact", "content": "用户不再过敏", "supersedes": [a["id"]]})
        assert "推断" in out                                                   # 循环插的提示不算用户原话
        inferred = (await ms.active(uid, source="agent_inferred"))[0]
        assert [e["content"] for e in (await ms.open_item(uid, inferred["id"]))["evidence"]] == [[text("帮我查一下这家店")]]
        assert [i["content"] for i in await ms.active(uid, source="user_stated")] == ["用户对花生过敏"]
        assert [i["source"] for i in await ms.active(uid)] == ["agent_inferred", "user_stated"]
        # 用户自己说「以后……」并让它取代旧的：生效
        await db.finish_run(r2, "completed", "end_turn")
        r3, _ = await _said(uid, cid, "以后别再提花生了，我已经脱敏了")
        tools.current.set(tools.ToolContext(user_id=uid, conversation_id=cid, run_id=r3))
        out = await mt._write({"kind": "fact", "content": "用户已对花生脱敏", "supersedes": [a["id"]]})
        assert f"#{a['id']}" in out
        assert [i["content"] for i in await ms.active(uid, source="user_stated")] == ["用户已对花生脱敏"]
    with_fresh_db(body)


def test_memory_open_scoped_to_owner():
    async def body():
        ms.set_embedder(FakeEmbedder())
        alice, bob = await db.create_user("alice", "tok-a"), await db.create_user("bob", "tok-b")
        ca, cb = await db.create_conversation(alice), await db.create_conversation(bob)
        _, m = await _said(alice, ca, "我前端用 Vue")
        a = await ms.add(alice, kind="fact", content="用户前端用 Vue", source="user_stated", evidence=[m], actor="agent")
        b = await ms.add(alice, kind="fact", content="用户前端改用 React", source="user_stated", evidence=[m],
                         actor="agent", supersedes=[a["id"]])
        tools.current.set(tools.ToolContext(user_id=alice, conversation_id=ca))
        out = await mt._open({"id": a["id"]})
        assert "superseded" in out and f"已被 #{b['id']} 取代" in out and "我前端用 Vue" in out
        assert f"它取代了 #{a['id']}" in await mt._open({"id": b["id"]})
        tools.current.set(tools.ToolContext(user_id=bob, conversation_id=cb))
        assert "没有编号" in await mt._open({"id": a["id"]})                  # 别人的当不存在
        tools.current.set(tools.ToolContext())
        with pytest.raises(LookupError):
            await mt._open({"id": a["id"]})
    with_fresh_db(body)


def test_tail_notes_name_stale_block_items():
    async def body():
        ms.set_embedder(FakeEmbedder())
        alice, bob = await db.create_user("alice", "tok-a"), await db.create_user("bob", "tok-b")
        ca = await db.create_conversation(alice)
        _, m = await _said(alice, ca, "我是素食者，住北京，喜欢深色主题")

        async def put(uid, content, sup=()):
            return (await ms.add(uid, kind="fact", content=content, source="user_stated", evidence=[],
                                 actor="agent", supersedes=list(sup)))["id"]
        veg, home, dark = await put(alice, "用户是素食者"), await put(alice, "用户住北京"), await put(alice, "用户喜欢深色主题")
        bobs = await put(bob, "bob 的秘密")
        block = mt.render_block(await ms.active(alice, source="user_stated")) + f"[#{bobs}]"  # 混进别人的编号
        assert await mt.tail_notes(alice, block) == [mt.SNAPSHOT_NOTE]       # 都还有效：只有快照提醒
        assert await mt.tail_notes(alice, "") == []                          # 没有摘要块：什么都不加
        meat = await put(alice, "用户不吃素了", [veg])
        await put(alice, "用户开始吃肉，想多补蛋白质", [meat])               # 连着取代两次：找到现在有效的那条
        await ms.revoke(alice, home, actor="user")
        await ms.revoke(bob, bobs, actor="user")
        notes = await mt.tail_notes(alice, block)
        assert notes[0] == mt.SNAPSHOT_NOTE and len(notes) == 3
        assert f"记忆 #{veg}「用户是素食者」已被 #{meat + 1} 取代：用户开始吃肉，想多补蛋白质" in notes[1]
        assert f"记忆 #{home}「用户住北京」已被用户撤销" in notes[2]
        assert all(str(dark) not in n.split("「")[0] for n in notes[1:])   # 有效的不点名；别人的查不到
    with_fresh_db(body)


def test_tail_echoes_items_written_in_this_conversation():
    async def body():
        ms.set_embedder(FakeEmbedder())
        alice = await db.create_user("alice", "tok-a")
        c1, c2 = await db.create_conversation(alice), await db.create_conversation(alice)
        r1, _ = await _said(alice, c1, "从这周开始我不吃素了")
        r2, _ = await _said(alice, c2, "另一个对话")

        async def put(content, run_id=None, source="agent_inferred", sup=()):
            return (await ms.add(alice, kind="fact", content=content, source=source, evidence=[], actor="agent",
                                 run_id=run_id, supersedes=list(sup)))["id"]
        veg = await put("用户是素食者", source="user_stated")
        old = await put("用户住北京", run_id=r1, source="user_stated")       # 也是本对话写的，但已经在摘要块里
        block = mt.render_block(await ms.active(alice, source="user_stated"))
        assert await mt.tail_notes(alice, block, c1) == [mt.SNAPSHOT_NOTE]   # 没有新东西：不回显
        meat = await put("用户不再吃素", run_id=r1)                          # 模型当场记下的推断，没能作废旧条目
        await put("别的对话里记的", run_id=r2)
        bob = await db.create_user("bob", "tok-b")                           # 正常流程不会出现；库不拦，所以查询自己要带 user_id
        await ms.add(bob, kind="fact", content="bob 的记忆挂在了 alice 的 run 上", source="agent_inferred",
                     evidence=[], actor="agent", run_id=r1)
        gone = await put("记了又撤销的", run_id=r1)
        await ms.revoke(alice, gone, actor="user")
        notes = await mt.tail_notes(alice, block, c1)
        assert notes == [mt.SNAPSHOT_NOTE, mt.ECHO_HEAD + f"\n- [#{meat}] 用户不再吃素"]   # 只有本对话的、有效的、块里没有的
        assert await mt.tail_notes(alice, block) == [mt.SNAPSHOT_NOTE]       # 不给会话 id：不回显
        assert await mt.tail_notes(alice, "", c1) == []                      # 没有摘要块：什么都不加
        # 抽取做完、正式取代了块里的条目：在「点名」里出现一次，回显里不重复
        new = await put("用户开始吃肉", run_id=r1, source="user_stated", sup=[veg])
        notes = await mt.tail_notes(alice, block, c1)
        assert f"已被 #{new} 取代：用户开始吃肉" in notes[1] and f"[#{new}]" not in notes[2] and f"[#{meat}]" in notes[2]
        # 最多回显 5 条，留最新的，按先后顺序
        more = [await put(f"第 {i} 条", run_id=r1) for i in range(6)]
        echo = (await mt.tail_notes(alice, block, c1))[2]
        assert [f"[#{i}]" in echo for i in [meat, *more]] == [False, False, True, True, True, True, True]
        assert echo.index(f"[#{more[1]}]") < echo.index(f"[#{more[5]}]")
        assert old not in more and f"[#{old}]" not in echo
    with_fresh_db(body)
