"""P2 第 5 步：混合检索。两字一组的关键词、向量门槛、RRF、状态和用户过滤在 SQL 里、一路挂了照样出结果。"""

import math

import pytest

from app import db, memory_search as msr, memory_store as ms, tools
from test_db import with_fresh_db

# 假的「语义」：同一组同义词落在同一个维度上，余弦相似度 = 共享维度的占比
GROUPS = [("主题", "深色", "浅色", "夜间", "暗黑", "配色"), ("过敏", "花生", "坚果"), ("车牌", "汽车"),
          ("React", "Vue", "前端")]


class FakeEmbedder:
    fail = False

    async def embed(self, texts):
        if self.fail:
            raise RuntimeError("embedding down")
        out = []
        for t in texts:
            v = [1.0 if any(w in t for w in g) else 0.0 for g in GROUPS] + [0.0] * (1024 - len(GROUPS))
            v[-1] = 0.3  # 每个文本都带一点公共分量，避免零向量
            n = math.sqrt(sum(x * x for x in v))
            out.append([x / n for x in v])
        return out


def test_terms():
    assert msr.terms("我喜欢什么主题") == ["喜欢", "主题"]           # 含虚字的组丢掉
    assert msr.terms("用户的车牌号") == ["车牌", "牌号"]              # 「用户」丢掉：每条记忆都有
    assert msr.terms("React 还是 Vue？预算 37 万") == ["react", "vue", "预算", "37", "万"]  # 「还是」含虚字「是」
    assert msr.terms("猫") == ["猫"] and msr.terms("a 的") == []


async def _setup():
    emb = FakeEmbedder()
    ms.set_embedder(emb)
    alice, bob = await db.create_user("alice", "tok-a"), await db.create_user("bob", "tok-b")

    async def put(uid, content, source="user_stated", supersedes=()):
        r = await ms.add(uid, kind="fact", content=content, source=source, evidence=[], actor="agent",
                         supersedes=list(supersedes))
        await ms.index(r["id"], content)
        return r["id"]
    return emb, alice, bob, put


def test_hybrid_search_paths_and_filters():
    async def body():
        emb, alice, bob, put = await _setup()
        dark = await put(alice, "用户喜欢深色的主题")
        nut = await put(alice, "用户对花生过敏")
        old = await put(alice, "用户前端用 Vue")
        new = await put(alice, "用户前端改用 React", supersedes=[old])
        await put(bob, "用户喜欢浅色主题")

        async def ids(q):
            return [h["id"] for h in (await msr.search(alice, q))[0]]
        assert await ids("我喜欢什么主题") == [dark]                    # 两路都命中
        assert await ids("夜间模式") == [dark]                           # 字面一个都不沾，靠向量
        assert await ids("我的车牌号") == []                             # 无关：低于门槛，关键词也没命中
        assert await ids("前端框架") == [new]                            # 被取代的旧条目两路都不返回
        assert nut not in await ids("主题")
        hit = (await msr.search(alice, "深色主题"))[0][0]
        assert hit["ranks"] == {"keyword": 1, "vector": 1}
        # RRF：两路都靠前的排第一；只在一路出现的排后面
        both = await put(alice, "用户的主题配色是暗黑")
        res = [h["id"] for h in (await msr.search(alice, "暗黑主题"))[0]]
        assert res[0] == both and dark in res
        assert await ids("50%") == []    # 分词只留汉字、字母、数字，% 到不了 SQL；SQL 里的转义是第二道防线
        # 向量挂了：关键词照样出结果，标出降级
        emb.fail = True
        hits, flags = await msr.search(alice, "我喜欢什么主题")
        assert flags == ["vector_failed"] and dark in [h["id"] for h in hits]
        tea = await put(alice, "用户喜欢喝茶")                            # 更新、但只命中「喜欢」一组
        assert [h["id"] for h in (await msr.search(alice, "喜欢深色主题"))[0]][:2] == [dark, tea]  # 命中组数多的在前
        assert (await msr.search(alice, "夜间模式"))[0] == []
    with_fresh_db(body)


def test_search_tool_output():
    async def body():
        emb, alice, bob, put = await _setup()
        a = await put(alice, "用户对花生过敏")
        b = await put(alice, "用户可能喜欢坚果零食", source="agent_inferred")
        tools.current.set(tools.ToolContext(user_id=alice))
        out = await msr._search_tool({"query": "花生过敏"})
        assert f"[#{a}] 用户对花生过敏（" in out and f"[#{b}]" in out and "推断" in out
        emb.fail = True
        assert "没搜到不代表没有" in await msr._search_tool({"query": "车牌"})
        tools.current.set(tools.ToolContext(user_id=bob))
        assert "没有找到" in await msr._search_tool({"query": "花生"})       # 别人的搜不到
        tools.current.set(tools.ToolContext())
        with pytest.raises(LookupError):
            await msr._search_tool({"query": "花生"})
    with_fresh_db(body)
