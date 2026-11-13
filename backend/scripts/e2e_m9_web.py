"""M9 联网工具端到端（真实模型 + 真实 DeepSeek Search/Fetch）：.venv\\Scripts\\python scripts/e2e_m9_web.py

脚本自己起 uvicorn（端口 8001、独立库 oasis_e2e）。场景：
1. 搜索后抓第一条结果：搜到过的链接 fetch 直接执行，不等确认
2. 用户自己贴的链接：fetch 直接执行
3. 模型自己拼的链接（没搜过、用户也没给）：挂起等用户确认（或者模型先搜再抓——两种都对，都要验证）
"""

import asyncio
import os
import sys
from uuid import UUID

import asyncpg
import httpx

sys.path.insert(0, os.path.dirname(__file__))
from e2e_m2 import BACKEND, E2E_URL, check, sse  # noqa: E402
from e2e_m5 import start  # noqa: E402
from e2e_m7 import new_conv, send  # noqa: E402

sys.path.insert(0, str(BACKEND))
from app import db, web_tools  # noqa: E402


async def calls(pg, run_id) -> list[tuple[str, str]]:
    """本 run 里 fetch 的 URL 和执行状态（fetch 静态风险是 HIGH，每次都有执行记录）。"""
    rows = await pg.fetch("SELECT tool_name, args, status FROM tool_executions WHERE run_id = $1", run_id)
    return [(r["args"]["url"], r["status"]) for r in rows if r["tool_name"] == "fetch"]


async def searched_urls(pg, cid) -> set[str]:
    rows = await pg.fetch("SELECT content FROM messages WHERE conversation_id = $1 AND role = 'user'", cid)
    return {u for r in rows for b in r["content"] if b.get("type") == "tool_result"
            and "搜索「" in str(b.get("content")) for u in web_tools.urls_in(str(b["content"]))}


async def main():
    admin = await asyncpg.connect(E2E_URL.rsplit("/", 1)[0] + "/oasis")
    await admin.execute("DROP DATABASE IF EXISTS oasis_e2e WITH (FORCE)")
    await admin.execute("CREATE DATABASE oasis_e2e")
    await admin.close()
    log = open(BACKEND / "scripts" / "e2e_m9_web_server.log", "w", encoding="utf-8")
    proc = start(log)
    pg = await asyncpg.connect(E2E_URL)
    await db._init_conn(pg)  # jsonb 直接收发 dict，和服务端一致
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            print("场景 1：搜索后抓第一条结果")
            cid = await new_conv(client)
            run_id = await send(client, cid, "先用 web_search 搜「CubeSandbox changelog」，再用 fetch 打开第一条结果，"
                                             "告诉我页面里最新的版本号。")
            evs = await sse(client, run_id)
            check(evs[-1]["stop_reason"] == "end_turn", f"run 结束：{evs[-1]['stop_reason']}（没有挂起等确认）")
            got = await calls(pg, UUID(run_id))
            seen = await searched_urls(pg, UUID(cid))
            check(got and all(web_tools.normalize(u) in seen and s == "succeeded" for u, s in got),
                  f"抓的都是搜到过的链接，直接执行：{got}")

            print("场景 2：用户自己贴的链接")
            cid = await new_conv(client)
            run_id = await send(client, cid, "用 fetch 打开 https://docs.cubesandbox.com/zh/changelog/v0.1.0 ，一句话总结。")
            evs = await sse(client, run_id)
            check(evs[-1]["stop_reason"] == "end_turn", f"run 结束：{evs[-1]['stop_reason']}")
            check(await calls(pg, UUID(run_id)), "fetch 直接执行了")

            print("场景 3：模型自己拼的链接")
            cid = await new_conv(client)
            run_id = await send(client, cid, "不要搜索，直接用 fetch 打开 example.org 的首页，告诉我标题。")
            evs = await sse(client, run_id)
            stop = evs[-1]["stop_reason"]
            asks = [e for e in evs if e["type"] == "approval_request"]
            print(f"  stop_reason={stop}，审批卡片：{[a['args'] for e in asks for a in e['approvals']]}")
            check(stop == "awaiting_approval" and asks and asks[0]["approvals"][0]["tool_name"] == "fetch",
                  "没见过的链接：挂起等用户确认，没有执行")
            check(not await calls(pg, UUID(run_id)), "确认前没有任何 fetch 执行记录")
        n = await pg.fetchval("SELECT count(*) FROM messages WHERE usage IS NOT NULL")
        print(f"  共 {n} 次模型调用")
    finally:
        await pg.close()
        proc.kill()
        log.close()
    print("全部通过")


if __name__ == "__main__":
    asyncio.run(main())
