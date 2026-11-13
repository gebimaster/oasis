"""M7 端到端验证（真实模型）：.venv\\Scripts\\python scripts/e2e_m7.py

脚本自己起 uvicorn（端口 8001、独立库 oasis_e2e），不碰 8000。
场景：
1. 派 3 个天气子任务：并行执行（开始时间重叠）、全部完成、父汇总
2. 子任务在执行外部操作（MCP slow_task 15 秒，high 风险，用一条 allow 规则放行）时用户点停止：
   父子都 cancelled，外部操作记结果未知，父历史补了真结果
3. 同样的子任务执行中强杀进程：重启后子 run 中断、外部操作结果未知、不自动重试；追问时模型不重做
注：第一次用「子任务转账」，模型看不到 M5 的规则，认定转账一定要确认，拒绝派给子任务、自己去转了；
   所以改用不涉及钱的耗时外部操作
"""

import asyncio
import os
import sys

import asyncpg
import httpx

sys.path.insert(0, os.path.dirname(__file__))
from e2e_m2 import BACKEND, BASE, E2E_URL, H, check, sse  # noqa: E402
from e2e_m5 import start  # noqa: E402

WEATHER = "请用 spawn_subagents 并行派 3 个子任务：分别查北京、上海、深圳的天气，各给一句穿衣建议。最后汇总成表格。"
SLOW = ("请用 spawn_subagents 派 2 个子任务并行去做：子任务 A、B 各自调用一次 mcp__demo__slow_task(seconds=15)，"
        "模拟一个耗时的外部同步操作，完成后汇报。")


async def executions(pg) -> int:
    return await pg.fetchval("SELECT count(*) FROM tool_executions")


async def send(client, cid, text) -> str:
    return (await client.post(f"{BASE}/conversations/{cid}/messages", headers=H, json={"text": text})).json()["run_id"]


async def new_conv(client) -> str:
    return (await client.post(f"{BASE}/conversations", headers=H)).json()["conversation_id"]


async def children(pg, parent):
    return await pg.fetch("SELECT title, status, stop_reason, created_at, finished_at, usage FROM runs "
                          "WHERE parent_run_id = $1 ORDER BY child_index", parent)


async def tool_result_of_spawn(client, cid) -> str:
    msgs = (await client.get(f"{BASE}/conversations/{cid}/messages", headers=H)).json()["messages"]
    return next(b["content"] for m in msgs if m["role"] == "user" and isinstance(m["content"], list)
                for b in m["content"] if b.get("type") == "tool_result" and "[SUBAGENT_RESULTS]" in str(b["content"]))


async def wait_executing(pg, n: int):
    for _ in range(600):  # 最多 60 秒：模型要先派任务、子任务再调 transfer
        if await pg.fetchval("SELECT count(*) FROM tool_executions WHERE status = 'executing'") >= n:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("子任务没有进入转账执行")


async def scenario_parallel(client, pg):
    print("场景 1：派 3 个子任务并行")
    cid = await new_conv(client)
    run_id = await send(client, cid, WEATHER)
    evs = await sse(client, run_id)
    check(evs[-1]["status"] == "completed", f"父 run 结束：{evs[-1]['stop_reason']}")
    kids = await children(pg, run_id)
    check(len(kids) == 3 and all(k["status"] == "completed" for k in kids), "3 个子 run 都完成")
    check(max(k["created_at"] for k in kids) < min(k["finished_at"] for k in kids), "开始时间互相重叠：真并行")
    for k in kids:
        secs = (k["finished_at"] - k["created_at"]).total_seconds()
        print(f"  {k['title']}：{secs:.1f}s usage={k['usage']}")
    ups = [e for e in evs if e["type"] == "subagent_update"]
    check({e["status"] for e in ups} >= {"queued", "running", "completed"}, "父流里有胶囊状态事件")
    check("完成 3，未完成 0" in await tool_result_of_spawn(client, cid), "父拿到的汇总：完成 3")


async def scenario_stop(client, pg):
    print("场景 2：子任务执行外部操作中，用户点停止")
    await pg.execute("INSERT INTO tool_rules (user_id, tool_pattern, conditions, effect) "
                     "VALUES (NULL, 'mcp__demo__slow_task', '[]', 'allow')")  # high 风险，没规则会被子任务拒绝
    cid = await new_conv(client)
    run_id = await send(client, cid, SLOW)
    await wait_executing(pg, 1)
    await client.post(f"{BASE}/runs/{run_id}/cancel", headers=H)
    evs = await sse(client, run_id)
    check(evs[-1]["status"] == "cancelled", f"父 run：{evs[-1]['status']}/{evs[-1]['stop_reason']}")
    kids = await children(pg, run_id)
    check(kids and all((k["status"], k["stop_reason"]) == ("cancelled", "parent_cancel") for k in kids),
          f"子 run 全部 cancelled/parent_cancel（{[(k['status'], k['stop_reason']) for k in kids]}）")
    unknown = await pg.fetchval("SELECT count(*) FROM tool_executions WHERE status = 'outcome_unknown'")
    check(unknown >= 1, f"被打断的外部操作记为结果未知（{unknown} 笔）")
    result = await tool_result_of_spawn(client, cid)
    check("已取消（用户停止）" in result and "结果未知" in result, "父历史补了真 tool_result（已取消 + 结果未知）")


async def scenario_kill(client, pg, log, proc):
    print("场景 3：子任务执行外部操作中强杀进程")
    await pg.execute("UPDATE tool_executions SET status = 'succeeded' WHERE status = 'outcome_unknown'")  # 只看本场景的
    cid = await new_conv(client)
    run_id = await send(client, cid, SLOW)
    await wait_executing(pg, 1)
    proc.kill()  # TerminateProcess：不走 lifespan 收尾
    proc.wait()
    print("  已强杀，重启（这次不睡）")
    proc = start(log)
    run = await pg.fetchrow("SELECT status, stop_reason FROM runs WHERE id = $1", run_id)
    check(tuple(run) == ("failed", "interrupted"), f"父 run：{tuple(run)}")
    kids = await children(pg, run_id)
    check(kids and all(k["status"] in ("failed", "completed") for k in kids)
          and any(k["stop_reason"] == "interrupted" for k in kids), f"子 run 已对账：{[k['stop_reason'] for k in kids]}")
    check(await pg.fetchval("SELECT count(*) FROM tool_executions WHERE status = 'outcome_unknown'") >= 1,
          "执行中的外部操作 → 结果未知")
    result = await tool_result_of_spawn(client, cid)
    check("已中断（服务重启）" in result and "结果未知" in result, "父历史补了真 tool_result（已中断 + 结果未知）")
    before = await executions(pg)
    await asyncio.sleep(3)
    check(await executions(pg) == before, "等 3 秒没有新的执行：没有自动重试")
    evs = await sse(client, await send(client, cid, "刚才那两个外部操作现在什么情况？"))
    msgs = (await client.get(f"{BASE}/conversations/{cid}/messages", headers=H)).json()["messages"]
    print("  模型回答：", "".join(b.get("text", "") for b in msgs[-1]["content"])[:160])
    check(evs[-1]["status"] in ("completed", "failed") and await executions(pg) == before, "追问后也没有重做")
    return proc


async def main():
    admin = await asyncpg.connect(E2E_URL.rsplit("/", 1)[0] + "/oasis")
    await admin.execute("DROP DATABASE IF EXISTS oasis_e2e WITH (FORCE)")
    await admin.execute("CREATE DATABASE oasis_e2e")
    await admin.close()
    log = open(BACKEND / "scripts" / "e2e_m7_server.log", "w", encoding="utf-8")
    proc = start(log)
    pg = await asyncpg.connect(E2E_URL)
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            await scenario_parallel(client, pg)
            await scenario_stop(client, pg)
            proc = await scenario_kill(client, pg, log, proc)
    finally:
        await pg.close()
        proc.kill()
        log.close()
    print("全部通过")


if __name__ == "__main__":
    asyncio.run(main())
