"""M5 端到端验证（真实模型）：.venv\\Scripts\\python scripts/e2e_m5.py

脚本自己起 uvicorn（端口 8001、独立库 oasis_e2e），不碰 8000。
场景：
1. 两个请求同时批准同一张卡片（真 HTTP 并发）：只有一个生效，账本只多一行
2. 拒绝：模型收到「[未执行] 用户拒绝」并正常收尾
3. 执行中强杀进程（transfer 先睡 8 秒）：重启后是 outcome_unknown、不自动重试；用户核实「未生效」后模型收尾
"""

import asyncio
import os
import subprocess
import sys
import time

import asyncpg
import httpx

sys.path.insert(0, os.path.dirname(__file__))
from e2e_m2 import BACKEND, BASE, E2E_URL, H, PORT, check, sse  # noqa: E402

ASK = "请直接调用 transfer 工具给 alice 转账 {n} 元，不要先问我确认。"


def start(log, delay: str = "0") -> subprocess.Popen:
    env = {**os.environ, "DATABASE_URL": E2E_URL, "TRANSFER_DELAY_SECONDS": delay}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
                            cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(100):
        try:
            httpx.get(BASE + "/", timeout=1)
            return proc
        except httpx.TransportError:
            time.sleep(0.2)
    raise RuntimeError("服务没起来")


async def ledger(pg) -> int:
    return await pg.fetchval("SELECT count(*) FROM ledger")


async def suspended(client, amount: int):
    """发一句话让模型转账（>100 → 要确认），跑到挂起，返回 (cid, tool_use_id)。"""
    cid = (await client.post(f"{BASE}/conversations", headers=H)).json()["conversation_id"]
    run_id = (await client.post(f"{BASE}/conversations/{cid}/messages", headers=H,
                                json={"text": ASK.format(n=amount)})).json()["run_id"]
    evs = await sse(client, run_id)
    check(evs[-1]["stop_reason"] == "awaiting_approval", f"run 以 awaiting_approval 结束（{evs[-1]['stop_reason']}）")
    req = next(e for e in evs if e["type"] == "approval_request")
    a = req["approvals"][0]
    check(a["tool_name"] == "transfer" and a["args"]["amount"] == amount, f"卡片：{a['tool_name']} {a['args']}")
    return cid, a["tool_use_id"]


async def decide(client, cid, tid, action):
    return await client.post(f"{BASE}/conversations/{cid}/approvals/{tid}", headers=H, json={"action": action})


async def approvals(client, cid):
    return {a["tool_use_id"]: a for a in (await client.get(f"{BASE}/conversations/{cid}/approvals", headers=H)).json()["approvals"]}


async def scenario_concurrent(client, pg):
    print("场景 1：两个请求同时批准")
    cid, tid = await suspended(client, 300)
    before = await ledger(pg)
    r1, r2 = await asyncio.gather(decide(client, cid, tid, "approve"), decide(client, cid, tid, "approve"))
    bodies = [r1.json(), r2.json()]
    check(sorted(r.status_code for r in (r1, r2)) == [200, 200], "两个都返回 200（重复的那个是幂等成功）")
    run_ids = [b["run_id"] for b in bodies if b.get("run_id")]
    check(len(run_ids) == 1 and sum(bool(b.get("already")) for b in bodies) == 1, "只开了一个续跑，另一个标记 already")
    evs = await sse(client, run_ids[0])
    check(evs[-1]["status"] == "completed", f"续跑结束：{evs[-1]['stop_reason']}")
    check(await ledger(pg) == before + 1, "账本只多一行")


async def scenario_reject(client, pg):
    print("场景 2：拒绝")
    cid, tid = await suspended(client, 400)
    before = await ledger(pg)
    r = await decide(client, cid, tid, "reject")
    evs = await sse(client, r.json()["run_id"])
    check(evs[-1]["stop_reason"] == "end_turn", "续跑正常收尾（end_turn）")
    msgs = (await client.get(f"{BASE}/conversations/{cid}/messages", headers=H)).json()["messages"]
    check("[未执行] 用户拒绝" in msgs[-2]["content"][0]["content"], "模型收到的 tool_result 是拒绝说明")
    print("  模型回答：", msgs[-1]["content"][-1].get("text", "")[:80])
    check(await ledger(pg) == before, "账本没变")


async def scenario_kill(client, pg, log):
    print("场景 3：执行中强杀进程")
    cid, tid = await suspended(client, 500)
    before = await ledger(pg)
    await decide(client, cid, tid, "approve")
    for _ in range(50):
        if (await approvals(client, cid))[tid]["status"] == "executing":
            break
        await asyncio.sleep(0.1)
    check((await approvals(client, cid))[tid]["status"] == "executing", "transfer 正在执行（睡 8 秒中）")
    return cid, tid, before


async def main():
    admin = await asyncpg.connect(E2E_URL.rsplit("/", 1)[0] + "/oasis")
    await admin.execute("DROP DATABASE IF EXISTS oasis_e2e WITH (FORCE)")
    await admin.execute("CREATE DATABASE oasis_e2e")
    await admin.close()
    log = open(BACKEND / "scripts" / "e2e_m5_server.log", "w", encoding="utf-8")
    proc = start(log, delay="8")
    pg = await asyncpg.connect(E2E_URL)
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            await scenario_concurrent(client, pg)
            await scenario_reject(client, pg)
            cid, tid, before = await scenario_kill(client, pg, log)
            proc.kill()                                     # TerminateProcess：不走 lifespan 收尾
            proc.wait()
            print("  已强杀，重启（这次不睡）")
            proc = start(log)
            a = (await approvals(client, cid))[tid]
            check(a["status"] == "outcome_unknown", f"重启后状态：{a['status']}")
            await asyncio.sleep(3)
            check(await ledger(pg) == before, "等 3 秒账本没变：没有自动重试")
            check((await client.get(f"{BASE}/conversations/{cid}/active_run", headers=H)).json()["run_id"] is None,
                  "没有自动续跑（等人核实）")
            r = await decide(client, cid, tid, "not_done")
            evs = await sse(client, r.json()["run_id"])
            check(evs[-1]["status"] == "completed", "核实「未生效」后续跑收尾")
            msgs = (await client.get(f"{BASE}/conversations/{cid}/messages", headers=H)).json()["messages"]
            check("[结果未知]" in msgs[-2]["content"][0]["content"], "模型收到结果未知 + 用户核实结论")
            print("  模型回答：", msgs[-1]["content"][-1].get("text", "")[:80])
            check(await ledger(pg) == before, "全程没有执行第二次")
    finally:
        await pg.close()
        proc.kill()
        log.close()
    print("全部通过")


if __name__ == "__main__":
    asyncio.run(main())
