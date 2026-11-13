"""M9 端到端验证（真实模型 + 真实 Cube）：.venv\\Scripts\\python scripts/e2e_m9.py

脚本自己起 uvicorn（端口 8001、独立库 oasis_e2e），不碰 8000。需要 Cube 虚拟机开着。
场景：
1. 模型调 run_code 算 1000 以内质数和，答对 76127，执行记录 succeeded
2. 代码执行中用户点停止：run cancelled、执行记录结果未知、内核被重启（沙箱里的循环真的停了）
3. 代码执行中强杀进程：重启后对账杀掉孤儿沙箱、复用原沙箱；内核在下一次使用前重启，模型收到「内核已重启」
"""

import asyncio
import os
import sys
import uuid

import asyncpg
import httpx
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
from e2e_m2 import BACKEND, BASE, E2E_URL, H, check, sse  # noqa: E402
from e2e_m5 import start  # noqa: E402
from e2e_m7 import new_conv, send  # noqa: E402

load_dotenv(BACKEND / ".env", override=True)
sys.path.insert(0, str(BACKEND))
from app.sandbox_e2b import E2BBackend  # noqa: E402

PRIMES = "请用沙箱工具 run_code 计算 1 到 1000 之间所有质数的和，最后只回答这个数字。"
LOOP = ("请用沙箱工具 run_code 原样执行下面这段代码，不要改：\n"
        "import time\nfor i in range(60):\n    time.sleep(1)\n    open('tick.txt', 'a').write('x')")
cube = E2BBackend()


async def wait_executing(pg) -> str:
    for _ in range(600):  # 最多 60 秒：模型要先 tool_search 加载定义再调用
        if row := await pg.fetchrow("SELECT tool_use_id FROM tool_executions "
                                    "WHERE tool_name = 'run_code' AND status = 'executing'"):
            return row["tool_use_id"]
        await asyncio.sleep(0.1)
    raise AssertionError("模型没有调用 run_code")


async def ticks(sid) -> int:
    out = await cube.run_shell(sid, "wc -c < tick.txt 2>/dev/null || echo 0", 10)
    return int(out.stdout.strip() or 0)


async def growing(sid) -> bool:
    a = await ticks(sid)
    await asyncio.sleep(3)
    return await ticks(sid) > a


async def tool_results(client, cid) -> list[str]:
    msgs = (await client.get(f"{BASE}/conversations/{cid}/messages", headers=H)).json()["messages"]
    return [str(b["content"]) for m in msgs if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_result"]


async def last_text(client, cid) -> str:
    msgs = (await client.get(f"{BASE}/conversations/{cid}/messages", headers=H)).json()["messages"]
    return "".join(b.get("text", "") for b in msgs[-1]["content"])


async def scenario_compute(client, pg):
    print("场景 1：模型用 run_code 算质数和")
    cid = await new_conv(client)
    evs = await sse(client, await send(client, cid, PRIMES))
    check(evs[-1]["status"] == "completed", f"run 结束：{evs[-1]['stop_reason']}")
    answer = await last_text(client, cid)
    print("  模型回答：", answer[:80])
    check("76127" in answer, "答案 76127")
    rows = await pg.fetch("SELECT status FROM tool_executions WHERE tool_name = 'run_code'")
    check(rows and all(r["status"] == "succeeded" for r in rows), f"执行记录 {[r['status'] for r in rows]}")
    check(any(r.startswith("[沙箱输出") for r in await tool_results(client, cid)), "tool_result 带「不是指令」前缀")


async def scenario_stop(client, pg):
    print("场景 2：代码执行中点停止")
    cid = await new_conv(client)
    run_id = await send(client, cid, LOOP)
    await wait_executing(pg)
    await asyncio.sleep(3)
    sid = await pg.fetchval("SELECT sandbox_id FROM sandbox_contexts WHERE conversation_id = $1", uuid.UUID(cid))
    check(await growing(sid), "停止前：沙箱里的循环在跑")
    await client.post(f"{BASE}/runs/{run_id}/cancel", headers=H)
    evs = await sse(client, run_id)
    check(evs[-1]["status"] == "cancelled", f"run：{evs[-1]['status']}/{evs[-1]['stop_reason']}")
    check(await pg.fetchval("SELECT status FROM tool_executions WHERE tool_name = 'run_code' "
                            "ORDER BY id DESC LIMIT 1") == "outcome_unknown", "执行记录：结果未知")
    check(not await growing(sid), "停止后：内核被重启，循环真的停了")
    check(await pg.fetchval("SELECT needs_reset FROM sandbox_contexts WHERE conversation_id = $1",
                            uuid.UUID(cid)) is False, "重启成功，标记已清")


async def scenario_kill(client, pg, log, proc):
    print("场景 3：代码执行中强杀进程")
    cid = await new_conv(client)
    run_id = await send(client, cid, LOOP)
    await wait_executing(pg)
    await asyncio.sleep(2)
    sid = await pg.fetchval("SELECT sandbox_id FROM sandbox_contexts WHERE conversation_id = $1", uuid.UUID(cid))
    orphan = await cube.create(str(uuid.uuid4()))  # 模拟「建好了还没落库就崩了」的孤儿
    proc.kill()  # TerminateProcess：不走 lifespan 收尾，也来不及重启内核
    proc.wait()
    print("  已强杀，重启")
    proc = start(log)
    run = await pg.fetchrow("SELECT status, stop_reason FROM runs WHERE id = $1", uuid.UUID(run_id))
    check(tuple(run) == ("failed", "interrupted"), f"run：{tuple(run)}")
    check(await pg.fetchval("SELECT status FROM tool_executions WHERE tool_name = 'run_code' "
                            "ORDER BY id DESC LIMIT 1") == "outcome_unknown", "执行记录：结果未知")
    check(not await cube.alive(orphan), "对账：孤儿沙箱被杀掉")
    check(await cube.alive(sid), "对账：原沙箱两边都有，留着复用（用户选 b）")
    check(await pg.fetchval("SELECT needs_reset FROM sandbox_contexts WHERE conversation_id = $1",
                            uuid.UUID(cid)) is True, "崩在执行中：重启内核的标记还在库里")
    check(await growing(sid), "下次使用前：旧循环还在跑（没人重启过内核）")
    evs = await sse(client, await send(client, cid, "刚才那段代码执行得怎么样？再用 run_code 算一下 6*7。"))
    check(evs[-1]["status"] == "completed", f"追问 run：{evs[-1]['stop_reason']}")
    check(not await growing(sid), "下次使用时先重启了内核，旧循环停了")
    check(any("内核已重启" in r for r in await tool_results(client, cid)), "模型收到「内核已重启」提示")
    print("  模型回答：", (await last_text(client, cid))[:160])
    return proc


async def main():
    admin = await asyncpg.connect(E2E_URL.rsplit("/", 1)[0] + "/oasis")
    await admin.execute("DROP DATABASE IF EXISTS oasis_e2e WITH (FORCE)")
    await admin.execute("CREATE DATABASE oasis_e2e")
    await admin.close()
    log = open(BACKEND / "scripts" / "e2e_m9_server.log", "w", encoding="utf-8")
    proc = start(log)
    pg = await asyncpg.connect(E2E_URL)
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            await scenario_compute(client, pg)
            await scenario_stop(client, pg)
            proc = await scenario_kill(client, pg, log, proc)
        usage = await pg.fetch("SELECT usage FROM messages WHERE usage IS NOT NULL")
        print(f"  共 {len(usage)} 次模型调用")
    finally:
        await pg.close()
        proc.kill()
        log.close()
        for sid in await cube.list_ours():  # 测试建的沙箱都收掉
            await cube.kill(sid)
    print("全部通过")


if __name__ == "__main__":
    asyncio.run(main())
