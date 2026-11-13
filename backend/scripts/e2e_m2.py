"""M2 端到端验证（真实模型）：uv run python scripts/e2e_m2.py

脚本自己起 uvicorn（端口 8001、独立库 oasis_e2e），不碰 8000 上的服务和它的库。
场景：
1. 流式进行中断开，带 Last-Event-ID 重连：两段拼起来 == 从 0 全量回放，不丢不重
2. run 进行中强杀进程（TerminateProcess，不走 lifespan 收尾）再重启：已落库的事件能续播，
   结尾是 recover 补写的 run_end(failed/interrupted)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import asyncpg
import httpx

PORT = 8001
BASE = f"http://127.0.0.1:{PORT}"
H = {"Authorization": "Bearer demo-token"}
ADMIN_URL = "postgresql://oasis:oasis@127.0.0.1:5433/oasis"
E2E_URL = "postgresql://oasis:oasis@127.0.0.1:5433/oasis_e2e"
LONG = "请写一篇 600 字左右介绍杭州西湖的短文，分 4 段，不要调用工具。"
BACKEND = Path(__file__).parent.parent


def start_server(log) -> subprocess.Popen:
    env = {**os.environ, "DATABASE_URL": E2E_URL}
    # 直接用当前 venv 的 python 起单进程 uvicorn：kill() 就是真正的强杀，没有子进程残留
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
                            cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(100):
        try:
            httpx.get(BASE + "/", timeout=1)
            return proc
        except httpx.TransportError:
            time.sleep(0.2)
    raise RuntimeError("服务没起来")


async def sse(client, run_id, last_event_id: int | None = None, limit: int | None = None) -> list[dict]:
    """读 SSE；limit 到了就主动断开（模拟断线）。"""
    headers = {**H, **({"Last-Event-ID": str(last_event_id)} if last_event_id is not None else {})}
    events = []
    async with client.stream("GET", f"{BASE}/runs/{run_id}/stream", headers=headers, timeout=180) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
                if events[-1]["type"] == "run_end" or (limit and len(events) >= limit):
                    break
    return events


def seqs(evs):
    return [e["seq"] for e in evs]


def check(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    assert cond, msg


async def new_run(client) -> str:
    cid = (await client.post(f"{BASE}/conversations", headers=H)).json()["conversation_id"]
    r = await client.post(f"{BASE}/conversations/{cid}/messages", headers=H, json={"text": LONG})
    return cid, r.json()["run_id"]


async def scenario_reconnect(client):
    print("场景 1：流式中断开，带 Last-Event-ID 重连")
    cid, run_id = await new_run(client)
    part1 = await sse(client, run_id, limit=5)
    check(part1[-1]["type"] != "run_end", f"第一段收到 {len(part1)} 个事件后断开，run 还没结束")
    await asyncio.sleep(2)  # 断开期间 run 继续产生事件
    part2 = await sse(client, run_id, last_event_id=part1[-1]["seq"])
    full = await sse(client, run_id)
    print(f"  第一段 {seqs(part1)}，第二段 {len(part2)} 个（从 {part2[0]['seq']} 开始），全量 {len(full)} 个")
    check(seqs(part1) + seqs(part2) == seqs(full), "两段拼起来 == 全量回放：不丢不重")
    check(seqs(full) == list(range(1, len(full) + 1)), "seq 从 1 连续递增，无洞")
    check(full[-1]["type"] == "run_end" and full[-1]["status"] == "completed", "以 run_end(completed) 结束")
    msgs = (await client.get(f"{BASE}/conversations/{cid}/messages", headers=H)).json()["messages"]
    saved = "".join(b["text"] for m in msgs if m["role"] == "assistant" for b in m["content"] if b["type"] == "text")
    streamed = "".join(e["text"] for e in full if e["type"] == "text_delta")
    check(streamed == saved, f"text_delta 拼起来 == 落库的助手消息（{len(saved)} 字）")


async def scenario_crash(client, log) -> subprocess.Popen:
    global server
    print("场景 2：run 进行中强杀进程再重启")
    _, run_id = await new_run(client)
    part1 = await sse(client, run_id, limit=8)
    await asyncio.sleep(1.5)  # 客户端已断开，run 继续写库：这些事件客户端没见过，要靠重启后续播补回
    server.kill()  # Windows 上是 TerminateProcess：不跑 lifespan，runner 来不及写 cancelled
    server.wait()
    print(f"  已强杀，客户端崩溃前收到 {seqs(part1)}")
    server = start_server(log)
    part2 = await sse(client, run_id, last_event_id=part1[-1]["seq"])
    full = await sse(client, run_id)
    print(f"  重启后续播 {len(part2)} 个：{[(e['seq'], e['type']) for e in part2][-4:]} ...")
    check(len(part2) > 1, f"断开后、崩溃前落库的 {len(part2) - 1} 个事件在重启后续播回来了")
    check(seqs(part1) + seqs(part2) == seqs(full), "崩溃前 + 重启后续播 == 全量回放")
    check(seqs(full) == list(range(1, len(full) + 1)), "seq 连续，重启没有复用号码")
    end = full[-1]
    check((end["type"], end["status"], end["stop_reason"]) == ("run_end", "failed", "interrupted"),
          "结尾是 recover 补写的 run_end(failed/interrupted)，前端不会一直等")


async def main():
    global server
    admin = await asyncpg.connect(ADMIN_URL)
    await admin.execute("DROP DATABASE IF EXISTS oasis_e2e WITH (FORCE)")
    await admin.execute("CREATE DATABASE oasis_e2e")
    await admin.close()
    with open(BACKEND / "scripts" / "e2e_m2_server.log", "w", encoding="utf-8") as log:
        server = start_server(log)
        try:
            async with httpx.AsyncClient() as client:
                await scenario_reconnect(client)
                await scenario_crash(client, log)
        finally:
            server.kill()
    print("全部通过")


server: subprocess.Popen
if __name__ == "__main__":  # M5 的 e2e 会 import 这里的工具函数，不能顺带跑一遍 M2
    asyncio.run(main())
