"""O2 双实例实测：run 跑在 A，SSE 连在 B。先起两个实例（假慢模型，不花钱）：
    uv run python scripts/serve_o1.py 8011   和   uv run python scripts/serve_o1.py 8012
再跑：uv run python scripts/e2e_o2.py [A端口 B端口]

场景（两个会话同时跑）：
1. 一直连着 B：中途 pg_terminate_backend 杀掉 B 的 LISTEN 连接，流不断，靠「重连后唤醒拉取任务」补齐。
2. 带游标重连：读到第 10 个事件断开 → 杀 B 的 LISTEN → 等 3 秒 → 带 Last-Event-ID 重连 B，补齐到 run_end。
断言：两路都 seq 从 1 连续到 run_end、不缺不重，并且和 A 上从 0 回放的结果逐条一致。
"""

import asyncio
import json
import subprocess
import sys

import asyncpg
import httpx

A, B = (int(p) for p in sys.argv[1:3]) if len(sys.argv) > 2 else (8011, 8012)
H = {"Authorization": "Bearer demo-token"}
DB = "postgresql://oasis:oasis@127.0.0.1:5433/oasis_o1"


def pid_on(port: int) -> int:
    """监听这个端口的进程号（= 该实例 LISTEN 连接 application_name 里的 pid）。"""
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    return int(next(l.split()[-1] for l in out.splitlines() if f"127.0.0.1:{port} " in l and "LISTEN" in l))


async def kill_listen(port: int) -> int:
    conn = await asyncpg.connect(DB)
    try:
        app = f"oasis-listen-{pid_on(port)}"
        killed = await conn.fetchval("SELECT count(pg_terminate_backend(pid)) FROM pg_stat_activity "
                                     "WHERE application_name = $1", app)
        print(f"  pg_terminate_backend {app}: {killed}")
        return killed
    finally:
        await conn.close()


async def read(c: httpx.AsyncClient, port: int, run_id: str, after: int, stop_after: int | None = None) -> list[dict]:
    """读 SSE 直到 run_end（或读够 stop_after 个就主动断开）。"""
    got = []
    async with c.stream("GET", f"http://127.0.0.1:{port}/runs/{run_id}/stream",
                        headers={**H, "Last-Event-ID": str(after)}) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                got.append(json.loads(line[6:]))
                if got[-1]["type"] == "run_end" or (stop_after and len(got) >= stop_after):
                    break
    return got


async def start_run(c: httpx.AsyncClient) -> str:
    cid = (await c.post(f"http://127.0.0.1:{A}/conversations", headers=H)).json()["conversation_id"]
    r = await c.post(f"http://127.0.0.1:{A}/conversations/{cid}/messages", headers=H, json={"text": "写长一点"})
    return r.json()["run_id"]


def check(name: str, events: list[dict], full: list[dict]) -> None:
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(1, len(seqs) + 1)), f"{name}: seq 不连续 {seqs}"
    assert events[-1]["type"] == "run_end" and events == full, f"{name}: 和 A 的全量回放不一致"
    print(f"  {name}: {len(seqs)} 个事件，seq 1..{seqs[-1]} 连续，与 A 回放一致")


async def main() -> None:
    async with httpx.AsyncClient(timeout=None) as c:
        run1, run2 = await start_run(c), await start_run(c)
        print(f"A={A} 上跑 run1={run1[:8]} run2={run2[:8]}，SSE 都连 B={B}")

        async def stay_connected():
            task = asyncio.create_task(read(c, B, run1, 0))
            await asyncio.sleep(8)
            await kill_listen(B)  # 流不断开：B 重连 LISTEN 后唤醒拉取任务补齐
            return await task

        async def reconnect_with_cursor():
            first = await read(c, B, run2, 0, stop_after=10)
            print(f"  run2 断开于 seq {first[-1]['seq']}")
            await asyncio.sleep(1)
            await kill_listen(B)
            await asyncio.sleep(3)  # 这期间 A 还在写，B 的 LISTEN 正在重连
            rest = await read(c, B, run2, first[-1]["seq"])
            print(f"  run2 带 Last-Event-ID={first[-1]['seq']} 重连，补到 seq {rest[-1]['seq']}")
            return first + rest

        got1, got2 = await asyncio.gather(stay_connected(), reconnect_with_cursor())
        check("一直连着", got1, await read(c, A, run1, 0))
        check("带游标重连", got2, await read(c, A, run2, 0))


asyncio.run(main())
