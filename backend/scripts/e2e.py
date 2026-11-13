"""端到端验证（真实服务 + 真实模型）：先启动服务，再 uv run python scripts/e2e.py

场景：
1. 两个订阅者同时看同一个 run，收到完全相同的事件；结束后历史 = user/assistant/tool_result/assistant
2. SSE 读到第一个事件就断开，run 仍在后台跑完
3. run 跑着时再发一条 → 409
4. 点停止 → run_end cancelled，会话恢复可发
"""

import asyncio
import json

import httpx

BASE = "http://127.0.0.1:8000"
H = {"Authorization": "Bearer demo-token"}


async def sse_events(client: httpx.AsyncClient, run_id: str, limit: int | None = None):
    """读 SSE，按空行切分事件，返回 data 解析后的 dict 列表。limit 到了就主动断开。"""
    events, buf = [], {}
    async with client.stream("GET", f"{BASE}/runs/{run_id}/stream", headers=H, timeout=180) as r:
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                buf["data"] = json.loads(line[6:])
            elif line == "" and "data" in buf:
                events.append(buf.pop("data"))
                if events[-1]["type"] == "run_end" or (limit and len(events) >= limit):
                    break
    return events


async def send(client, cid, text):
    return await client.post(f"{BASE}/conversations/{cid}/messages", headers=H, json={"text": text})


async def wait_idle(client, cid, timeout=180):
    for _ in range(timeout * 2):
        r = await client.get(f"{BASE}/conversations/{cid}/active_run", headers=H)
        if r.json()["run_id"] is None:
            return
        await asyncio.sleep(0.5)
    raise TimeoutError("run 没有结束")


def check(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    assert cond, msg


async def main():
    async with httpx.AsyncClient() as c:
        cid = (await c.post(f"{BASE}/conversations", headers=H)).json()["conversation_id"]

        print("场景 1：两个订阅者 + 工具调用")
        run_id = (await send(c, cid, "北京和上海今天天气怎么样？")).json()["run_id"]
        a, b = await asyncio.gather(sse_events(c, run_id), sse_events(c, run_id))
        check([e["seq"] for e in a] == [e["seq"] for e in b], f"两个订阅者事件一致（{len(a)} 个）")
        check(a[-1]["status"] == "completed", f"run_end = {a[-1]['status']}/{a[-1]['stop_reason']}")
        print("  回答：" + "".join(e["text"] for e in a if e["type"] == "text_delta")[:80])
        msgs = (await c.get(f"{BASE}/conversations/{cid}/messages", headers=H)).json()["messages"]
        roles = [m["role"] for m in msgs]
        check(roles[0] == "user" and roles[-1] == "assistant" and len(msgs) >= 4, f"历史已落库：{roles}")
        saved = {e["message_id"] for e in a if e["type"] == "message_saved"}
        check(saved <= {m["id"] for m in msgs}, "message_saved 的 id 都能在历史里找到（可去重）")

        print("场景 2：SSE 中途断开，run 继续")
        n_before = len(msgs)
        run_id = (await send(c, cid, "深圳呢？")).json()["run_id"]
        first = await sse_events(c, run_id, limit=1)
        check(len(first) == 1, "读到 1 个事件后主动断开")
        await wait_idle(c, cid)
        msgs = (await c.get(f"{BASE}/conversations/{cid}/messages", headers=H)).json()["messages"]
        check(len(msgs) > n_before and msgs[-1]["role"] == "assistant", "断开后 run 仍跑完并存了回答")

        print("场景 3 + 4：并发发送 409，然后停止")
        run_id = (await send(c, cid, "把三个城市的温度加起来，用 calculate 工具算")).json()["run_id"]
        r2 = await send(c, cid, "插队")
        check(r2.status_code == 409, f"第二条被拒：{r2.status_code} {r2.json()['detail']}")
        listener = asyncio.create_task(sse_events(c, run_id))
        await asyncio.sleep(1)
        await c.post(f"{BASE}/runs/{run_id}/cancel", headers=H)
        end = (await listener)[-1]
        check((end["status"], end["stop_reason"]) == ("cancelled", "user_cancel"), f"run_end = {end['status']}")
        check((await send(c, cid, "你好")).status_code == 202, "停止后会话可以继续发消息")
        await wait_idle(c, cid)
        print("全部通过")


if __name__ == "__main__":
    asyncio.run(main())
