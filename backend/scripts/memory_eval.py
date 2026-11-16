"""P2 验收：长期记忆的收益 / 回归（真实模型，花钱）。
.venv\\Scripts\\python scripts/memory_eval.py [--cases gain-name,safe-poison] [--base-trials 6] [--mem-trials 3]

协议（WikiTrace 的收益 / 回归拆分）：
- 每次 trial 一个新用户：先用真实接口跑「之前的对话」（sessions），再开一个新对话问 input。
- base = MEMORY=0：不抽取、没有摘要块；记忆工具还在但库是空的，recall 照样能翻旧对话原文（原文检索基线）。
  mem  = MEMORY=1：之前的对话结束后在后台抽取，等抽取做完才开新对话。两边只差记忆本身。
- base 跑 6 次：前 3 次多数定基准，后 3 次多数和前 3 次比，翻转的比例就是噪声底。mem 跑 3 次取多数。
  收益 = 基准没过、mem 过了；回归 = 基准过了、mem 没过。回归数要和噪声底比，才知道是不是真的变差。
- 判分全用确定性断言（answer_has / answer_lacks），另有两种记忆库断言：items_max（最多写几条）、items_lack（不许出现的字）。
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import asyncpg
import httpx

sys.path.insert(0, str(Path(__file__).parent))
import agent_eval as ae  # noqa: E402  复用起服务、建用户、等 run、收集
from app import db  # noqa: E402
from app.eval_collect import add_usage, cost_usd  # noqa: E402
from app.eval_grade import grade  # noqa: E402

CASES = ae.BACKEND / "eval" / "memory_cases.jsonl"
PERMS = ["weather:read"]
CONFIGS = {"base": {"MEMORY": "0", "WEB_TOOLS": "0", "LESSONS": "0"}, "mem": {"MEMORY": "1", "WEB_TOOLS": "0"}}
JOB_WAIT = 120
EXTRACT_EST = {"input_tokens": 1500, "output_tokens": 1500}  # 抽取请求的 usage 不落库，按每次估（输出含 thinking）
MEM_TOOLS = ("memory_search", "memory_open", "memory_write", "recall")


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def say(client, pg, h, cid, text: str) -> str:
    run_id = (await client.post(f"{ae.BASE}/conversations/{cid}/messages", headers=h, json={"text": text})).json()["run_id"]
    await ae.wait_run(pg, run_id)
    return run_id


async def wait_jobs(pg, run_ids: list[str]) -> None:
    """正常结束的 run 都要等到抽取任务 done / failed；没正常结束的 run 本来就不抽。"""
    q = ("SELECT count(*) FROM runs r WHERE r.id = ANY($1::uuid[]) AND (r.status <> 'completed' OR EXISTS ("
         "SELECT 1 FROM memory_jobs j WHERE j.run_id = r.id AND j.status IN ('done', 'failed')))")
    for _ in range(JOB_WAIT * 2):
        if await pg.fetchval(q, run_ids) == len(run_ids):
            return
        await asyncio.sleep(0.5)
    raise TimeoutError("抽取任务没做完")


async def session(client, pg, token: str, msgs: list[str], mem: bool) -> None:
    h = auth(token)
    cid = (await client.post(f"{ae.BASE}/conversations", headers=h)).json()["conversation_id"]
    runs = [await say(client, pg, h, cid, m) for m in msgs]
    if mem:
        await wait_jobs(pg, runs)


def item_checks(case: dict, items: list[dict]) -> list[dict]:
    out = []
    if "items_max" in case:
        out.append({"check": "items_max", "passed": len(items) <= case["items_max"], "detail": f"写了 {len(items)} 条"})
    for w in case.get("items_lack", []):
        bad = [i["content"] for i in items if w in i["content"]]
        out.append({"check": "items_lack", "passed": not bad, "detail": f"含「{w}」：{bad}"})
    return out


async def run_trial(client, pg, case: dict, cfg: str, trial: int) -> dict:
    rec = {"config": cfg, "case": case["id"], "category": case["category"], "trial": trial}
    mem = cfg == "mem"
    try:
        uid, token = await ae.new_user(pg, PERMS, [])
        for msgs in case.get("others", []):  # 别的用户说过的话：不该被这个用户看到
            await session(client, pg, (await ae.new_user(pg, PERMS, []))[1], msgs, mem)
        for msgs in case.get("sessions", []):
            await session(client, pg, token, msgs, mem)
        h = auth(token)
        cid = (await client.post(f"{ae.BASE}/conversations", headers=h)).json()["conversation_id"]
        for p in case.get("pre", []):  # 新对话里先说的话（比如中途改主意）
            await say(client, pg, h, cid, p)
        await say(client, pg, h, cid, case["input"])
        t, metrics = await ae.collect(pg, cid, uid, ae.MODEL)
        g = await grade(case, t)
        items = [dict(r) for r in await pg.fetch(
            "SELECT id, kind, content, source, status, invalid_at FROM memory_items WHERE user_id = $1 ORDER BY id", uid)]
        ic = item_checks(case, items)
        usage: dict = {}  # 这个用户所有对话的花费（之前的对话 + 新对话）；抽取请求另外估
        for r in await pg.fetch("SELECT m.usage FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                                "WHERE c.user_id = $1 AND m.usage IS NOT NULL", uid):
            add_usage(usage, r["usage"])
        jobs = await pg.fetchval("SELECT count(*) FROM memory_jobs WHERE user_id = $1", uid)
        return rec | {"passed": g["passed"] if not ic else (g["passed"] and all(c["passed"] for c in ic)),
                      "grades": g["grades"] + ic, "answer": t.answer, "items": items, "jobs": jobs,
                      "tools": {n: sum(c["name"] == n for c in t.calls) for n in MEM_TOOLS},
                      # 新对话里 memory_write 的入参：看模型当场记了什么（尾部回显靠它）
                      "writes": [c.get("input") for c in t.calls if c["name"] == "memory_write"],
                      "cost_usd": round(cost_usd(usage, ae.MODEL) + jobs * cost_usd(EXTRACT_EST, ae.MODEL), 6),
                      "latency_s": metrics["latency_s"], "token": token}
    except Exception as e:  # 环境故障单独记，不当成答错
        return rec | {"passed": None, "env_error": f"{type(e).__name__}: {e}"}


def majority(flags: list) -> bool | None:
    ok = [f for f in flags if f is not None]
    return None if not ok or sum(ok) * 2 == len(ok) else sum(ok) * 2 > len(ok)


def analyze(cases: list[dict], recs: list[dict], base_n: int) -> dict:
    rows, half = [], base_n // 2
    for c in cases:
        r = [x for x in recs if x["case"] == c["id"]]
        b = sorted((x for x in r if x["config"] == "base"), key=lambda x: x["trial"])
        m = [x for x in r if x["config"] == "mem"]
        b1, b2 = majority([x["passed"] for x in b[:half]]), majority([x["passed"] for x in b[half:]])
        mm = majority([x["passed"] for x in m])
        rows.append({"case": c["id"], "category": c["category"], "base1": b1, "base2": b2, "mem": mm,
                     "flip": None if None in (b1, b2) else b1 != b2,
                     "gain": b1 is False and mm is True, "regression": b1 is True and mm is False,
                     "base_pass": [x["passed"] for x in b], "mem_pass": [x["passed"] for x in m]})
    judged = [x for x in rows if x["flip"] is not None]
    return {"rows": rows, "gains": sum(x["gain"] for x in rows), "regressions": sum(x["regression"] for x in rows),
            "noise_flips": sum(x["flip"] for x in judged), "noise_n": len(judged)}


def report(snapshot: dict, a: dict, recs: list[dict]) -> str:
    f = lambda v: {True: "过", False: "没过", None: "—"}[v]  # noqa: E731
    L = [f"# P2 记忆验收 {snapshot['time']}", "", f"模型 {snapshot['model']}；base {snapshot['base_trials']} 次、"
         f"mem {snapshot['mem_trials']} 次；花费合计约 ${snapshot['cost_usd']:.3f}（含抽取估算）", "",
         f"**收益 {a['gains']} 道，回归 {a['regressions']} 道；噪声底：base 两组多数翻转 {a['noise_flips']}/{a['noise_n']} 道**", "",
         "| 题 | 类别 | 基准(前 3) | 复测(后 3) | 记忆 | base 各次 | mem 各次 |", "|---|---|---|---|---|---|---|"]
    for r in a["rows"]:
        L.append(f"| {r['case']} | {r['category']} | {f(r['base1'])} | {f(r['base2'])} | {f(r['mem'])} | "
                 f"{''.join({True: '✓', False: '✗', None: '?'}[p] for p in r['base_pass'])} | "
                 f"{''.join({True: '✓', False: '✗', None: '?'}[p] for p in r['mem_pass'])} |")
    for cfg in ("base", "mem"):
        rs = [x for x in recs if x["config"] == cfg and "tools" in x]
        tot = {n: sum(x["tools"][n] for x in rs) for n in MEM_TOOLS}
        L += ["", f"{cfg}：{len(rs)} 次，工具调用合计 {tot}，平均花费 ${sum(x['cost_usd'] for x in rs) / max(len(rs), 1):.4f}"]
    L += ["", "## 记忆库里写了什么（mem 每题第 0 次）", ""]
    for x in recs:
        if x["config"] == "mem" and x["trial"] == 0 and "items" in x:
            until = lambda i: (f"/到 {i['invalid_at'].astimezone():%Y-%m-%d}"  # noqa: E731  库里是 UTC，换回本地日期
                               if i.get("invalid_at") and i["status"] == "active" else "")
            L.append(f"- {x['case']}：" + ("；".join(f"[{i['source']}/{i['status']}{until(i)}] {i['content']}"
                                                   for i in x["items"]) or "（无）"))
    L += ["", "## 没过的回答", ""]
    for x in recs:
        if x.get("passed") is False or x.get("env_error"):
            why = x.get("env_error") or "；".join(g["detail"] for g in x["grades"] if g["passed"] is False)
            L.append(f"- {x['config']} {x['case']}#{x['trial']}：{why} —— {(x.get('answer') or '')[:160]!r}")
    return "\n".join(L) + "\n"


async def run_config(cfg: str, cases: list[dict], trials: int, log) -> list[dict]:
    proc = ae.start_server(CONFIGS[cfg], log)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            pg = await asyncpg.create_pool(ae.EVAL_URL, init=db._init_conn)
            sem = asyncio.Semaphore(ae.PARALLEL)

            async def one(c, i):
                async with sem:
                    r = await run_trial(client, pg, c, cfg, i)
                    print(f"[{cfg}] {c['id']}#{i} passed={r.get('passed')} {r.get('env_error', '')}", flush=True)
                    return r
            recs = await asyncio.gather(*(one(c, i) for c in cases for i in range(trials)))
            await pg.close()
            return recs
    finally:
        proc.kill()
        proc.wait()


async def main(args) -> None:
    cases = [json.loads(x) for x in CASES.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases.split(",")]
    admin = await asyncpg.connect(ae.ADMIN_URL)  # 每次全新的评测库
    await admin.execute("DROP DATABASE IF EXISTS oasis_eval WITH (FORCE)")
    await admin.execute("CREATE DATABASE oasis_eval")
    await admin.close()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(ae.BACKEND / "scripts" / "memory_eval_server.log", "w", encoding="utf-8") as log:
        recs = await run_config("base", cases, args.base_trials, log) + await run_config("mem", cases, args.mem_trials, log)
    a = analyze(cases, recs, args.base_trials)
    snapshot = {"time": stamp, "model": ae.MODEL, "base_trials": args.base_trials, "mem_trials": args.mem_trials,
                "dataset_sha": ae.sha(CASES), "configs": CONFIGS, "parallel": ae.PARALLEL,
                "cost_usd": sum(x.get("cost_usd", 0) for x in recs), "extract_estimate": EXTRACT_EST}
    out = ae.BACKEND / "eval" / "results" / f"memory-eval-{stamp}"
    out.with_suffix(".json").write_text(json.dumps({"snapshot": snapshot, "analysis": a, "records": recs},
                                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(report(snapshot, a, recs), encoding="utf-8")
    print("写入", out.with_suffix(".md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--base-trials", type=int, default=6)
    ap.add_argument("--mem-trials", type=int, default=3)
    asyncio.run(main(ap.parse_args()))
