"""M8 第 3 步：端到端评测 + 三配置对比（真实模型，花钱）。
.venv\\Scripts\\python scripts/agent_eval.py [--configs A,B,C] [--trials 3] [--cases kb-taxi,tool-calc]

用户的选择：① 走 HTTP：脚本自己起 uvicorn（端口 8002、独立库 oasis_eval、独立 Chroma 目录，每次全新），
切配置 = 带不同环境变量重启服务 ② 每次 trial 一个新用户（权限、规则、ledger 互不干扰）。原定「知识库题共用一个
预先入库的用户」，全量时发现 recall 会翻到同一用户别的会话里上一次 trial 的回答（配置 B 的库外题靠它答对），
改成知识库题每次 trial 也新建用户并重新入库（bge-m3 免费，每次约 15 块）③ 并发 3：延迟是在并发 3 下测的，三种配置同条件。
压缩题要很小的上下文窗口，HTTP 传不进单题的预算：只在配置 A 下另起一个 CONTEXT_WINDOW 调小的服务跑，B/C 不碰压缩，记「不适用」。
结果：eval/results/agent-eval-<时间>.json（配置快照 + 每次 trial）和同名 .md 对比表。
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import asyncpg
import httpx
from dotenv import load_dotenv

BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND))
load_dotenv(BACKEND / ".env", override=True)  # shell 里有 ANTHROPIC_* 时必须覆盖（M7 踩过）

from app import db, llm  # noqa: E402
from app.eval_collect import PRICES, collect  # noqa: E402
from app.eval_grade import JUDGE_MODEL, grade  # noqa: E402
from app.eval_report import report  # noqa: E402

PORT, PARALLEL, RUN_TIMEOUT = 8002, 3, 300
BASE = f"http://127.0.0.1:{PORT}"
ADMIN_URL = "postgresql://oasis:oasis@127.0.0.1:5433/oasis"
EVAL_URL = "postgresql://oasis:oasis@127.0.0.1:5433/oasis_eval"
KB_DIR = BACKEND / "data" / "chroma_eval"
CASES = BACKEND / "eval" / "agent_cases.jsonl"
CORPUS = sorted([*(BACKEND / "kb_samples" / "demo").glob("*.md"), *(BACKEND / "kb_samples" / "eval_extra").glob("*.md")])
MODEL = os.getenv("MODEL", "deepseek-flash")
CONFIGS = {"A": {"name": "基线：混合+rerank，子智能体开", "env": {}, "disabled": []},
           "B": {"name": "只用向量（无 BM25、无 rerank）", "env": {"KB_MODE": "vector"}, "disabled": []},
           "C": {"name": "关掉子智能体", "env": {"SUBAGENTS": "0"}, "disabled": ["subagents"]}}
# 触发线 = 40000×0.8 − 6000 = 26000、目标线 22000；垫 200 轮闲聊（本地估算约 2.9 万）。
# 全量时窗口 16000 触发了但「no valid cut」：目标线扣掉 system/工具和摘要预留后几乎没有原文额度
COMPACT_ENV = {"CONTEXT_WINDOW": "40000", "MAX_TOKENS": "6000"}
# O7：截断题把 max_tokens 压到写不完一份报告，逼出续写。DeepSeek 先思考，太小会全被 thinking 吃掉（也是要测的一种截断）
# 真实验收：700 时模型这次刚好写得下、没触发续写（机制断言没测到），降到 400
TRUNC_ENV = {"MAX_TOKENS": "400"}
SPECIAL_ENV = {"compaction": COMPACT_ENV, "truncation": TRUNC_ENV}  # 这些类别只在配置 A 下用各自的环境跑
FILLER_Q = "第{i}个闲聊：随便推荐一个周末在城市里能做的活动，说说理由，越具体越好，比如去哪、几点去、带什么。"
FILLER_A = "第{i}个建议：周末上午九点去城市公园散步，带上水和薄外套，顺路去附近的早市买点水果，下午找家安静的咖啡馆看书，傍晚沿河边骑共享单车回家。"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def start_server(env: dict, log) -> subprocess.Popen:
    full = {**os.environ, "DATABASE_URL": EVAL_URL, "KB_DIR": str(KB_DIR), **env,
            "ENV_PINNED": ",".join(env)}  # 服务端加载 .env 时别覆盖这些（见 main.py）
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
                            cwd=BACKEND, env=full, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(150):
        try:
            httpx.get(BASE + "/", timeout=1)
            return proc
        except httpx.TransportError:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("服务没起来")


async def new_user(pg, perms: list[str], rules: list[dict]) -> tuple:
    token = f"eval-{uuid4().hex[:12]}"  # 只在本地评测库里有效的一次性 token，记进结果文件供查看 trace
    uid = await pg.fetchval("INSERT INTO users (name, api_token) VALUES ('评测用户', $1) RETURNING id", token)
    await pg.executemany("INSERT INTO user_permissions VALUES ($1, $2)", [(uid, p) for p in perms])
    for r in rules:
        await pg.execute("INSERT INTO tool_rules (user_id, tool_pattern, conditions, effect) VALUES ($1, $2, $3, $4)",
                         uid, r["tool_pattern"], r["conditions"], r["effect"])
    return uid, token


async def ingest_kb(client, token: str) -> None:
    h = {"Authorization": f"Bearer {token}"}
    ids = []
    for p in CORPUS:
        lines = p.read_text(encoding="utf-8").splitlines()
        title, text = lines[0].lstrip("# ").strip(), "\n".join(lines[1:]).strip()
        ids.append((await client.post(f"{BASE}/kb/documents", headers=h, json={"title": title, "text": text})).json()["doc_id"])
    for d in ids:  # 202 后台入库：等到 ready
        for _ in range(120):
            st = (await client.get(f"{BASE}/kb/documents/{d}", headers=h)).json()["status"]
            if st != "processing":
                assert st == "ready", f"入库失败 {d}"
                break
            await asyncio.sleep(0.5)


async def wait_run(pg, run_id) -> str:
    for _ in range(RUN_TIMEOUT * 2):
        r = await pg.fetchrow("SELECT status, stop_reason FROM runs WHERE id = $1", run_id)
        if r["status"] not in ("running", "cancelling"):
            return r["stop_reason"]
        await asyncio.sleep(0.5)
    raise TimeoutError(f"run {run_id} 超过 {RUN_TIMEOUT}s")


async def judge(**kw) -> str:
    # 先 thinking 再写 JSON：8000 时有被截断 / 空正文；又不能超过 21333——SDK 对非流式请求按 3600s×max_tokens/128000
    # 估时，超过 10 分钟直接抛 ValueError（anthropic/_base_client.py:857-866）
    return await llm.complete(model=JUDGE_MODEL, max_tokens=16000, **kw)


async def run_trial(client, pg, case: dict, cfg: str, trial: int) -> dict:
    setup = case["setup"]
    uid, token = await new_user(pg, setup["permissions"], setup.get("rules", []))
    if case["category"].startswith("kb"):
        await ingest_kb(client, token)
    h = {"Authorization": f"Bearer {token}"}
    cid = (await client.post(f"{BASE}/conversations", headers=h)).json()["conversation_id"]
    turns = [*setup.get("history", []), *[[FILLER_Q.format(i=i), FILLER_A.format(i=i)] for i in range(setup.get("filler", 0))]]
    for q, a in turns:  # 压缩题：直接往库里垫历史（白盒准备，跑的还是真实服务）
        for role, text in (("user", q), ("assistant", a)):
            await pg.execute("INSERT INTO messages (conversation_id, role, content) VALUES ($1, $2, $3)",
                             cid, role, [{"type": "text", "text": text}])
    rec = {"config": cfg, "case": case["id"], "category": case["category"], "trial": trial, "token": token}
    try:
        run_id = (await client.post(f"{BASE}/conversations/{cid}/messages", headers=h, json={"text": case["input"]})).json()["run_id"]
        stop = await wait_run(pg, run_id)
        if stop == "awaiting_approval" and (act := setup.get("on_approval")):  # 评测脚本代替人点同意 / 拒绝
            for a in (await client.get(f"{BASE}/conversations/{cid}/approvals", headers=h)).json()["approvals"]:
                if a["status"] == "pending":
                    r = (await client.post(f"{BASE}/conversations/{cid}/approvals/{a['tool_use_id']}",
                                           headers=h, json={"action": act})).json()
                    if r.get("run_id"):
                        await wait_run(pg, r["run_id"])
        t, metrics = await collect(pg, cid, uid, MODEL)
        g = await grade(case, t, judge, frozenset(CONFIGS[cfg]["disabled"]))
        # O8 第四步：存下模型实际发出的 kb_search query 和返回的原文。生产里 query 是模型写的，评测题用的是用户原句；
        # 有了它才能回放（scripts/kb_eval.py --replay），引用校验也能离线重算，不用依赖评测库（每次评测都会被清空）
        kb_calls = [{"query": (c.get("input") or {}).get("query"), "is_error": c["is_error"], "result": str(c["result"])}
                    for c in t.calls if c["name"] == "kb_search"]
        return rec | metrics | g | {"answer": t.answer, "kb_calls": kb_calls}  # O8 复核⑪：引用要完整回答才能离线重算
    except Exception as e:  # 环境故障（超时、服务挂了）单独记，不当成模型答错
        return rec | {"passed": None, "env_error": f"{type(e).__name__}: {e}"}


async def run_config(cfg: str, env: dict, cases: list[dict], trials: int, log) -> list:
    proc = start_server(env, log)
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            pg = await asyncpg.create_pool(EVAL_URL, init=db._init_conn)  # jsonb 直接收发 dict，和服务端一致
            sem = asyncio.Semaphore(PARALLEL)

            async def one(case, i):
                async with sem:
                    r = await run_trial(client, pg, case, cfg, i)
                    print(f"[{cfg}] {r['case']}#{i} passed={r.get('passed')} {r.get('env_error', '')}", flush=True)
                    return r
            recs = await asyncio.gather(*(one(c, i) for c in cases for i in range(trials)))
            await pg.close()
            return recs
    finally:
        proc.kill()
        proc.wait()


async def main(args) -> None:
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases.split(",")]
    admin = await asyncpg.connect(ADMIN_URL)  # 每次全新的库和 Chroma 目录：上次的残留不影响这次
    await admin.execute("DROP DATABASE IF EXISTS oasis_eval WITH (FORCE)")
    await admin.execute("CREATE DATABASE oasis_eval")
    await admin.close()
    shutil.rmtree(KB_DIR, ignore_errors=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = BACKEND / "eval" / "results" / f"agent-eval-{stamp}"
    records = []
    with open(BACKEND / "scripts" / "agent_eval_server.log", "w", encoding="utf-8") as log:
        for cfg in args.configs.split(","):
            normal = [c for c in cases if c["category"] not in SPECIAL_ENV]
            recs = await run_config(cfg, CONFIGS[cfg]["env"], normal, args.trials, log)
            records += recs
            for category, env in SPECIAL_ENV.items():
                special = [c for c in cases if c["category"] == category]
                if cfg == "A" and special:
                    records += await run_config(cfg, env, special, args.trials, log)
    snapshot = {"time": stamp, "model": MODEL, "judge_model": JUDGE_MODEL, "trials": args.trials,
                "parallel": PARALLEL, "configs": {k: CONFIGS[k] for k in args.configs.split(",")},
                "compaction_env": COMPACT_ENV, "truncation_env": TRUNC_ENV, "dataset_sha": sha(CASES), "n_cases": len(cases),
                "corpus": {p.stem: sha(p) for p in CORPUS}, "prices_usd_per_m_peak": PRICES,
                "latency_note": f"并发 {PARALLEL} 下测得；审批题不含人决定的等待"}
    out.with_suffix(".json").write_text(json.dumps({"snapshot": snapshot, "records": records}, ensure_ascii=False,
                                                   indent=1, default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(report(snapshot, records), encoding="utf-8")
    print("写入", out.with_suffix(".json"), out.with_suffix(".md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="A,B,C")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--cases", default="")
    asyncio.run(main(ap.parse_args()))
