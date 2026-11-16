"""P4 验收：经验记忆的门禁和收益 / 回归（真实模型，花钱）。
.venv\\Scripts\\python scripts/lesson_eval.py [--smoke]

流程（用户选 A）：种子对话跑出带信号的 run → 提炼候选，再混进已知的坏经验 → L0 → 题库基线 → 每条候选跑 L1 →
在留出的测试集上比「无经验 ×6 / 不过门禁全部提交 ×3 / 过门禁 ×3 / 去掉证伪题的消融 ×3」。
所有 trial 用同一个服务配置（MEMORY=0、LESSONS=1、LESSON_DEMO=1），「无经验」= 这个用户名下没有经验，tools 完全一样。
每次 trial 一个新用户；要带经验时把原经验复制给他（gate.copy_of 记原编号），注入记录按原编号归因。
--smoke：只跑种子 + 提炼 + L0 + 4 道题库基线，用来先确认演示工具真的会让模型犯错（约 $0.05）。
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import agent_eval as ae  # noqa: E402  复用起服务、建用户、等 run、收集
from app import db, lesson_demo, lesson_distill as ld, lesson_gate as lg, lesson_l1 as l1  # noqa: E402
from app import lesson_report as lr, lesson_store as ls, lesson_trace as lt, memory_store as ms, runner  # noqa: E402,F401
from app.eval_collect import add_usage, cost_usd  # noqa: E402
from app.eval_grade import grade  # noqa: E402

CASES = ae.BACKEND / "eval" / "lesson_cases.jsonl"
ENV = {"MEMORY": "0", "LESSONS": "1", "LESSON_DEMO": "1", "WEB_TOOLS": "0", "SUBAGENTS": "0"}
PERMS = ["weather:read", "stock:read"]
DISTILL_EST = {"input_tokens": 1500, "output_tokens": 1500}  # 提炼请求的 usage 不落库，按每次估
SMOKE_BANK = ("b-exp-1", "b-fl-1", "b-calc-1", "b-wx-1")
_COPY = ("INSERT INTO memory_lessons (user_id, precondition, behavior, expected, verify, tools, rationale, outcome, "
         "source_run, status, gate, embedding) SELECT $1, precondition, behavior, expected, verify, tools, rationale, "
         "outcome, source_run, 'active', jsonb_build_object('copy_of', id, 'l1', jsonb_build_object('summary', gate->'l1'->'summary')), embedding FROM memory_lessons "
         "WHERE id = ANY($2::bigint[])")


def checks_of(verify: list) -> list[dict]:
    return [{"kind": "code", "name": v["name"], "args": v.get("args", {})} for v in verify]


async def say(client, token, cid, text) -> str:
    r = await client.post(f"{ae.BASE}/conversations/{cid}/messages", headers={"Authorization": f"Bearer {token}"},
                          json={"text": text})
    run_id = r.json()["run_id"]
    await ae.wait_run(db.pool(), run_id)
    return run_id


async def trial(client, text: str, checks: list, lessons: list[int], tag: dict) -> dict:
    """一次 trial：新用户（可带经验副本）→ 新对话问一句 → 判分 → 读注入记录。环境故障记 passed=None，不当答错。"""
    pool = db.pool()
    try:
        uid, token = await ae.new_user(pool, PERMS, [])
        if lessons:
            await pool.execute(_COPY, uid, lessons)
        cid = (await client.post(f"{ae.BASE}/conversations", headers={"Authorization": f"Bearer {token}"})).json()[
            "conversation_id"]
        run_id = await say(client, token, cid, text)
        t, _ = await ae.collect(pool, cid, uid, ae.MODEL)
        g = await grade({"checks": checks}, t)
        injected = [int(r["o"]) for r in await pool.fetch(
            "SELECT l.gate->>'copy_of' AS o FROM lesson_uses u JOIN memory_lessons l ON l.id = u.lesson_id "
            "WHERE u.run_id = $1", run_id)]
        usage: dict = {}
        for r in await pool.fetch("SELECT usage FROM messages WHERE conversation_id = $1 AND usage IS NOT NULL", cid):
            add_usage(usage, r["usage"])
        return tag | {"passed": g["passed"], "grades": g["grades"], "answer": t.answer[:300], "injected": injected,
                      "calls": [(c["name"], c["input"], c["is_error"]) for c in t.calls],
                      "cost_usd": cost_usd(usage, ae.MODEL)}
    except Exception as e:
        return tag | {"passed": None, "injected": [], "cost_usd": 0.0, "env_error": f"{type(e).__name__}: {e}"}


async def many(client, jobs: list[tuple]) -> list[dict]:
    """jobs = [(text, checks, lessons, tag)]，并发 ae.PARALLEL 个。"""
    sem = asyncio.Semaphore(ae.PARALLEL)

    async def one(j):
        async with sem:
            r = await trial(client, *j)
            print(f"  {r.get('phase')} {r.get('probe')} {r.get('config', '')}#{r.get('trial')} "
                  f"passed={r['passed']} inj={r['injected']} {r.get('env_error', '')}", flush=True)
            return r
    return await asyncio.gather(*(one(j) for j in jobs))


async def seed_and_distill(client, cases) -> tuple:
    learner, token = await ae.new_user(db.pool(), PERMS, [])
    seed_run, scans = {}, []
    for s in [c for c in cases if c["set"] == "seed"]:
        cid = (await client.post(f"{ae.BASE}/conversations", headers={"Authorization": f"Bearer {token}"})).json()[
            "conversation_id"]
        runs = [await say(client, token, cid, m) for m in s["messages"]]
        seed_run[s["id"]] = runs[0]
        for r in runs:
            if (out := await ld.scan(r)) is not None:
                scans.append({"seed": s["id"], "run": str(r), **out})
                print("  scan", s["id"], out, flush=True)
    labels = {}
    for i in (x for s in scans for x in s.get("candidates", [])):
        labels[i] = ("real", None)
    for b in [c for c in cases if c["set"] == "bad"]:
        ep = await lt.episode(seed_run[b["source"]])
        lid = await ls.add_candidate(learner, seed_run[b["source"]], {**b["h"], "outcome": ep.outcome or "recovered"}, [])
        await ls.index(lid, b["h"]["precondition"])
        labels[lid] = (b["id"], b["expect"])
    for lid in sorted(labels):
        why = await lg.run_l0(learner, lid)
        print(f"  L0 L{lid} {labels[lid][0]}: {'过' if not why else why}", flush=True)
    return learner, scans, labels


async def run_l1(client, learner, labels, bank, base_recs) -> None:
    by = {p["id"]: p for p in bank}
    for p in bank:
        p["base"] = l1.majority([r["passed"] for r in base_recs if r["probe"] == p["id"]])
    bank_vec = dict(zip([p["id"] for p in bank], await ms.embedder().embed([p["input"] for p in bank])))
    for lid in sorted(labels):
        x = await ls.get(learner, lid)
        if x["status"] != "candidate":
            continue
        [v] = await ms.embedder().embed([x["precondition"]])
        sims = {k: sum(a * b for a, b in zip(v, w)) for k, w in bank_vec.items()}  # bge-m3 输出已归一化
        roles = l1.pick(x, bank, sims)
        ep = await lt.episode(x["source_run"])
        src_role = "source" if x["outcome"] != "failed" else "source_info"  # 彻底失败的源任务没有对照，不计收益
        src_checks = checks_of(x["verify"])
        jobs = [(ep.said[0][1], src_checks, [], {"phase": "l1", "lesson": lid, "probe": "source", "role": "base", "trial": i})
                for i in range(3)]
        jobs += [(ep.said[0][1], src_checks, [lid], {"phase": "l1", "lesson": lid, "probe": "source", "role": src_role,
                                                     "trial": i}) for i in range(3)]
        jobs += [(p["input"], p["checks"], [lid], {"phase": "l1", "lesson": lid, "probe": p["id"], "role": role, "trial": i})
                 for role, ps in roles.items() for p in ps for i in range(3)]
        got = await many(client, jobs)
        src_base = l1.majority([r["passed"] for r in got if r["role"] == "base"])
        rows = [{"probe": "source", "role": src_role, "base": src_base,
                 "with": [r["passed"] for r in got if r["role"] == src_role]}]
        rows += [{"probe": p["id"], "role": role, "base": by[p["id"]]["base"],
                  "with": [r["passed"] for r in got if r["probe"] == p["id"]]} for role, ps in roles.items() for p in ps]
        d, abl = l1.decide(rows), l1.decide(rows, use_falsify=False)
        await ls.decide(lid, "l1", {**d, "ablation": abl, "rows": rows,
                                    "cost_usd": sum(r["cost_usd"] for r in got)}, "active" if d["commit"] else "rejected")
        print(f"  L1 L{lid} {labels[lid][0]}: {d['summary']} → {'提交' if d['commit'] else d['why']}；"
              f"去掉证伪题 → {'提交' if abl['commit'] else '拒'}", flush=True)


async def test_phase(client, tests, configs: dict[str, list[int]]) -> list[dict]:
    jobs = [(t["input"], t["checks"], [], {"phase": "test", "config": "base", "probe": t["id"], "trial": i})
            for t in tests for i in range(6)]
    jobs += [(t["input"], t["checks"], ids, {"phase": "test", "config": c, "probe": t["id"], "trial": i})
             for c, ids in configs.items() for t in tests for i in range(3)]
    return await many(client, jobs)


def report(snap: dict, cands: list[dict], gate: dict, cmp: dict | None, harm: dict, bank: list[dict]) -> str:
    f = lambda v: {True: "过", False: "没过", None: "—"}[v]  # noqa: E731
    s = lambda fl: "".join({True: "✓", False: "✗", None: "?"}[p] for p in fl)  # noqa: E731
    L = [f"# P4 经验记忆验收 {snap['time']}", "", f"模型 {snap['model']}；花费合计约 ${snap['cost_usd']:.3f}"
         f"（含提炼估算）；{'冒烟' if snap['smoke'] else '全量'}", "", "## 题库基线（各 3 次）", ""]
    L += [f"- {p['id']}：{f(p.get('base'))} {s(p.get('base_pass', []))}" for p in bank]
    L += ["", "## 候选和门禁", "", "| 编号 | 来源 | 应拦在 | L0 | L1 | 提交 | 去掉证伪题 |", "|---|---|---|---|---|---|---|"]
    for c in cands:
        L.append(f"| L{c['id']} | {c['label']} | {c['expect'] or '真实提炼'} | {'过' if c['l0_ok'] else '；'.join(c['l0_why'])} "
                 f"| {c['l1']} | {'是' if c['commit'] else '否'} | {'是' if c['commit_noF'] else '否'} |")
    for c in cands:
        L.append(f"- L{c['id']} 前提：{c['precondition']}；做法：{c['behavior']}")
    L += ["", f"**已知坏经验 {gate['known_bad']} 条，拦下 {gate['blocked']} 条（{gate['blocked_rate']:.0%}），"
          f"其中 {gate['at_expected_level']} 条拦在预期那一级；去掉证伪题会放进来的：{gate['leaked_without_falsify'] or '无'}**"]
    if cmp:
        cfgs = [c for c in ("all", "gated", "noF") if c in cmp["totals"]]
        L += ["", "## 测试集（无经验 6 次，其余各 3 次）", "",
              f"噪声底：无经验前 3 次和后 3 次多数翻转 {cmp['noise_flips']}/{cmp['noise_n']} 道", ""]
        for c in cfgs:
            t, h = cmp["totals"][c], harm[c]
            L.append(f"- **{c}**：收益 {t['gains']}、回归 {t['regressions']}；提交 {h['committed']} 条，"
                     f"有害 {len(h['harmful'])} 条（{h['rate']:.0%}）{h['harmful'] or ''}")
        L += ["", "| 题 | 基准 | 复测 | 无经验各次 | " + " | ".join(cfgs) + " |", "|---" * (4 + len(cfgs)) + "|"]
        for r in cmp["rows"]:
            L.append(f"| {r['probe']} | {f(r['base1'])} | {f(r['base2'])} | {s(r['base_pass'])} | " + " | ".join(
                f"{s(r[c]['pass'])}{' 收益' if r[c]['gain'] else ''}{' 回归' if r[c]['regression'] else ''}"
                f"{' 注入' + str(r[c]['blamed']) if r[c]['blamed'] else ''}" for c in cfgs) + " |")
    return "\n".join(L) + "\n"


async def main(args) -> None:
    import asyncpg
    cases = [json.loads(x) for x in CASES.read_text(encoding="utf-8").splitlines() if x.strip()]
    bank = [c for c in cases if c["set"] == "bank" and (not args.smoke or c["id"] in SMOKE_BANK)]
    tests = [c for c in cases if c["set"] == "test"]
    admin = await asyncpg.connect(ae.ADMIN_URL)  # 每次全新的评测库
    await admin.execute("DROP DATABASE IF EXISTS oasis_eval WITH (FORCE)")
    await admin.execute("CREATE DATABASE oasis_eval")
    await admin.close()
    lesson_demo.register()  # 脚本进程里也要有这两个工具：L0 要查工具表
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    recs, cmp, harm = [], None, {}
    with open(ae.BACKEND / "scripts" / "lesson_eval_server.log", "w", encoding="utf-8") as log:
        proc = ae.start_server(ENV, log)
        try:
            await db.connect(ae.EVAL_URL)
            async with httpx.AsyncClient(timeout=60) as client:
                learner, scans, labels = await seed_and_distill(client, cases)
                recs += await many(client, [(p["input"], p["checks"], [], {"phase": "bank", "probe": p["id"], "trial": i})
                                            for p in bank for i in range(3)])
                for p in bank:
                    p["base_pass"] = [r["passed"] for r in recs if r.get("probe") == p["id"]]
                    p["base"] = l1.majority(p["base_pass"])
                if not args.smoke:
                    await run_l1(client, learner, labels, bank, recs)
            rows = {r["id"]: r for r in await db.pool().fetch(
                "SELECT id, precondition, behavior, status, gate FROM memory_lessons WHERE user_id = $1", learner)}
            cands = [{"id": i, "label": lab, "expect": exp_, "precondition": rows[i]["precondition"],
                      "behavior": rows[i]["behavior"], "l0_ok": rows[i]["gate"]["l0"]["ok"],
                      "l0_why": rows[i]["gate"]["l0"]["why"], "l1": rows[i]["gate"].get("l1", {}).get("summary", "—"),
                      "commit": rows[i]["status"] == "active",
                      "commit_noF": bool(rows[i]["gate"].get("l1", {}).get("ablation", {}).get("commit"))}
                     for i, (lab, exp_) in sorted(labels.items())]
            if not args.smoke:
                async with httpx.AsyncClient(timeout=60) as client:
                    configs = {"all": [c["id"] for c in cands], "gated": [c["id"] for c in cands if c["commit"]],
                               "noF": [c["id"] for c in cands if c["commit_noF"]]}
                    trecs = await test_phase(client, tests, configs)
                recs += trecs
                cmp = lr.compare(tests, trecs, list(configs))
                harm = {c: lr.harmful(cmp, c, ids) for c, ids in configs.items()}
            spent = await db.pool().fetch("SELECT m.usage FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                                          "WHERE c.user_id = $1 AND m.usage IS NOT NULL", learner)
            seed_cost = sum(cost_usd(r["usage"], ae.MODEL) for r in spent)
            n_distill = sum(1 for s in scans if s.get("signals"))
            l1_cost = sum(r["gate"].get("l1", {}).get("cost_usd", 0) for r in rows.values())
            await db.close()
        finally:
            proc.kill()
            proc.wait()
    gate = lr.gate_table(cands)
    snap = {"time": stamp, "model": ae.MODEL, "smoke": args.smoke, "env": ENV, "dataset_sha": ae.sha(CASES),
            "cost_usd": sum(r.get("cost_usd", 0) for r in recs) + seed_cost + l1_cost
            + n_distill * cost_usd(DISTILL_EST, ae.MODEL), "scans": scans}
    out = ae.BACKEND / "eval" / "results" / f"lesson-eval-{stamp}"
    out.parent.mkdir(parents=True, exist_ok=True)  # 公开仓库不带 eval/results，先建好，免得付费跑完写结果时报错
    out.with_suffix(".json").write_text(json.dumps({"snapshot": snap, "candidates": cands, "gate": gate, "compare": cmp,
                                                    "harmful": harm, "records": recs}, ensure_ascii=False, indent=1,
                                                   default=str), encoding="utf-8")
    out.with_suffix(".md").write_text(report(snap, cands, gate, cmp, harm, bank), encoding="utf-8")
    print("写入", out.with_suffix(".md"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    asyncio.run(main(ap.parse_args()))
