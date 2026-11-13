"""M8：改了确定性断言后，用评测库里现成的数据重新判分（不重跑模型，不花钱）。
.venv\Scripts\python scripts/eval_regrade.py <结果名> <题号,题号>   只重判这些题，且它们不能有 LLM 评委断言。"""

import asyncio
import json
import sys
from pathlib import Path

import asyncpg

BACKEND = Path(__file__).parent.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).parent))
from agent_eval import CASES, CONFIGS, EVAL_URL, MODEL, sha  # noqa: E402
from app import db  # noqa: E402
from app.eval_collect import collect  # noqa: E402
from app.eval_grade import grade  # noqa: E402
from app.eval_report import report  # noqa: E402


async def main(name: str, ids: set[str]) -> None:
    path = BACKEND / "eval" / "results" / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in map(json.loads, CASES.read_text(encoding="utf-8").splitlines()) if c["id"] in ids}
    assert all(k["kind"] == "code" for c in cases.values() for k in c["checks"]), "有评委断言的题要重跑，不能只重判"
    pg = await asyncpg.create_pool(EVAL_URL, init=db._init_conn)
    for r in data["records"]:
        if r["case"] in cases and r.get("run_ids"):
            row = await pg.fetchrow("SELECT conversation_id, c.user_id FROM runs JOIN conversations c "
                                    "ON c.id = runs.conversation_id WHERE runs.id = $1", r["run_ids"][0])
            t, _ = await collect(pg, row["conversation_id"], row["user_id"], MODEL)
            g = await grade(cases[r["case"]], t, None, frozenset(CONFIGS[r["config"]]["disabled"]))
            print(r["config"], r["case"], r["trial"], r["passed"], "->", g["passed"])
            r.update(g)
    data["snapshot"].setdefault("regraded", []).append({"cases": sorted(ids), "dataset_sha": sha(CASES)})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    path.with_suffix(".md").write_text(report(data["snapshot"], data["records"]), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], set(sys.argv[2].split(","))))
