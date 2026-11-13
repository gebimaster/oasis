"""M8 第 4 步：trace 和评测结果的查询接口 + 两个原生 JS 页面（/trace、/eval）。

工具的参数和结果在 span 里只存 tool_use_id（用户要看全文，全文在 messages），这里按 tool_use_id 关联回来，
权限和看消息一样：只能看自己的 run。评测结果是本地文件，只给本机开发用，文件名走白名单正则。
"""

import json
import re
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app import db, trace

STATIC = Path(__file__).parent.parent / "static"
RESULTS = Path(__file__).parent.parent / "eval" / "results"
RESULT_NAME = re.compile(r"^agent-eval-\d{8}-\d{6}$")


async def tool_io(run_ids: list[UUID]) -> dict[str, dict]:
    """这些 run 的消息里的 tool_use 输入和 tool_result 输出，按 tool_use_id 取。"""
    rows = await db.pool().fetch("SELECT content FROM messages WHERE run_id = ANY($1::uuid[]) ORDER BY id", run_ids)
    io: dict[str, dict] = {}
    for r in rows:
        for b in r["content"] if isinstance(r["content"], list) else []:
            if b.get("type") == "tool_use":
                io.setdefault(b["id"], {})["input"] = b.get("input")
            elif b.get("type") == "tool_result":
                io.setdefault(b.get("tool_use_id"), {})["result"] = b.get("content")
    return io


def make_router(owned_run) -> APIRouter:
    """owned_run 由 main 传进来（它依赖 main 里的鉴权），避免循环导入。"""
    r = APIRouter()

    @r.get("/runs/{run_id}/trace")
    async def get_trace(run: dict = Depends(owned_run)):
        trace_id = await db.pool().fetchval("SELECT trace_id FROM spans WHERE run_id = $1 LIMIT 1", run["id"])
        if trace_id is None:
            raise HTTPException(404, "这个 run 没有 trace（M8 之前跑的）")
        rows = [dict(x) for x in await db.pool().fetch("SELECT * FROM spans WHERE trace_id = $1", trace_id)]
        io = await tool_io(list({x["run_id"] for x in rows}))
        for x in rows:
            if x["kind"] == "tool":
                x["io"] = io.get(x["attrs"].get("tool_use_id"), {})
        titles = {t["id"]: t["title"] for t in await db.pool().fetch(
            "SELECT id, title FROM runs WHERE id = ANY($1::uuid[])", list({x["run_id"] for x in rows}))}
        for x in rows:
            if x["kind"] == "run" and titles.get(x["run_id"]):
                x["name"] = f"{x['name']}：{titles[x['run_id']]}"  # 子 run 显示任务名
        return {"trace_id": trace_id, "focus_run_id": run["id"], "roots": trace.build_tree(rows)}

    @r.get("/eval/results")
    async def list_results():
        return {"results": sorted((p.stem for p in RESULTS.glob("agent-eval-*.json")), reverse=True)}

    @r.get("/eval/results/{name}")
    async def get_result(name: str):
        if not RESULT_NAME.match(name) or not (p := RESULTS / f"{name}.json").exists():
            raise HTTPException(404)
        return json.loads(p.read_text(encoding="utf-8"))

    @r.get("/trace")
    async def trace_page():
        return FileResponse(STATIC / "trace.html")

    @r.get("/eval")
    async def eval_page():
        return FileResponse(STATIC / "eval.html")

    return r
