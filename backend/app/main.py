"""接入层：FastAPI 路由。鉴权从请求头拿 user_id，绝不信请求体。

启动：uv run uvicorn app.main:app --port 8000
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterable
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

# ENV_PINNED 里列的变量以启动进程时传进来的为准、不被 .env 覆盖。评测脚本按类别传 MAX_TOKENS（截断题 400、压缩题 6000），
# 以前被 .env 里的 MAX_TOKENS 盖掉了，这些类别实际跑的是 .env 的值（O7 真实验收时发现）
_pinned = {k: os.environ[k] for k in os.environ.get("ENV_PINNED", "").split(",") if k in os.environ}
load_dotenv(Path(__file__).parent.parent / ".env", override=True)
os.environ.update(_pinned)

from app import approvals, context, db, kb_api, mcp_client, runner, subagent_tool, tools, trace_api  # noqa: E402  需在加载 .env 之后导入
from app import sandbox, sandbox_tools, web_tools  # noqa: E402  M9
from app.hub import END, RESET, hub  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
DEMO_TOKEN = os.getenv("DEMO_TOKEN", "demo-token")  # M1 简易鉴权，仅开发用
# M4 演示用户：(名字, token, 权限)。guest 没有 stock:read，用来验证「看不到也调不了」
MCP_DEMO = ["mcp:demo:echo_upper", "mcp:demo:unit_convert", "mcp:demo:slow_task"]  # MCP 工具每个一条权限
DEMO_USERS = [("demo", DEMO_TOKEN, ["weather:read", "stock:read", "transfer:write", sandbox_tools.PERMISSION,
                                    web_tools.PERMISSION, *MCP_DEMO]),
              ("guest", os.getenv("GUEST_TOKEN", "guest-token"), ["weather:read", "mcp:demo:echo_upper"])]


async def _start_sandbox() -> asyncio.Task | None:
    """M9：装真实后端 → 启动对账 → 每分钟回收空闲沙箱。没配 Cube 或 SANDBOX=0 时，模型看不到沙箱工具。"""
    if os.getenv("SANDBOX") == "0" or not os.getenv("E2B_API_URL"):
        for name in sandbox_tools.NAMES:
            tools.unregister(name)
        return None
    from app.sandbox_e2b import E2BBackend  # 不开沙箱就不加载 SDK
    sandbox.pool = sandbox.Pool(E2BBackend())
    try:
        killed, dropped = await sandbox.pool.reconcile()
        log.warning("sandbox reconcile: killed %d orphans, dropped %d dead rows", killed, dropped)
    except Exception as e:  # Cube 没开不挡启动：调用时会报 NOT_STARTED
        log.warning("sandbox reconcile skipped: %s: %s", type(e).__name__, e)

    async def reap_loop():
        while True:
            await asyncio.sleep(60)
            try:
                if n := await sandbox.pool.reap():
                    log.warning("reaped %d idle sandboxes", n)
            except Exception as e:  # 这一轮失败下一轮再来，循环不能死
                log.warning("sandbox reap failed: %s: %s", type(e).__name__, e)
    return asyncio.create_task(reap_loop())


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.connect()
    # O1：只接管租约过期的 run（连同它们的工具执行、审批执行、span）；别的实例正在跑的不动。之后由 lease_loop 周期扫
    if n := await db.recover_expired_runs(close_blocks=subagent_tool.closing_blocks):
        log.warning("recovered %d runs with expired leases", n)
    if n := await db.recover_kb_documents():  # M6：上次没入完的文档标 failed
        log.warning("marked %d interrupted kb documents as failed", n)
    for name, token, perms in DEMO_USERS:
        uid = await db.user_id_by_token(token) or await db.create_user(name, token)
        await db.grant_permissions(uid, perms)  # 幂等：已有的跳过
        if name == "demo" and not await db.get_rules(uid):  # M5 演示规则：只在还没有规则时种一次
            await db.add_rule(None, "transfer", [{"field": "to", "op": "eq", "value": "blocked_user"}], "deny")
            await db.add_rule(uid, "transfer", [{"field": "amount", "op": "lte", "value": 100}], "allow")
    # M5：已经不等人、但结果还没写回的批次，开续跑接着做（executing → 结果未知 已在上面的过期扫描里做了）
    for conversation_id, run_id in await db.ready_batches():
        runner.start(conversation_id, run_id)
    if os.getenv("SUBAGENTS") == "0":  # M8 对比配置 C：关掉子智能体（模型看不到 spawn_subagents）
        tools.unregister(subagent_tool.NAME)
    if os.getenv("WEB_TOOLS") == "0":  # M9：关掉联网工具（比如评测时不想花搜索费）
        for name in web_tools.NAMES:
            tools.unregister(name)
    reaper = await _start_sandbox()
    leaser = asyncio.create_task(runner.lease_loop())  # O1：续租 + 接管过期 run
    stopper = asyncio.create_task(runner.stop_loop())  # O1：发现别的实例上点的停止
    listener = asyncio.create_task(hub.listen(os.getenv("DATABASE_URL", db.DEFAULT_URL)))  # O2：跨实例推送
    mcp_client.start_all()  # 后台连接，不阻塞启动
    yield
    leaser.cancel()
    stopper.cancel()
    listener.cancel()
    if reaper:
        reaper.cancel()
    await runner.shutdown()
    await mcp_client.stop_all()
    await db.close()


app = FastAPI(title="oasis", lifespan=lifespan)


# ---------------- 鉴权与归属校验（依赖在响应开始前执行，失败直接 401/404） ----------------

async def current_user(authorization: str = Header(default="")) -> UUID:
    token = authorization.removeprefix("Bearer ").strip()
    user_id = await db.user_id_by_token(token) if token else None
    if user_id is None:
        raise HTTPException(401, "invalid token")
    return user_id


async def owned_conversation(conversation_id: UUID, user_id: UUID = Depends(current_user)) -> UUID:
    if not await db.owns_conversation(user_id, conversation_id):
        raise HTTPException(404, "conversation not found")  # 不区分「不存在」和「不是你的」
    return conversation_id


async def owned_run(run_id: UUID, user_id: UUID = Depends(current_user)) -> dict:
    run = await db.get_run(run_id)
    if run is None or not await db.owns_conversation(user_id, run["conversation_id"]):
        raise HTTPException(404, "run not found")
    return run


# ---------------- 路由 ----------------

class SendBody(BaseModel):
    text: str


@app.post("/conversations")
async def create_conversation(user_id: UUID = Depends(current_user)):
    return {"conversation_id": await db.create_conversation(user_id)}


@app.get("/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: UUID = Depends(owned_conversation)):
    return {"messages": await db.list_messages(conversation_id)}


@app.get("/conversations/{conversation_id}/active_run")
async def active_run(conversation_id: UUID = Depends(owned_conversation)):
    return {"run_id": await db.active_run_id(conversation_id)}


@app.post("/conversations/{conversation_id}/messages", status_code=202)
async def send_message(body: SendBody, conversation_id: UUID = Depends(owned_conversation)):
    await approvals.resolve_open(conversation_id)  # O10 复核②：收口前先按键查一次
    try:
        run_id = await db.start_run(conversation_id, [{"type": "text", "text": context.escape_loop_note(body.text)}],
                                    close_batch_blocks=approvals.blocks_for_new_message)
    except db.RunConflict:
        raise HTTPException(409, "上一条消息还在回答中") from None
    runner.start(conversation_id, run_id)  # 后台跑，请求立刻返回
    return {"run_id": run_id}


# M5：动作 → (SET, WHERE)。都是常量，不拼用户输入
DECISIONS = {
    "approve": ("status = 'approved'", "status = 'pending'"),
    "reject": ("status = 'rejected'", "status = 'pending'"),
    "done": ("resolution = 'done', resolved_by = 'user'", "status = 'outcome_unknown' AND resolution IS NULL"),
    "not_done": ("resolution = 'not_done', resolved_by = 'user'", "status = 'outcome_unknown' AND resolution IS NULL"),
    # O10：「让系统核实」：服务端按幂等键查到结论后走这两条（resolved_by 记成 reconcile）
    **{f"reconcile:{v}": sql for v, sql in approvals.RECONCILE_SQL.items()},
}
# 重复提交同一个决定算成功（幂等）：批准之后状态会继续往下走，所以这些状态都算「已经批准过」
SAME = {"approve": {"approved", "executing", "succeeded", "failed", "outcome_unknown"}, "reject": {"rejected"}}


class DecisionBody(BaseModel):
    action: str


@app.get("/conversations/{conversation_id}/approvals")
async def list_approvals(conversation_id: UUID = Depends(owned_conversation)):
    # O10 复核⑤：只有支持按键查询的工具才给「让系统查」按钮
    return {"approvals": [a | {"reconcilable": bool(getattr(tools._REGISTRY.get(a["tool_name"]), "reconcile", None))}
                          for a in await db.list_approvals(conversation_id)]}


@app.post("/conversations/{conversation_id}/approvals/{tool_use_id}")
async def decide(tool_use_id: str, body: DecisionBody, conversation_id: UUID = Depends(owned_conversation)):
    """只收「决定」，不收参数：执行的永远是库里存的那份参数。"""
    action = body.action
    if action == "reconcile":  # O10：不用人去外部系统查，服务端按这次调用的幂等键查一次
        row = next((a for a in await db.open_batch(conversation_id) if a["tool_use_id"] == tool_use_id), None)
        if row is None or row["status"] != "outcome_unknown" or row["resolution"] is not None:
            raise HTTPException(409, {"error": "这条调用不是待核实的结果未知状态"})
        verdict = await approvals.reconcile(row, (await db.conversation_user(conversation_id))["id"])
        if verdict is None:
            raise HTTPException(409, {"error": "暂时查不出结论：这个工具不支持按键查询，或刚执行不久外部可能还在处理。"
                                               "请稍后再试，或自己核实后选「已生效 / 未生效」"})
        action = f"reconcile:{verdict}"
    elif action not in DECISIONS or action.startswith("reconcile:"):
        raise HTTPException(422, "action 只能是 approve / reject / done / not_done / reconcile")
    try:
        r = await db.decide_approval(conversation_id, tool_use_id, *DECISIONS[action])
    except db.RunConflict:
        raise HTTPException(409, {"error": "会话正忙，请稍后再试"}) from None
    if r is None:
        raise HTTPException(404, "approval not found")
    if r.pop("updated"):
        if r["run_id"]:
            runner.start(conversation_id, r["run_id"])  # 批次里没有等人的了：续跑
        return r
    if r["status"] in SAME.get(action, ()) or r["resolution"] == action.removeprefix("reconcile:"):
        return {**r, "already": True}  # 重复点击 / 重连后重放：同一个决定，不报错也不再执行
    raise HTTPException(409, {"error": "状态已经变了", **r})  # 晚到 / 相反的决定：前端按真实状态刷新卡片


@app.get("/conversations/{conversation_id}/subagents")
async def list_subagents(conversation_id: UUID = Depends(owned_conversation)):
    return {"subagents": await db.list_subagents(conversation_id)}


@app.post("/runs/{run_id}/cancel", status_code=202)
async def cancel_run(run: dict = Depends(owned_run)):
    """O1：只回「已受理」：意图落库就返回，真正停下以 SSE 的 run_end 为准（属主可能在别的实例、工具可能吞取消）。
    重复点算受理（状态还是 cancelling）；已结束的 run 回 accepted=false 和它的终态。"""
    changed = await runner.cancel(run["id"])
    status = (await db.get_run(run["id"]))["status"]
    return {"accepted": changed or status == "cancelling", "status": status}


@app.get("/runs/{run_id}/stream", response_class=EventSourceResponse)
async def stream_run(run: dict = Depends(owned_run), after_seq: int | None = None,
                     last_event_id: int | None = Header(default=None)) -> AsyncIterable[ServerSentEvent]:
    """续播：先订阅 → 回放 seq > 游标 → 读实时队列；seq <= 游标的丢掉。run_end 是唯一的结束信号。
    O2：实时队列由 hub 按游标从库里拉，run 在哪个实例上跑都一样；队列满了收到 RESET，不发 run_end 直接结束，前端带游标重连。"""
    run_id = run["id"]
    cursor = last_event_id if last_event_id is not None else (after_seq or 0)  # 重连时 header 为准
    log.info("stream run=%s from seq %s", run_id, cursor)  # O2：看得出刷新/重连是不是带着游标来的
    q = hub.subscribe(run_id, cursor)  # 先订阅再查库：查库期间新推的事件进队列，不会丢
    try:
        for ev in await db.list_events(run_id, cursor):
            yield ServerSentEvent(event=ev["type"], data=ev, id=str(ev["seq"]))
            cursor = ev["seq"]
            if ev["type"] == "run_end":
                return
        # M1 时期结束的 run 没有 run_end 事件：按 runs 表合成一个。M7：cancelling 还没结束（task 在收尾），照常等 run_end
        if run["status"] not in ("running", "cancelling"):
            yield ServerSentEvent(event="run_end", data={
                "type": "run_end", "status": run["status"], "stop_reason": run["stop_reason"]})
            return
        while (ev := await q.get()) is not END and ev is not RESET:
            if ev["seq"] <= cursor:  # 回放里已经发过（订阅和查库之间推的那几条）
                continue
            yield ServerSentEvent(event=ev["type"], data=ev, id=str(ev["seq"]))
            cursor = ev["seq"]
    finally:  # 客户端断开时 FastAPI 会取消这个生成器：只退订，run 继续跑
        hub.unsubscribe(run_id, q)


# ---------------- M6 知识库文档：上传 / 更新 / 删除 / 查状态 ----------------

class KbDocBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


def _check_size(body: KbDocBody) -> None:
    # 注意：到这里 body 已经整个读进内存了；挡的是后面的切块和 embedding 开销，不是内存
    if len(body.text) > kb_api.MAX_CHARS:
        raise HTTPException(413, f"正文超过 {kb_api.MAX_CHARS} 字")


async def _claim(user_id: UUID, doc_id: UUID, title: str | None = None) -> None:
    got = await db.claim_kb_document(user_id, doc_id, title)
    if got == "missing":
        raise HTTPException(404, "document not found")  # 不区分「不存在」和「不是你的」
    if got == "busy":
        raise HTTPException(409, "document is processing")  # 两次 ingest 交错会留下新旧混杂的块


@app.post("/kb/documents", status_code=202)
async def upload_kb_document(body: KbDocBody, user_id: UUID = Depends(current_user)):
    _check_size(body)
    doc_id = await db.create_kb_document(user_id, body.title)
    kb_api.start(user_id, doc_id, body.title, body.text)
    return {"doc_id": doc_id, "status": "processing"}


@app.put("/kb/documents/{doc_id}", status_code=202)
async def update_kb_document(doc_id: UUID, body: KbDocBody, user_id: UUID = Depends(current_user)):
    _check_size(body)
    await _claim(user_id, doc_id, body.title)
    kb_api.start(user_id, doc_id, body.title, body.text)
    return {"doc_id": doc_id, "status": "processing"}


@app.get("/kb/documents")
async def list_kb_documents(user_id: UUID = Depends(current_user)):
    return await db.list_kb_documents(user_id)


@app.get("/kb/documents/{doc_id}")
async def get_kb_document(doc_id: UUID, user_id: UUID = Depends(current_user)):
    if (doc := await db.get_kb_document(user_id, doc_id)) is None:
        raise HTTPException(404, "document not found")
    return doc


@app.delete("/kb/documents/{doc_id}", status_code=204)
async def delete_kb_document(doc_id: UUID, user_id: UUID = Depends(current_user)):
    await _claim(user_id, doc_id)
    await kb_api.delete(user_id, doc_id)


app.include_router(trace_api.make_router(owned_run))  # M8：/runs/{id}/trace、/eval/results、/trace、/eval 页面


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent.parent / "static" / "index.html")


@app.get("/static/{name}.js")
async def static_js(name: str):  # ES module 必须是 JS 的 MIME 类型，否则浏览器拒绝加载
    if name not in ("stream", "approval", "subagents", "trace"):  # 白名单，不拼任意路径
        raise HTTPException(404)
    return FileResponse(Path(__file__).parent.parent / "static" / f"{name}.js", media_type="text/javascript")
