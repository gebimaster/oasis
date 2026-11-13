"""M5 演示写工具：transfer。每执行一次往 ledger 插一行，测试靠数行数证明「只执行一次」。

TRANSFER_DELAY_SECONDS：执行前先睡这么久，给「执行中强杀进程」的验证留出窗口。

O10：带幂等键（= tool_use_id，从 tools.call_id 拿，模型传不进来）。同键重复调用返回同一笔的结果、不重复扣款；
参数不同则报错不执行（同 Stripe 的幂等键语义）。reconcile 是它对外的「按键查询」：结果未知时用它判断转没转。
"""

import asyncio
import os
from uuid import UUID

from app import db, tools


async def _transfer(args: dict) -> str:
    await asyncio.sleep(float(os.getenv("TRANSFER_DELAY_SECONDS", "0")))
    uid, key = tools.context().user_id, tools.call_id.get() or None  # 身份和键都来自服务端，不来自模型
    # O10 复核④：围栏放在同一条语句里。这次调用的执行记录已经不是 executing（被接管、记成结果未知了）就不插：
    # 卡住的旧进程醒来时，「结果未知 → 查不到 → 没转」的结论可能已经给出、用户也可能重做了，再插就是重复扣款
    ledger_id = await db.pool().fetchval(
        "INSERT INTO ledger (user_id, to_name, amount, idem_key) SELECT $1, $2, $3, $4 "
        "WHERE NOT EXISTS (SELECT 1 FROM tool_approval WHERE tool_use_id = $4 AND status <> 'executing') "
        "AND NOT EXISTS (SELECT 1 FROM tool_executions WHERE tool_use_id = $4 AND status <> 'executing') "
        "ON CONFLICT (user_id, idem_key) WHERE idem_key IS NOT NULL DO NOTHING RETURNING id",
        uid, args["to"], args["amount"], key)
    if ledger_id is not None:
        return f"已向 {args['to']} 转账 {args['amount']} 元（示例账本，流水号 {ledger_id}）"
    old = await db.pool().fetchrow("SELECT id, to_name, amount FROM ledger WHERE user_id = $1 AND idem_key = $2",
                                   uid, key)
    if old is None:  # 没插进去、也没有同键的旧账：被围栏挡住了
        raise ValueError("FENCED 这次调用已经被判为结果未知（执行被接管），没有执行")
    if (old["to_name"], float(old["amount"])) != (args["to"], float(args["amount"])):
        raise ValueError("IDEMPOTENCY_CONFLICT 同一个调用编号上次用的参数不同，这次没有执行")
    return (f"已向 {args['to']} 转账 {args['amount']} 元（示例账本，流水号 {old['id']}；"
            "同一调用重复提交，按同一笔处理，没有重复扣款）")


async def _lookup(user_id: UUID, args: dict, key: str) -> str:
    """按键查询：查到就是转了；查不到就是没转。调用方负责「查不到」要等多久才能信（见 approvals.RECONCILE_GRACE）。"""
    found = await db.pool().fetchval("SELECT 1 FROM ledger WHERE user_id = $1 AND idem_key = $2", user_id, key)
    return "done" if found else "not_done"


tools.register(tools.Tool(
    name="transfer",
    description="给指定收款人转账（示例，只记账不真转）。金额单位：元。",
    input_schema={
        "type": "object",
        "properties": {"to": {"type": "string", "minLength": 1, "description": "收款人"},
                       "amount": {"type": "number", "exclusiveMinimum": 0}},
        "required": ["to", "amount"],
        "additionalProperties": False,
    },
    handler=_transfer,
    permission="transfer:write",
    risk=tools.HIGH,  # 有副作用：没有规则命中时默认要人确认
    reconcile=_lookup,
))
