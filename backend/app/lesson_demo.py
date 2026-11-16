"""P4 演示工具（用户选 A）：两个带「文档没写的规矩」的工具，只在 LESSON_DEMO=1 时注册（main.py 启动时）。

真实 API 常有这种坑：描述写得像能随便填，实际有隐含约定。经验记忆要学的就是这个。
- submit_expense：amount 的单位是「分」。传 32 不报错，单子上是 0.32 元，只有用户看了才会纠正（静默错误）。
- search_flights：城市只认三字码（PEK、CTU……），传中文名报 INVALID_CITY，报错不说该怎么改（模糊错误）。
都是假数据、没有副作用，所以标 low，不走审批。
"""

import hashlib
import itertools

from app import tools

CODES = {"PEK": "北京", "PKX": "北京", "BJS": "北京", "SHA": "上海", "PVG": "上海", "CTU": "成都", "TFU": "成都",
         "SZX": "深圳", "CAN": "广州", "HGH": "杭州", "XIY": "西安", "WUH": "武汉", "NKG": "南京"}
_seq = itertools.count(1001)


async def _submit_expense(args: dict) -> str:
    amount = args["amount"]  # 单位：分（描述里故意没写）
    return f"报销单 E{next(_seq)} 已提交：{args['category']} {amount / 100:.2f} 元"


async def _search_flights(args: dict) -> str:
    o, d = args["origin"].strip().upper(), args["destination"].strip().upper()
    for v in (o, d):
        if v not in CODES:
            raise ValueError(f"INVALID_CITY: {v}")
    h = int(hashlib.sha256(f"{o}{d}{args['date']}".encode()).hexdigest(), 16)  # 同样的查询给同样的结果
    return (f"{args['date']} {CODES[o]}({o}) → {CODES[d]}({d})：CA{1000 + h % 9000} 08:05 起飞，"
            f"MU{1000 + h // 9000 % 9000} 13:40 起飞")


def register() -> None:
    tools.register(tools.Tool(
        name="submit_expense", description="提交一笔报销单。",
        input_schema={"type": "object", "properties": {
            "amount": {"type": "number", "description": "报销金额"},
            "category": {"type": "string", "description": "类别，如 打车、餐饮、住宿"}},
            "required": ["amount", "category"], "additionalProperties": False},
        handler=_submit_expense, permission=tools.PUBLIC, risk=tools.LOW))
    tools.register(tools.Tool(
        name="search_flights", description="查询某天两个城市之间的航班。",
        input_schema={"type": "object", "properties": {
            "origin": {"type": "string", "description": "出发城市"},
            "destination": {"type": "string", "description": "到达城市"},
            "date": {"type": "string", "description": "日期，YYYY-MM-DD"}},
            "required": ["origin", "destination", "date"], "additionalProperties": False},
        handler=_search_flights, permission=tools.PUBLIC, risk=tools.LOW, concurrency_safe=True))
