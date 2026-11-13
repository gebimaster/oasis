"""O9 第二步：清旧工具结果（不调模型的一级压缩，排在 LLM 摘要前面）。纯函数，循环和 runner 负责存水位。

为什么它比摘要省：摘要把历史前缀整段换掉，之后第一次请求整段 miss（M3 实测命中从 2304 掉到 512）；
清旧结果也改前缀，但只要「投影是确定的」，它只在清的那一轮 miss 一次，之后每轮前缀都一样、照常命中，
而且不花一次摘要调用。投影确定靠三条规则：

1. 替换结果固定：占位符只由原结果算出来（工具名、原字数、kb_search 的块头），同一条永远换成同一段字。
2. 水位只升不降、存库：id <= 水位的消息里可清的结果一律换掉。第 N 轮和第 N+1 轮拿同一个水位，拼出来的前缀逐字节一样；
   不靠「每次重新算该清哪些」，那样历史一变（新结果、压缩）就可能把清过的又放回来。
3. 分批：只有这一批能省下至少 MIN_SAVING 才抬水位，不够一批宁可不清，避免每轮清一点、前缀每轮都变。
最近 KEEP_RECENT 条工具结果消息永远原样：模型正在用的东西不动。
"""

import re

from app import budget, tools

# 哪些工具的旧结果能清：代码里的固定集合，不读运行时注册表（O9 复核②：原来按 tools._REGISTRY 里的 clearable 判断，
# 重启时 WEB_TOOLS=0、沙箱没开、MCP 工具没了、多实例配置不同，同一份历史就会投影出不同的字节，清过的又被放回来）。
# 改这个集合等于改投影规则，和改占位符文字一样，是一次发版级别的前缀变化。
# 不清 fetch / web_search（复核⑤）：seen_urls 每个 run 只从用户文字重建，清掉以后再抓同一个链接要用户重新确认；
# 搜索结果的占位符又没留链接，模型只能再搜一次、花联网额度。要清得先把链接留在占位符里、并从历史里恢复 seen_urls
CLEARABLE = frozenset({"get_weather", "get_stock", "kb_search", "recall", "read_file", "read_tool_output"})
KEEP_RECENT = 3       # 最近几条工具结果消息不清（按消息数，一批并行结果在同一条 user 消息里）
# 一批至少省多少 token 才值得动前缀（也可以按窗口的 1% 定）。复核指出：它比一条抓取结果还小，实际上「一批」
# 常常就是一条消息；请求一直在线上（摘要被拒、额度用完）时水位会每轮往前挪一条。代价不大（只有那条之后的尾巴 miss），
# 但要如实说：这是「每次至少省 2000」，不是攒够很多条才清
MIN_SAVING = 2000
PLACEHOLDER = "[已清理的旧结果] 这是较早一次 {name} 的输出（原 {n} 字），为节省上下文已从对话里移除；需要的话重新调用。"
_KB_HEADER = re.compile(r"^--- 《[^\n]*》[^\n]*第\d+段$", re.M)


def _names(history: list[dict]) -> dict[str, str]:
    """tool_use_id → 工具真名（call_tool 拆包）。"""
    out = {}
    for m in history:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    inp = b.get("input")
                    # 复核①：格式坏的 call_tool（tool_name 不是字符串、input 不是对象）原样留在历史里，
                    # 不检查类型的话后面查集合会抛 TypeError，会话之后每个 run 都崩
                    if b["name"] == tools.CALL_TOOL and isinstance(inp, dict) and isinstance(inp.get("tool_name"), str):
                        out[b["id"]] = inp["tool_name"]
                    else:
                        out[b["id"]] = b["name"]
    return out


def _clearable(name) -> bool:
    return isinstance(name, str) and name in CLEARABLE


def placeholder(name: str, content) -> str:
    text = content if isinstance(content, str) else str(content)
    out = PLACEHOLDER.format(name=name, n=len(text))
    if name == "kb_search":  # 留下块头：模型还知道当时引过哪些段，【标题 第N段】仍对得上
        heads = _KB_HEADER.findall(text)
        if heads:
            out += "\n当时返回过：" + "；".join(h[4:] for h in heads)
    return out


def _result_messages(history: list[dict], names: dict) -> list[dict]:
    return [m for m in history if m["role"] == "user" and isinstance(m["content"], list) and m.get("id") is not None
            and any(b.get("type") == "tool_result" and _clearable(names.get(b.get("tool_use_id"))) for b in m["content"])]


def project(history: list[dict], until: int) -> list[dict]:
    """拼请求用的投影：id <= until 的消息里，可清、且不是错误的工具结果换成占位符。原 history 不动（库里原文也不动）。"""
    if until <= 0:
        return history
    names = _names(history)
    out = []
    for m in history:
        if m["role"] == "user" and isinstance(m["content"], list) and (m.get("id") or 0) and m["id"] <= until:
            blocks = []
            for b in m["content"]:
                name = names.get(b.get("tool_use_id"))
                if b.get("type") == "tool_result" and not b.get("is_error") and _clearable(name):
                    b = {**b, "content": placeholder(name, b.get("content"))}
                blocks.append(b)
            m = {**m, "content": blocks}
        out.append(m)
    return out


def next_watermark(history: list[dict], until: int) -> int | None:
    """下一批：水位之上、最近 KEEP_RECENT 条之前的工具结果消息。省得够 MIN_SAVING 才返回新水位（这一批最新那条的 id）。"""
    names = _names(history)
    msgs = _result_messages(history, names)
    batch = [m for m in msgs[:max(0, len(msgs) - KEEP_RECENT)] if m["id"] > until]
    if not batch:
        return None
    saving = 0
    for m in batch:
        for b in m["content"]:
            name = names.get(b.get("tool_use_id"))
            if b.get("type") == "tool_result" and not b.get("is_error") and _clearable(name):
                saving += budget.estimate_json(b.get("content")) - budget.estimate_text(placeholder(name, b.get("content")))
    return batch[-1]["id"] if saving >= MIN_SAVING else None
