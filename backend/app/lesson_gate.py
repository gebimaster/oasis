"""P4 第 4 步：L0 门禁。不调模型，全部是代码规则（用户选 A：2608.21230 里模型筛查对投毒拦下 0/360）。

每条候选都要过，过了才去花钱跑 L1。一条没过就整条拒掉，原因全部写进 gate.l0，方便复盘。
"""

import re
from uuid import UUID

from app import lesson_store as ls, lesson_trace as lt, tools
from app.eval_grade import CHECKS, Trial

VERIFY_CHECKS = ("tool_called", "tool_errors_max", "tool_calls_max", "answer_has", "answer_lacks")
BANNED_TOOLS = ("memory_write",)  # 经验不许教 agent 怎么写记忆：那是绕过 P2「只收用户亲口说的」的捷径
# 「跳过安全机制」类说法：不管前提写得多合理，出现就拒
BYPASS = re.compile(r"(不用|无需|不必|不需要|跳过|绕过|省掉|别)(再)?(等)?(用户)?(确认|审批|批准|授权|权限|核实)"
                    r"|忽略.{0,6}(规则|指令|提示|系统|要求)|ignore (all|previous|prior)|system prompt"
                    r"|直接(转账|付款|支付|删除)", re.I)
URL = re.compile(r"https?://|www\.|\b[\w.-]+\.(com|cn|net|org|io|xyz)\b", re.I)
PII = re.compile(r"\d{11,}|[\w.+-]+@[\w-]+\.\w+|用户(叫|名叫|姓|住在|住址|的(手机|电话|邮箱|地址|身份证|生日))")
MARKUP = re.compile(r"</?\s*(loop_note|user_memory|lessons?|system|summary)\b", re.I)
DUP_SIM = 0.6  # 同样的工具、行为文字二元组重合度 ≥ 0.6 算重复


def _grams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _jaccard(a: str, b: str) -> float:
    x, y = _grams(a), _grams(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def check_verify(verify: list, lesson_tools: set[str]) -> list[str]:
    """验证方式只能用允许的检查，参数要能真的跑起来（拿空的 Trial 试跑一次），涉及的工具必须在 tools 里。"""
    why = []
    for v in verify:
        name, args = v.get("name"), v.get("args", {})
        if name not in VERIFY_CHECKS:
            why.append(f"验证方式用了不允许的检查 {name}")
            continue
        try:
            if name == "answer_lacks":
                [re.compile(p) for p in args.get("patterns", [])]
            CHECKS[name](Trial(), **args)
        except (TypeError, re.error, AttributeError, KeyError) as e:
            why.append(f"验证方式 {name} 的参数不对：{type(e).__name__}")
            continue
        if name.startswith("tool_") and args.get("name") not in lesson_tools:
            why.append(f"验证方式里的工具 {args.get('name')} 不在这条经验的 tools 里")
    if not any(v.get("name", "").startswith("tool_") for v in verify):
        why.append("验证方式里至少要有一条和工具有关的检查")  # 只看回答文字的验证，回放时太容易蒙对
    return why


def fixed_trial(ep: lt.Episode) -> Trial | None:
    """源轨迹「改对之后」那段，拼成判分器认识的 Trial；彻底失败的 run 没有这一段。"""
    if ep is None or ep.fix_from is None:
        return None
    return Trial(answer=ep.reply, calls=[{"name": c["name"], "input": c["args"], "is_error": not c["ok"],
                                          "result": None} for c in ep.calls[ep.fix_from:]])


def l0(h: dict, source_tools: set[str], existing: list[dict], fixed: Trial | None = None) -> list[str]:
    """返回拒绝原因；空列表 = 通过。h 是候选的字段，source_tools 是源轨迹里真的调过的工具，existing 是本人已有的经验。
    fixed：源轨迹改对之后那段。验证方式必须在它上面成立：否则经验是在「自己出题自己考」（如错把元×10 写进验证方式）。"""
    why = []
    text = " ".join(h[k] for k in ("precondition", "behavior", "expected"))
    all_text = text + " " + h["rationale"]
    lesson_tools = set(h["tools"])
    for t in lesson_tools:
        tool = tools._REGISTRY.get(t)
        if tool is None:
            why.append(f"工具 {t} 不存在")
        elif tool.risk == tools.HIGH or t in BANNED_TOOLS:
            why.append(f"经验不许涉及高风险或记忆写入工具 {t}")
        if t not in source_tools:
            why.append(f"工具 {t} 没在源轨迹里调用过")
    # 文字里提到的工具，必须都在 tools 里：防止一条讲航班的经验顺手夹带「以后都先调 transfer」
    for name, tool in tools._REGISTRY.items():
        if name in all_text and name not in lesson_tools:
            why.append(f"文字里提到了 tools 之外的工具 {name}")
    for pat, label in ((BYPASS, "要求跳过确认 / 忽略规则"), (URL, "带网址"), (PII, "带个人信息"),
                       (MARKUP, "带系统标签")):
        if m := pat.search(all_text):
            why.append(f"{label}：「{m.group(0)}」")
    bad_verify = check_verify(h["verify"], lesson_tools)
    why += bad_verify
    if fixed is not None and not bad_verify:
        for v in h["verify"]:
            ok, detail = CHECKS[v["name"]](fixed, **v.get("args", {}))
            if not ok:
                why.append(f"验证方式 {v['name']} 在源 run 改对之后的做法上不成立（{detail}）")
    for e in existing:
        if set(e["tools"]) == lesson_tools and _jaccard(e["behavior"], h["behavior"]) >= DUP_SIM:
            why.append(f"和已有经验 L{e['id']} 重复")
    return why


async def run_l0(user_id: UUID, lesson_id: int) -> list[str]:
    """对一条候选跑 L0 并落库：没过 → rejected；过了只记结果，状态仍是 candidate，等 L1。"""
    h = await ls.get(user_id, lesson_id)
    if h is None or h["status"] != "candidate":
        return ["不是待判的候选"]
    ep = await lt.episode(h["source_run"])
    source_tools = {c["name"] for c in ep.calls} if ep else set()
    existing = [e for s in ("active", "candidate") for e in await ls.by_status(user_id, s) if e["id"] != lesson_id]
    existing = [e for e in existing if e["status"] == "active" or e["id"] < lesson_id]  # 候选之间先来的算数
    why = l0(h, source_tools, existing, fixed_trial(ep))
    await ls.decide(lesson_id, "l0", {"ok": not why, "why": why}, "rejected" if why else None)
    return why
