"""Agent 循环验证：用假模型代替真实 API，覆盖各个停止分支。"""

import asyncio
import copy

import anthropic
import httpx
import pytest

import httpx2

from app import agent_loop, budget, llm, tools, transfer  # noqa: F401  transfer：注册转账工具，出口表里的挂起用例要用
from app.agent_loop import repair_orphan_tool_uses, run_loop


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """O7：流中途重试的退避改成 0，测试不真等。"""
    monkeypatch.setattr(agent_loop, "RETRY_DELAYS", (0, 0))


def fake_model(script):
    """script 是每一轮的 (content, stop_reason[, usage])，或一个要抛出的异常；同时记录每轮收到的参数。"""
    calls = []

    async def stream(**kwargs):
        # 存深拷贝快照：run_loop 每轮传的是同一个 tool_defs 列表对象，按引用存的话「tools 不变」恒真（O6 复核②）
        calls.append(copy.deepcopy(kwargs))
        step = script[len(calls) - 1]
        if isinstance(step, Exception):
            raise step
        content, stop, *usage = step
        for b in content:
            if b["type"] == "text":
                yield {"type": "text_delta", "text": b["text"]}
        yield {"type": "final", "content": content, "stop_reason": stop} | ({"usage": usage[0]} if usage else {})

    return stream, calls


def run(messages, stream, **kw):
    async def collect():
        return [ev async for ev in run_loop(messages, model_stream=stream, **kw)]
    return asyncio.run(collect())


USER = [{"role": "user", "content": "北京天气？"}]
THINK = {"type": "thinking", "thinking": "", "signature": "sig-1"}
CALL = {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "北京"}}


def test_tool_then_answer():
    stream, calls = fake_model([
        ([THINK, {"type": "text", "text": "我查一下"}, CALL], "tool_use"),
        ([{"type": "text", "text": "北京晴"}], "end_turn"),
    ])
    events = run(USER, stream)
    types = [e["type"] for e in events]
    assert types == ["text_delta", "message", "tool_start", "tool_end", "message",
                     "text_delta", "message", "run_end"]
    assert events[-1] == {"type": "run_end", "status": "completed", "stop_reason": "end_turn"}
    # 第 2 轮收到的历史：[0] 是会话上下文；thinking 块原样传回，tool_result 紧跟在后面
    second = calls[1]["messages"]
    assert second[2]["content"][0] == THINK
    assert second[3]["content"][0]["tool_use_id"] == "t1"
    assert calls[0]["system"] == calls[1]["system"]  # 每轮 system 一致


def test_max_turns_last_turn_sets_tool_choice_none():
    stream, calls = fake_model([([CALL], "tool_use")] * 2 + [([{"type": "text", "text": "总结"}], "end_turn")])
    events = run(USER, stream, max_turns=3)
    assert [c["tool_choice"] for c in calls] == [{"type": "auto"}, {"type": "auto"}, {"type": "none"}]
    assert calls[0]["system"] == calls[2]["system"]  # 最后一轮的提示进尾部，不动 system
    assert "工具调用上限" in calls[2]["messages"][-1]["content"][-1]["text"]
    assert events[-1]["stop_reason"] == "max_turns" and events[-1]["status"] == "completed"


def test_refusal_does_not_run_tools():
    stream, _ = fake_model([([CALL], "refusal")])
    events = run(USER, stream)
    assert "tool_start" not in [e["type"] for e in events]
    assert events[-1] == {"type": "run_end", "status": "failed", "stop_reason": "refusal"}


def test_invalid_json_retries_same_turn():
    attempts = []

    async def flaky(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("bad json")
            yield  # 让它成为 async generator
        yield {"type": "final", "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}

    events = run(USER, flaky, max_turns=1)
    assert len(attempts) == 2  # 重发了一次
    assert events[-1]["stop_reason"] == "max_turns"  # 重试不占轮数：仍是第 1 轮（也是最后一轮）


def test_repair_orphan_tool_use():
    history = USER + [{"role": "assistant", "content": [CALL]}, {"role": "user", "content": "还在吗"}]
    fixed = repair_orphan_tool_uses(history)
    [fill] = fixed[2]["content"]  # O4：补的结果单独一条，不拼进用户消息（否则「当前提问」认不出来）
    assert fill["tool_use_id"] == "t1" and fill["is_error"]
    assert fixed[3] == {"role": "user", "content": "还在吗"}
    assert history[2]["content"] == "还在吗"  # 没改原数据
    # 下一条已有部分结果：补进同一条（一次调用的结果必须全在紧接着的那条里）
    partial = USER + [{"role": "assistant", "content": [CALL, CALL2]},
                      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "晴"}]}]
    fixed = repair_orphan_tool_uses(partial)
    assert len(fixed) == 3 and [b["tool_use_id"] for b in fixed[2]["content"]] == ["t2", "t1"]


# ---------------- O3：出口表 ----------------
# 每个出口一行 (status, stop_reason)。O4/O6/O7 新增的 stop_reason 都往这里加行。

TEXT = ([{"type": "text", "text": "答复"}], "end_turn")
PAY = {"type": "tool_use", "id": "p1", "name": "transfer", "input": {"to": "alice", "amount": 500}}
CUT = {"type": "tool_use", "id": "t9", "name": "get_weather", "input": {}}  # 被截断的入参
BIG = {"input_tokens": 1000}
CALL2 = {**CALL, "id": "t2"}  # 同一个 run 里 id 不能重复，否则孤儿会被前一轮的结果「配上」

def api_error(cls, status, body):
    req = httpx.Request("POST", "https://api.deepseek.com/anthropic/v1/messages")
    sdk = cls(f"Error code: {status} - {body}", response=httpx.Response(status, request=req), body=body)
    try:  # 和 provider 一样包一层（raise LLMError.wrap(e) from e），循环只 except LLMError
        raise llm.LLMError.wrap(sdk, "anthropic-messages") from sdk
    except llm.LLMError as wrapped:
        return wrapped


# DeepSeek Anthropic 兼容接口的真实超窗错误体（O4 采集，一次）：OpenAI 格式，code 是笼统的 invalid_request_error
DEEPSEEK_PTL = {"error": {"message": "This model's maximum context length is 1048576 tokens. However, you requested "
                                     "1078571 tokens (694571 in the messages, 384000 in the completion). Please reduce "
                                     "the length of the messages or completion. (request_id: 43d9805f)",
                          "type": "invalid_request_error", "param": None, "code": "invalid_request_error"}}


def ptl():
    return api_error(anthropic.BadRequestError, 400, DEEPSEEK_PTL)


def net_error(mid_stream=True):
    """O7：流开始以后断网。provider 把 httpx2 的原始异常包成 mid_stream 的 LLMError（见 test_llm_providers）。"""
    try:
        raise llm.LLMError.wrap(httpx2.ReadError("connection reset"), "anthropic-messages",
                                mid_stream=mid_stream) from httpx2.ReadError("connection reset")
    except llm.LLMError as e:
        return e


def cut(i, inp=None):
    return {"type": "tool_use", "id": f"c{i}", "name": "get_weather", "input": {} if inp is None else inp}


def say(text, stop="end_turn"):
    return ([{"type": "text", "text": text}], stop)


@pytest.fixture
def fast_tools():
    """O6：两个不 sleep 的测试工具。lookup 每次都返回同一句；poll_job 声明了 polling（轮询类，阈值放宽）。"""
    async def lookup(args):
        return "没有找到"

    async def poll(args):
        return "任务还在运行"
    for name, polling, handler in (("lookup", False, lookup), ("poll_job", True, poll)):
        tools.register(tools.Tool(name=name, description="测试用", input_schema={"type": "object"}, handler=handler,
                                  permission=tools.PUBLIC, risk=tools.LOW, polling=polling))
    yield
    tools.unregister("lookup")
    tools.unregister("poll_job")


@pytest.fixture
def failing_tools():
    """O6 第二步：down / down2 总是报依赖不可用；flaky 不可用、成功、不可用轮流；
    poll_down 是轮询工具、也总是不可用；bad 是普通业务错误（参数不对），不算依赖不可用。"""
    state = {"flaky": 0}

    async def down(args):
        raise tools.Unavailable("UNAVAILABLE server unavailable")

    async def flaky(args):
        state["flaky"] += 1
        if state["flaky"] % 2:
            raise tools.Unavailable("UNAVAILABLE server unavailable")
        return "ok"

    async def bad(args):
        raise ValueError("没有这个城市")
    specs = [("down", down, False), ("down2", down, False), ("flaky", flaky, False),
             ("poll_down", down, True), ("bad", bad, False)]
    for name, handler, polling in specs:
        tools.register(tools.Tool(name=name, description="测试用", input_schema={"type": "object"}, handler=handler,
                                  permission=tools.PUBLIC, risk=tools.LOW, polling=polling))
    yield
    for name, *_ in specs:
        tools.unregister(name)


def rep(i, args=None, name="lookup"):
    """第 i 次调用：id 各不相同（孤儿断言要求），入参默认完全相同。"""
    return {"type": "tool_use", "id": f"r{i}", "name": name, "input": {"q": "火星"} if args is None else args}


REPEAT5 = [([rep(i)], "tool_use") for i in range(1, 6)]


def recording_compactor(result=lambda history, summary: ("摘要", history[-1:])):
    calls = []

    async def compactor(history, summary, room, measure):
        calls.append(len(history))
        return result(history, summary)
    return compactor, calls


EXITS = [
    # id, 脚本, run_loop 参数, 期望 (status, stop_reason), 期望调用模型的次数
    ("end_turn", [TEXT], {}, ("completed", "end_turn"), 1),
    ("end_turn_with_tool_use", [([CALL], "end_turn"), TEXT], {}, ("completed", "end_turn"), 2),  # 以内容为准：执行
    ("max_turns", [([CALL], "tool_use"), ([CALL2], "tool_use"), TEXT], {"max_turns": 3}, ("completed", "max_turns"), 3),
    ("budget", [([CALL], "tool_use", BIG), TEXT], {"context_limit": 500}, ("completed", "budget"), 2),
    ("awaiting_approval", [([CALL, PAY], "tool_use")], {}, ("completed", "awaiting_approval"), 1),
    ("api_error", [llm.LLMError("boom")], {}, ("failed", "api_error"), 1),
    ("invalid_tool_json", [ValueError("bad json")] * 3, {}, ("failed", "invalid_tool_json"), 3),
    ("refusal_with_tool_use", [([CALL], "refusal")], {}, ("failed", "refusal"), 1),
    # O7 截断续写：不占轮数，三种截断共用 MAX_CONTINUATIONS=3；截断的工具调用一个都不执行，当场补拒绝结果
    ("max_tokens_with_tool_use", [([CUT], "max_tokens"), TEXT], {}, ("completed", "end_turn"), 2),
    ("max_tokens_text_continued", [say("前半", "max_tokens"), say("后半")], {}, ("completed", "end_turn"), 2),
    ("max_tokens_thinking_only", [([THINK], "max_tokens"), TEXT], {}, ("completed", "end_turn"), 2),
    ("max_tokens_limit", [say("又被截", "max_tokens")] * 4, {}, ("failed", "max_tokens"), 4),
    ("max_tokens_tool_use_limit", [([cut(i)], "max_tokens") for i in range(4)], {}, ("failed", "max_tokens"), 4),
    ("max_tokens_on_last_turn", [([CALL], "tool_use"), say("半", "max_tokens"), TEXT], {"max_turns": 2},
     ("completed", "max_turns"), 3),  # 续写不占轮数：收尾轮被截断，续写仍是收尾轮
    # O7 空回复：提醒一次（不占轮数），还空才 failed
    ("empty_after_tool", [([CALL], "tool_use"), ([], "end_turn"), TEXT], {}, ("completed", "end_turn"), 3),
    ("empty_response", [([CALL], "tool_use"), ([], "end_turn"), ([THINK], "end_turn")], {},
     ("failed", "empty_response"), 3),
    # O7 流中途断网：SDK 不重试的那段由循环重试 2 次；发请求阶段的（SDK 已重试过）不叠加
    ("network_mid_stream_retried", [net_error(), TEXT], {}, ("completed", "end_turn"), 2),
    ("network_mid_stream_exhausted", [net_error()] * 3, {}, ("failed", "network"), 3),
    ("network_before_stream", [net_error(mid_stream=False)], {}, ("failed", "network"), 1),
    # 收尾轮违约：补「未执行」，再给一次纯文本机会（不算轮数）；还违约才 failed
    ("final_turn_violation_then_text", [([CALL], "tool_use"), TEXT], {"max_turns": 1}, ("completed", "max_turns"), 2),
    ("final_turn_end_turn_with_tool_use", [([CALL], "end_turn"), TEXT], {"max_turns": 1}, ("completed", "max_turns"), 2),
    ("final_turn_violation_twice", [([CALL], "tool_use"), ([CALL2], "tool_use")], {"max_turns": 1}, ("failed", "final_turn_violation"), 2),
    ("budget_violation_then_text", [([CALL], "tool_use", BIG), ([CALL2], "tool_use"), TEXT], {"context_limit": 500},
     ("completed", "budget"), 3),
    # pause_turn：原样续写，不算轮数；连续超过上限才停
    ("pause_turn_then_end", [([{"type": "text", "text": "半"}], "pause_turn"), TEXT], {}, ("completed", "end_turn"), 2),
    ("pause_turn_limit", [([{"type": "text", "text": "半"}], "pause_turn")] * 4, {}, ("failed", "pause_turn_limit"), 4),
    # O4 超窗：每一轮只恢复一次（强制压缩，压不下来再临时截窗口）；恢复不了或恢复后仍超窗都是 context_overflow
    ("context_overflow_recovered", [ptl(), TEXT], {"compactor": recording_compactor()[0]}, ("completed", "end_turn"), 2),
    ("context_overflow", [ptl(), ptl()], {"compactor": recording_compactor()[0]}, ("failed", "context_overflow"), 2),
    # 没有压缩器、只有一条提问：截不出更小的请求，原样重发必然同样 400，所以不重放
    ("context_overflow_unrecoverable", [ptl()], {}, ("failed", "context_overflow"), 1),
    ("rate_limited", [api_error(anthropic.RateLimitError, 429, {"error": {"message": "slow"}})], {},
     ("failed", "rate_limited"), 1),
    ("server_error", [api_error(anthropic.InternalServerError, 500, {"error": {"message": "x"}})], {},
     ("failed", "server_error"), 1),
    # O6 第一步：同一调用连续 5 次（第 5 次照常执行），下一轮复用收尾轮强制总结，判 failed
    ("repeated_action", REPEAT5 + [TEXT], {}, ("failed", "repeated_action"), 6),
    # 强制总结轮违约走 O3 的兜底：补未执行、再给一次纯文本机会，还违约才 final_turn_violation
    ("repeated_action_violation_twice", REPEAT5 + [([rep(6)], "tool_use"), ([rep(7)], "tool_use")], {},
     ("failed", "final_turn_violation"), 7),
    # O6 第二步：同一工具本 run 累计 2 次依赖不可用（每次换参数也算），下一轮强制总结，判 completed
    ("tool_unavailable", [([rep(1, {"q": 1}, "down")], "tool_use"), ([rep(2, {"q": 2}, "down")], "tool_use"), TEXT],
     {}, ("completed", "tool_unavailable"), 3),
]


def orphan_ids(history: list[dict]) -> set[str]:
    """修补函数会补结果的那些 tool_use id：补出来的就是孤儿。"""
    def answered(msgs):
        return {b.get("tool_use_id") for m in msgs if isinstance(m["content"], list) for b in m["content"]
                if b.get("type") == "tool_result"}
    return answered(repair_orphan_tool_uses(history)) - answered(history)


@pytest.mark.parametrize("script, kw, expected, n_calls", [e[1:] for e in EXITS], ids=[e[0] for e in EXITS])
def test_exit_table(script, kw, expected, n_calls, fast_tools, failing_tools):
    tools.current.set(tools.ToolContext(permissions=frozenset({"weather:read", "transfer:write"})))
    stream, calls = fake_model(script)
    events = run(USER, stream, **kw)
    end = events[-1]
    assert end["type"] == "run_end" and [e["type"] for e in events].count("run_end") == 1
    # 配对不变量（先断言它）：出口之后的历史（循环吐出的 message 事件）没有孤儿。
    # 挂起是唯一例外：整批结果（含已执行的）都在审批记录里，续跑时拼成一条 tool_result 写回（M5）
    history = USER + [e["message"] for e in events if e["type"] == "message"]
    assert orphan_ids(history) == {a["tool_use_id"] for a in end.get("approvals", [])}
    assert (end["status"], end["stop_reason"]) == expected
    assert len(calls) == n_calls


def test_final_turn_retry_keeps_tools_and_sees_rejection():
    """收尾轮违约后的重试：tools 字节不变（缓存）、仍禁止调用、能看到「未执行」的结果，工具没被执行。"""
    stream, calls = fake_model([([CALL], "tool_use"), TEXT])
    events = run(USER, stream, max_turns=1)
    assert calls[1]["tools"] == calls[0]["tools"]
    assert calls[1]["tool_choice"] == {"type": "none"}
    rejected = [b for b in calls[1]["messages"][-1]["content"] if b.get("type") == "tool_result"]
    assert rejected[0]["tool_use_id"] == "t1" and rejected[0]["is_error"]
    assert "tool_start" not in [e["type"] for e in events]


def test_compactions_capped_per_run():
    """O4：连续 4 轮都高于触发线，一个 run 最多压 MAX_COMPACTIONS 次，用完照发（超窗交给被动恢复）。"""
    tools.current.set(tools.ToolContext(permissions=frozenset({"weather:read"})))
    attempts = []

    async def compactor(*args):
        attempts.append(1)
        return None  # 压不下来：每轮都还在线上
    stream, calls = fake_model([([CALL], "tool_use"), ([CALL2], "tool_use"), ([{**CALL, "id": "t3"}], "tool_use"), TEXT])
    events = run(USER, stream, compactor=compactor, limits={"trigger": 0, "target": 1})
    assert len(calls) == 4 and events[-1]["stop_reason"] == "end_turn"
    assert len(attempts) == 2


def test_trigger_counts_whole_request():
    """O4 复核补：没有 usage 时，触发线估的是拼好的整份请求（摘要在 messages 里，不在 history 里）。"""
    big = "旧摘要的内容" * 4000
    compactor, attempts = recording_compactor(result=lambda h, s: None)
    stream, _ = fake_model([TEXT])
    run(USER, stream, summary=big, compactor=compactor,
        limits={"trigger": budget.estimate_text(big) // 2, "target": 10**9})
    assert len(attempts) == 1


def test_forced_compaction_not_capped():
    """O4 复核补：超窗时的强制压缩不占 MAX_COMPACTIONS，触发线的两次用完了，超窗仍先试压缩再退截窗口。"""
    tools.current.set(tools.ToolContext(permissions=frozenset({"weather:read"})))
    compactor, attempts = recording_compactor(result=lambda h, s: None)
    stream, _ = fake_model([([CALL], "tool_use"), ([CALL2], "tool_use"), ptl(), TEXT])
    events = run(USER, stream, compactor=compactor, limits={"trigger": 0, "target": 1})
    assert len(attempts) == 3 and events[-1]["stop_reason"] == "end_turn"


# ---------------- O4 第 2 步：错误分类、超窗被动恢复 ----------------

def test_classify_error():
    assert llm.classify_error(ptl()) == "context_overflow"                     # 只有文本认得出
    anthropic_ptl = {"type": "error", "error": {"type": "invalid_request_error", "message": "prompt is too long: 1M"}}
    assert llm.classify_error(api_error(anthropic.BadRequestError, 400, anthropic_ptl)) == "context_overflow"
    coded = {"error": {"message": "too big", "code": "context_length_exceeded"}}
    assert llm.classify_error(api_error(anthropic.BadRequestError, 400, coded)) == "context_overflow"
    assert llm.classify_error(api_error(anthropic.APIStatusError, 413, {"error": {"message": "x"}})) == "context_overflow"
    orphan = {"error": {"message": "tool_result block(s) provided without a matching tool_use", "code": "invalid_request_error"}}
    assert llm.classify_error(api_error(anthropic.BadRequestError, 400, orphan)) == "api_error"  # 400 不都是超窗
    assert llm.classify_error(api_error(anthropic.RateLimitError, 429, {"error": {"message": "slow"}})) == "rate_limited"
    assert llm.classify_error(api_error(anthropic.InternalServerError, 500, {"error": {"message": "x"}})) == "server_error"
    try:  # 外层包了一层（比如以后的网关、trace），原文只在 __cause__ 里
        try:
            raise ptl().__cause__  # 取 SDK 原始异常（ptl() 外面已经包了 LLMError）
        except anthropic.BadRequestError as e:
            raise anthropic.APIError("model call failed", httpx.Request("POST", "https://x"), body=None) from e
    except anthropic.APIError as wrapped:
        assert llm.classify_error(wrapped) == "context_overflow"


def test_overflow_forces_compaction_once_then_retries():
    """估算没超线（不会主动压缩），服务端却回了超窗：强制压缩 1 次、重发 1 次。"""
    compactor, attempts = recording_compactor()
    stream, calls = fake_model([ptl(), TEXT])
    events = run(USER, stream, compactor=compactor, limits={"trigger": 10**9, "target": 10**9})
    assert events[-1] == {"type": "run_end", "status": "completed", "stop_reason": "end_turn"}
    assert len(attempts) == 1 and len(calls) == 2
    assert next(e for e in events if e["type"] == "compacted")["reason"] == "overflow"


def test_overflow_recovery_once_per_turn_not_per_run():
    """第 1 轮超窗恢复了；第 2 轮是新的一轮，历史又长了，可以再恢复一次。"""
    tools.current.set(tools.ToolContext(permissions=frozenset({"weather:read"})))
    compactor, attempts = recording_compactor()
    stream, calls = fake_model([ptl(), ([CALL], "tool_use"), ptl(), TEXT])
    events = run(USER, stream, compactor=compactor, limits={"trigger": 10**9, "target": 10**9})
    assert events[-1]["stop_reason"] == "end_turn" and len(attempts) == 2 and len(calls) == 4


def test_overflow_falls_back_to_temporary_window():
    """压缩救不回来（这里没有压缩器）：临时截窗口，只留 Q + 保护区，不摘要。"""
    history = [{"role": "user", "content": "对比 3 个城市"}]
    for i in range(1, 4):
        history += [{"role": "assistant", "content": [{**CALL, "id": f"w{i}"}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"w{i}", "content": "大" * 500}]}]
    stream, calls = fake_model([ptl(), TEXT])
    events = run(history, stream)
    sent = calls[1]["messages"]
    assert [m["content"] for m in sent[1:3]] == [history[0]["content"], history[-2]["content"]]
    assert sent[3]["content"][0] == history[-1]["content"][0] and len(sent) == 4
    assert next(e for e in events if e["type"] == "compacted")["reason"] == "window"
    assert events[-1]["stop_reason"] == "end_turn"


def test_no_compaction_after_window():
    """O4 复核：截过窗口，内存里的 history 只剩 Q + 保护区；再压就会把没摘要过的原文越过水位，永远丢掉。
    所以截过窗口后本 run 不再调压缩器，再超窗就再截；下个 run 从库里重新加载完整原文再压。"""
    history = [{"role": "user", "content": "对比 3 个城市"}]
    for i in range(1, 4):
        history += [{"role": "assistant", "content": [{**CALL, "id": f"w{i}"}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"w{i}", "content": "大" * 500}]}]
    compactor, attempts = recording_compactor(result=lambda h, s: None)  # 压不下来，逼它截窗口
    stream, calls = fake_model([ptl(), ([{**CALL, "id": "w4"}], "tool_use"), ptl(), TEXT])
    events = run(history, stream, compactor=compactor, limits={"trigger": 0, "target": 10**9})
    assert len(attempts) == 2  # 第 1 轮：触发线一次 + 超窗强制一次；截过窗口后一次都不再调
    assert [e["reason"] for e in events if e["type"] == "compacted"] == ["window", "window"]
    assert events[-1]["stop_reason"] == "end_turn"


def test_rate_limit_does_not_compact():
    compactor, attempts = recording_compactor()
    stream, calls = fake_model([api_error(anthropic.RateLimitError, 429, {"error": {"message": "slow down"}})])
    events = run(USER, stream, compactor=compactor, limits={"trigger": 10**9, "target": 10**9})
    assert events[-1]["stop_reason"] == "rate_limited" and attempts == [] and len(calls) == 1


# ---------------- O6 第一步：动作指纹 + 连续重复的分级处置 ----------------

def tail(call) -> str:
    """这一次请求的运行时尾部（最后一条 user 消息的最后一个 text 块）。"""
    return call["messages"][-1]["content"][-1]["text"]


def guard_events(events):
    return [(e["action"], e["tool"], e["count"]) for e in events if e["type"] == "loop_guard"]


def test_repeat_nudges_at_3_and_forces_summary_after_5(fast_tools):
    """同一调用连续 5 次：第 3、4 次之后的请求尾部带提醒；第 5 次照常执行，下一轮禁止调工具、提示写明原因；
    system 和 tools 字节始终不变（提醒只进尾部，不动缓存前缀）；调模型 6 次，远少于 max_turns。"""
    stream, calls = fake_model(REPEAT5 + [TEXT])
    events = run(USER, stream)
    assert len(calls) == 6
    assert [c["tool_choice"]["type"] for c in calls] == ["auto"] * 5 + ["none"]
    assert all("连续" not in tail(c) for c in calls[:3])
    assert "连续 3 次" in tail(calls[3]) and "lookup" in tail(calls[3])
    assert "连续 4 次" in tail(calls[4])
    assert "连续 5 次" in tail(calls[5]) and "终止" in tail(calls[5]) and "不要编造" in tail(calls[5])
    assert all(c["system"] == calls[0]["system"] and c["tools"] == calls[0]["tools"] for c in calls)
    assert sum(e["type"] == "tool_end" for e in events) == 5  # 第 5 次也执行了（你选的 G4-B）
    assert guard_events(events) == [("nudge", "lookup", 3), ("nudge", "lookup", 4), ("finalize", "lookup", 5)]
    assert events[-1]["status"] == "failed" and events[-1]["stop_reason"] == "repeated_action"


def test_key_order_and_spelling_variants_count_as_same(fast_tools):
    """键顺序不同、1 和 1.0、首尾空白，都算同一个动作（去掉键排序这条变异要让它变红）。
    全角和大小写的规范化不在这里测（「火星」没有大小写之分），在 tests/test_loop_guard.py 里测（复核⑧）。"""
    variants = [{"city": "火星", "n": 1}, {"n": 1.0, "city": "火星"}, {"n": 1, "city": " 火星 "},
                {"n": 1, "city": "火星"}, {"city": "火星", "n": 1.0}]
    script = [([rep(i, v)], "tool_use") for i, v in enumerate(variants, 1)] + [TEXT]
    stream, calls = fake_model(script)
    events = run(USER, stream)
    assert events[-1]["stop_reason"] == "repeated_action" and len(calls) == 6


def test_streak_resets_when_call_is_absent(fast_tools):
    """A A B A A A：中间夹了 B，A 的计数清零，最后只到 3（提醒），不熔断。去掉清零的话 A 会数到 5。"""
    b = {"q": "金星"}
    script = [([rep(1)], "tool_use"), ([rep(2)], "tool_use"), ([rep(3, b)], "tool_use"),
              ([rep(4)], "tool_use"), ([rep(5)], "tool_use"), ([rep(6)], "tool_use"), TEXT]
    stream, calls = fake_model(script)
    events = run(USER, stream)
    assert events[-1] == {"type": "run_end", "status": "completed", "stop_reason": "end_turn"}
    assert "连续" not in tail(calls[5]) and "连续 3 次" in tail(calls[6])
    assert guard_events(events) == [("nudge", "lookup", 3)]


def test_stuck_call_hidden_in_varying_batch_is_caught(fast_tools):
    """按单个调用算（你选的 Q1-B）：每批都夹一个不同的调用，卡住的那个照样数到 5。按整批算就抓不住。"""
    script = [([rep(i), rep(100 + i, {"q": f"城市{i}"})], "tool_use") for i in range(1, 6)] + [TEXT]
    stream, calls = fake_model(script)
    events = run(USER, stream)
    assert events[-1]["stop_reason"] == "repeated_action" and len(calls) == 6


def test_duplicates_in_one_batch_count_each(fast_tools):
    """同一批里出现几次就算几次（你选的 F1-ii）：一批 5 个相同调用，当场到硬阈值，下一轮就收尾。"""
    stream, calls = fake_model([([rep(i) for i in range(1, 6)], "tool_use"), TEXT])
    events = run(USER, stream)
    assert len(calls) == 2 and calls[1]["tool_choice"] == {"type": "none"}
    assert events[-1]["stop_reason"] == "repeated_action"


def test_polling_tool_gets_double_thresholds(fast_tools):
    """声明了 polling 的工具阈值翻倍 (6, 10)：连续 9 次都不熔断，第 6 次之后才提醒；
    硬阈值 10 不小于 max_turns，实际由 max_turns 收尾（你选的 G2-A）。"""
    script = [([rep(i, name="poll_job")], "tool_use") for i in range(1, 10)] + [TEXT]
    stream, calls = fake_model(script)
    events = run(USER, stream)
    assert events[-1] == {"type": "run_end", "status": "completed", "stop_reason": "max_turns"}
    assert "连续" not in tail(calls[5]) and "连续 6 次" in tail(calls[6]) and "连续 9 次" in tail(calls[9])
    assert all(a == "nudge" for a, _, _ in guard_events(events))
    # 复核①：最后一轮（tool_choice none）不能再说「可以继续」，换成说明原因、要求直接回答的版本（你选的 B）
    assert calls[9]["tool_choice"] == {"type": "none"} and "可以继续" not in tail(calls[9])
    assert "不能再调用工具" in tail(calls[9])


def test_last_turn_nudge_does_not_contradict_final_note(fast_tools):
    """复核①：普通工具 max_turns=5，原样调 4 次后第 5 轮是收尾轮。尾部有「不要再调用工具」，
    重复提醒不能再叫它「换参数或换办法」，而是说明连续了几次、请基于已有结果回答。"""
    stream, calls = fake_model(REPEAT5[:4] + [TEXT])
    events = run(USER, stream, max_turns=5)
    last_tail = tail(calls[4])
    assert calls[4]["tool_choice"] == {"type": "none"} and "工具调用上限" in last_tail
    assert "连续 4 次" in last_tail and "不能再调用工具" in last_tail and "换参数" not in last_tail
    assert "换参数" in tail(calls[3])  # 不是最后一轮时照旧提醒换办法
    assert events[-1] == {"type": "run_end", "status": "completed", "stop_reason": "max_turns"}


def test_repeat_counts_unwrapped_call_tool(fast_tools):
    """经 call_tool 代理调用和直接调用是同一个动作：指纹用拆包后的真名和真参数。"""
    proxied = lambda i: {"type": "tool_use", "id": f"c{i}", "name": "call_tool",
                         "input": {"tool_name": "lookup", "arguments": {"q": "火星"}}}
    script = [([rep(1)], "tool_use"), ([proxied(2)], "tool_use"), ([rep(3)], "tool_use"),
              ([proxied(4)], "tool_use"), ([rep(5)], "tool_use"), TEXT]
    stream, calls = fake_model(script)
    events = run(USER, stream)
    assert events[-1]["stop_reason"] == "repeated_action"


# ---------------- O6 第二步：依赖不可用累计熔断 ----------------

def d(i, q, name="down"):
    return rep(i, {"q": q}, name)


def test_unavailable_trips_after_2_even_with_new_args(failing_tools):
    """同一工具每次换参数，服务一直挂着：累计 2 次就进收尾轮。提示点名工具、说明没返回数据、要求按三部分写、不要编造；
    调模型 3 次，远少于 max_turns；判 completed（服务挂了不是模型的错，你选的 R3）。"""
    stream, calls = fake_model([([d(1, 1)], "tool_use"), ([d(2, 2)], "tool_use"), TEXT])
    events = run(USER, stream)
    assert len(calls) == 3 and calls[2]["tool_choice"] == {"type": "none"}
    note = tail(calls[2])
    for phrase in ("down", "不可用", "没有返回任何数据", "不要编造", "已拿到", "没拿到", "下一步"):
        assert phrase in note, phrase
    assert "不可用" not in tail(calls[1])  # 第 1 次失败后不提示
    assert all(e["content"].startswith("[TOOL_ERROR] UNAVAILABLE") for e in events if e["type"] == "tool_end")
    assert guard_events(events) == [("unavailable", "down", 2)]
    assert events[-1]["status"] == "completed" and events[-1]["stop_reason"] == "tool_unavailable"


def test_ordinary_errors_do_not_count(failing_tools):
    """参数错误这类业务错误不算依赖不可用（你选的 S1-C）：失败 4 次照样往下走，由模型自己调整。"""
    stream, calls = fake_model([([d(i, i, "bad")], "tool_use") for i in range(1, 5)] + [TEXT])
    events = run(USER, stream)
    assert events[-1] == {"type": "run_end", "status": "completed", "stop_reason": "end_turn"} and len(calls) == 5


def test_unavailable_is_cumulative_success_does_not_reset(failing_tools):
    """本 run 累计（你选的 S2-b）：不可用、成功、不可用，第 3 轮之后熔断。改成连续计数的话中间那次成功会清零。"""
    stream, calls = fake_model([([d(i, i, "flaky")], "tool_use") for i in range(1, 4)] + [TEXT])
    events = run(USER, stream)
    assert events[-1]["stop_reason"] == "tool_unavailable" and len(calls) == 4
    # 复核③：中间那次真的成功过，提示不能说成「什么都没返回」，要让模型照常用已经拿到的结果（你选的 A）
    assert "照常使用" in tail(calls[3])


def test_unavailable_counted_per_tool(failing_tools):
    """按工具名累计（你选的 T1-甲）：两个不同的工具各挂 1 次，都没到 2，不熔断。"""
    stream, calls = fake_model([([d(1, 1)], "tool_use"), ([d(2, 1, "down2")], "tool_use"), TEXT])
    events = run(USER, stream)
    assert events[-1]["stop_reason"] == "end_turn" and len(calls) == 3


def test_two_unavailable_in_one_batch_trip_at_once(failing_tools):
    """同一批里同一工具挂 2 次，也是累计 2 次：下一轮就收尾。"""
    stream, calls = fake_model([([d(1, 1), d(2, 2)], "tool_use"), TEXT])
    events = run(USER, stream)
    assert events[-1]["stop_reason"] == "tool_unavailable" and len(calls) == 2


def test_polling_tool_unavailable_not_counted(failing_tools):
    """轮询工具的失败不计数（你选的 S3-B）：挂 3 次也不熔断。"""
    stream, calls = fake_model([([d(i, i, "poll_down")], "tool_use") for i in range(1, 4)] + [TEXT])
    events = run(USER, stream)
    assert events[-1]["stop_reason"] == "end_turn" and len(calls) == 4


def test_unavailable_wins_over_repeat_in_same_round(fast_tools, failing_tools):
    """同一轮两种熔断都到了：以不可用为准（你选的 R4-A），提示写服务不可用，判 completed/tool_unavailable。"""
    batch = [rep(i) for i in range(1, 6)] + [d(6, 1), d(7, 2)]
    stream, calls = fake_model([(batch, "tool_use"), TEXT])
    events = run(USER, stream)
    assert events[-1]["status"] == "completed" and events[-1]["stop_reason"] == "tool_unavailable"
    assert "不可用" in tail(calls[1]) and "已经连续 5 次" not in tail(calls[1])
