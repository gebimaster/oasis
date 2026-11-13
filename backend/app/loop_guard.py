"""O6 循环熔断。循环不碰数据库，这里也不碰，只做计数。
第一步 RepeatGuard：动作指纹 + 连续重复的分级处置；第二步 UnavailableGuard：同一工具累计依赖不可用（在文件后半）。

指纹按单个调用算：工具名原样 + 入参规范化后取哈希。规范化只动值、不动键：
- 字符串：NFKC（全角转半角）→ casefold（忽略大小写）→ 去首尾空白，中间的空白不动；
- 整数值的浮点转成 int（1.0 → 1）。方向不能反过来：int 转 float 会让 2**53+1 和 2**53 撞成一个；
- 布尔保持布尔（Python 里 True == 1，转成数字就和 1 撞了）；
- 键只排序不规范化：{"a": 1, "A": 2} 规范化键会合成一个键、丢信息。
接受的误判：区分大小写的值（沙箱路径 /workspace/A.txt 和 a.txt、代码里的变量名）会被当成同一个动作。

计数：严格连续。一个指纹这一轮出现几次就加几次（同一批里的重复也算），这一轮没出现就清零。
发过提醒不清零，硬阈值数的是总数。不看结果：轮询工具结果在变也照样计数，所以靠 Tool.polling 放宽阈值。
一个 run 一个实例；审批续跑、用户发新消息都是新 run，计数自然从零开始。
"""

import hashlib
import json
import unicodedata
from collections import Counter
from typing import Any

SOFT_REPEATS = 3    # 连续第 3 次之后，下一次请求的尾部带提醒
HARD_REPEATS = 5    # 连续第 5 次照常执行，之后进收尾轮（强制总结），run 判 failed/repeated_action
POLLING_FACTOR = 2  # 轮询工具两个阈值都乘它：(6, 10)。10 不小于 MAX_TURNS，主 run 里实际由 max_turns 收尾


def normalize(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, float):
        return int(v) if v.is_integer() else v
    if isinstance(v, str):  # casefold 之后再做一次 NFKC（Unicode 推荐的 NFKC_Casefold 做法）
        return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", v).casefold()).strip()
    if isinstance(v, dict):
        return {k: normalize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [normalize(x) for x in v]
    return v


def fingerprint(name: str, args: Any) -> str:
    canonical = json.dumps([name, normalize(args)], ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), default=str)
    # 复核甲：入参里可能有 JSON 合法的孤立代理字符（"\ud800"），严格编码会抛异常冒出 run_loop
    return hashlib.sha256(canonical.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def thresholds(polling: bool) -> tuple[int, int]:
    f = POLLING_FACTOR if polling else 1
    return SOFT_REPEATS * f, HARD_REPEATS * f


class RepeatGuard:
    """observe 每批调用（执行之后）。calls 里每个元素要有 name、args、tool（tools.Call，call_tool 已拆包）。"""

    def __init__(self) -> None:
        self.streaks: dict[str, int] = {}
        self._info: dict[str, tuple[str, bool]] = {}  # 指纹 → (工具名, 是否轮询)

    def observe(self, calls: list) -> None:
        now: Counter[str] = Counter()
        info = {}
        for c in calls:
            fp = fingerprint(c.name, c.args)
            now[fp] += 1
            info[fp] = (c.name, bool(c.tool is not None and c.tool.polling))
        self.streaks = {fp: self.streaks.get(fp, 0) + n for fp, n in now.items()}  # 没出现的指纹不带过来 = 清零
        self._info = info

    def _rows(self):
        for fp, n in self.streaks.items():
            name, polling = self._info[fp]
            yield name, n, polling, *thresholds(polling)

    def nudges(self) -> list[tuple[str, int, bool]]:
        """到了软阈值、还没到硬阈值的：(工具名, 连续次数, 是否轮询)。"""
        return [(name, n, polling) for name, n, polling, soft, hard in self._rows() if soft <= n < hard]

    def tripped(self) -> tuple[str, int] | None:
        """到了硬阈值的第一个：(工具名, 连续次数)。"""
        return next(((name, n) for name, n, _, _, hard in self._rows() if n >= hard), None)


# ---------------- 第二步：依赖不可用累计熔断 ----------------

UNAVAILABLE_LIMIT = 2  # 同一个工具本 run 累计 2 次「依赖不可用」就熔断：服务挂了基本不会在几秒内自己好


class UnavailableGuard:
    """按工具名（call_tool 拆包后的真名）累计本 run 里 error_code == UNAVAILABLE 的次数。
    - 只认工具主动抛的 tools.Unavailable：参数错误、业务错误、用户代码报错不计（它们是模型能自己调整的）；
    - 本 run 累计，中间成功了也不清零（服务时好时坏照样抓得到）；同一批里挂几次就算几次；
    - 轮询工具（Tool.polling）的失败不计；
    - 一个 run 一个实例：用户下一条消息、审批续跑都从零开始，挂着的服务会被重新试一次。
    已知漏洞：按工具名数，同一个 MCP server 下的多个工具各算各的，模型换着调能多撞几次。"""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def observe(self, calls: list, code: str) -> None:
        for c in calls:
            if c.error_code == code and not (c.tool is not None and c.tool.polling):
                self.counts[c.name] += 1

    def tripped(self) -> tuple[str, int] | None:
        return next(((name, n) for name, n in self.counts.items() if n >= UNAVAILABLE_LIMIT), None)
