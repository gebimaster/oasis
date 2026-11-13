"""O8 第二步：查询改写器。kb_search.search 只在第一轮弱的时候调用它（见 kb_search._with_rewrites）。

为什么用模型、不用同义词表：同义词表要人照着语料写（「干活 → 加班」「换假 → 调休」），写出来正好贴合这几篇文档，
评测会虚高，换一批文档就失效。模型改写是通用的，代价是一次调用（约 1–3 秒，一次几分之一美分），
所以只在第一轮弱时才改，正常命中的查询不花这笔钱。

为什么默认关（KB_REWRITE=1 才开）：改写有回归风险（可能把原来答对的题改坏），要先在评测上同时数
「改好几题、改坏几题」再决定默认开不开。生产里 query 本来由 Agent 模型写，和改写器有重叠；
这里保留原句那一路，所以改写只会多找回候选，不会丢掉原句能找到的。
"""

import json
import os
import re
from collections import OrderedDict
from pathlib import Path

from app import llm

PROMPT = ("你是检索查询改写器。把用户的问题改写成 {n} 条适合在内部制度、手册类文档里检索的查询：\n"
          "- 把口语、俚语、缩略说法换成文档里常用的书面说法，补上问题隐含的主题词；\n"
          "- 每条都要保留原问题的意思，不要扩大或缩小范围；\n"
          "- 不要回答问题，不要编造数字、金额、日期；\n"
          "- 只输出查询本身，每行一条，不要编号、不要解释。")


_MARK = re.compile(r"^\s*(?:[-*•·]|\d+[.、)）]|[（(]\d+[)）])\s*")


def parse(raw: str, n: int) -> list[str]:
    """一行一条；去掉模型常加的编号、项目符号、引号，丢掉「以下是改写：」这类以冒号结尾的前言。
    O8 复核⑨：初版用 lstrip 去编号，会把「10 天以上病假」的 10 一起剥掉。"""
    out = []
    for line in raw.splitlines():
        q = _MARK.sub("", line).strip().strip("\"'“”「」")
        if q and len(q) <= 200 and not q.endswith(("：", ":")):
            out.append(q)
    return out[:n]


class LLMRewriter:
    """一次模型调用产出 n 条改写。进程内 LRU 缓存：同一句话重复问不再花钱。"""

    def __init__(self, n: int = 2, model: str | None = None, cache_size: int = 256, complete=None):
        self.n, self.model = n, model
        self.name = f"llm:{model or os.getenv('MODEL', 'default')}"
        self._complete = complete or llm.complete
        self._cache: OrderedDict[str, list[str]] = OrderedDict()
        self._size = cache_size

    async def rewrite(self, query: str) -> list[str]:
        if query in self._cache:
            self._cache.move_to_end(query)
            return self._cache[query]
        # max_tokens 给小一点：改写几十个字就够；默认模型会先 thinking，留 1500 给它（实测延迟还没量）
        raw = await self._complete(system=PROMPT.format(n=self.n), prompt=query, model=self.model, max_tokens=1500)
        got = parse(raw, self.n)
        self._cache[query] = got
        if len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return got


class DiskCachedRewriter:
    """评测用：改写结果落盘，重跑评测不重复花钱（PLAN 的约束）。key 里带改写器名字，换模型不会误用旧缓存。"""

    def __init__(self, inner, path: Path):
        self.inner, self.path = inner, Path(path)
        self.name = f"cached:{inner.name}"
        self._data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    async def rewrite(self, query: str) -> list[str]:
        key = f"{self.inner.name}\t{query}"
        if key not in self._data:
            self._data[key] = await self.inner.rewrite(query)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
        return self._data[key]


_default = None


def default() -> LLMRewriter | None:
    """生产用的改写器：KB_REWRITE=1 才开（见模块说明）。"""
    global _default
    if os.getenv("KB_REWRITE") != "1":
        return None
    if _default is None:
        _default = LLMRewriter(model=os.getenv("KB_REWRITE_MODEL") or None)
    return _default
