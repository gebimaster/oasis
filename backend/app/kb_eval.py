"""知识库检索评测的指标（M6 第 7 步；O5 第 1 步加片段级指标）。纯函数，跑评测的脚本在 scripts/kb_eval.py。

用户的选择：① 文档级标注（改块大小 / 改块 id 算法后标注仍然有效）② hit@1、hit@3、MRR@5 + 干扰文档
③ 库外题单独算拒答 ④ 原句和口语混合。
hit_rate 和 MRR 一律截到 k（在全部结果上算的话，返回得越多分越高）。
"""

import re
import unicodedata

from app.kb_search import Fused


# ---- O5 第 1 步：片段级指标。标注是「答案片段 + 相关条目原文」，都是文档的子串，和切法无关 ----

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    """用户选 C：先 NFKC（全角数字、字母、空格统一成半角），再去掉所有空白。
    切块会吃掉空白（段落之间的 \\n\\n 变成 \\n，断句时 strip 掉句首句尾的空格），比对之前两边都要去掉。"""
    return _WS.sub("", unicodedata.normalize("NFKC", s))


def span_hit(hits: list[Fused], spans: list[str], k: int) -> float:
    """用户选「任一」：前 k 块里只要有一块含任一答案片段，就记 1。"""
    texts = [norm(f.hit.text) for f in hits[:k]]
    return 1.0 if any(norm(s) in t for s in spans for t in texts) else 0.0


def _overlap(chunk: str, entry: str) -> int:
    """块和相关条目共有的字数。两者都是同一篇文档里的连续片段（归一化后），所以交集只有三种：
    条目整个在块里、块整个在条目里（长条目被切成几块）、首尾相接的一段（重叠 carry 造成）。
    首尾相接按最长的前后缀匹配算，偶然相同的一两个字会多算；调用方先按 doc_id 过滤，只在同篇里比。"""
    if entry in chunk:
        return len(entry)
    if chunk in entry:
        return len(chunk)
    for n in range(min(len(chunk), len(entry)) - 1, 0, -1):
        if chunk.endswith(entry[:n]) or chunk.startswith(entry[-n:]):
            return n
    return 0


def context_precision(hits: list[Fused], entries: list[str], relevant_docs: list[str]) -> float | None:
    """用户选 B（按字）：返回内容里属于相关条目的字数 / 返回的总字数。合进来的邻近条目算噪声。
    没有返回任何块时返回 None（分母为 0，不是 0 分也不是满分）。"""
    total = sum(len(norm(f.hit.text)) for f in hits)
    if total == 0:
        return None
    ents = [norm(e) for e in entries]
    rel = sum(min(len(t), sum(_overlap(t, e) for e in ents))
              for f in hits if f.hit.doc_id in relevant_docs for t in [norm(f.hit.text)])
    return rel / total


def returned_chars(hits: list[Fused]) -> int:
    return sum(len(norm(f.hit.text)) for f in hits)


def production_cut(hits: list[Fused], threshold: float, top_k: int) -> list[Fused]:
    """生产口径（用户选 Q4=C 的一半）：rerank 分过阈值、取前 top_k。hits 要求已按 rerank 分降序。
    O5 复核后评测直接用生产参数调 search(keep_pool=True)，这个函数只用来自检（整池截出来 == 生产返回），
    以及拿结果文件里存的整池离线换阈值重算。整池必须来自生产参数的那次调用：top_k 不同，候选数
    max(4*top_k, 20) 就不同，选进 rerank 的块也不同。"""
    return [f for f in hits if f.rerank is not None and f.rerank >= threshold][:top_k]


def span_summary(rows: list[dict]) -> dict:
    """rows: [{spans, span_hit@1, span_hit@3, cp@3, chars@3, cp_prod?, chars_prod?}]，只统计有答案的题。
    precision 为 None 的题（一块都没返回）不进均值，单独报个数。"""
    pos = [r for r in rows if r.get("spans")]
    if not pos:
        return {"n": 0}
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    out = {"n": len(pos)}
    for key in ("span_hit@1", "span_hit@3", "cp@3", "chars@3", "cp_prod", "chars_prod"):
        if not any(key in r for r in pos):  # 没 rerank 的配置没有生产口径
            continue
        out[key] = mean([r[key] for r in pos if r.get(key) is not None])
        if key.startswith("cp"):
            out[f"{key}_none"] = sum(1 for r in pos if r.get(key) is None)
    return out


def doc_ranking(hits: list[Fused]) -> list[str]:
    """块排名 → 文档排名：每篇文档取它最靠前的块的位置。同一篇的多个块不能重复占名次。"""
    seen: list[str] = []
    for f in hits:
        if f.hit.doc_id not in seen:
            seen.append(f.hit.doc_id)
    return seen


def hit_at(docs: list[str], relevant: list[str], k: int) -> float:
    return 1.0 if set(docs[:k]) & set(relevant) else 0.0


def mrr_at(docs: list[str], relevant: list[str], k: int) -> float:
    """第一个相关文档排第 r 名（r ≤ k）记 1/r，前 k 名里没有记 0。"""
    for r, d in enumerate(docs[:k], start=1):
        if d in relevant:
            return 1 / r
    return 0.0


def summarize(rows: list[dict], ks=(1, 3), mrr_k: int = 5) -> dict:
    """rows: [{relevant, docs}]，只统计有答案的题（库外题没有「命中」可言）。"""
    pos = [r for r in rows if r["relevant"]]
    if not pos:
        return {"n": 0}
    out = {"n": len(pos)}
    for k in ks:
        out[f"hit@{k}"] = round(sum(hit_at(r["docs"], r["relevant"], k) for r in pos) / len(pos), 3)
    out[f"mrr@{mrr_k}"] = round(sum(mrr_at(r["docs"], r["relevant"], mrr_k) for r in pos) / len(pos), 3)
    return out


def threshold_table(pos_best: list[float | None], neg_max: list[float], thresholds: list[float]) -> list[dict]:
    """阈值校准。pos_best：每道库内题里相关文档的最高 rerank 分（相关文档没进候选池记 None）；
    neg_max：每道库外题所有候选的最高分。答对 = 相关块过线；拒对 = 全部候选都没过线。"""
    rows = []
    for t in thresholds:
        answered = sum(1 for s in pos_best if s is not None and s >= t)
        refused = sum(1 for s in neg_max if s < t)
        rows.append({"threshold": t, "pos_answered": f"{answered}/{len(pos_best)}",
                     "neg_refused": f"{refused}/{len(neg_max)}"})
    return rows


# ---- O8 第三步：校准集 / 测试集划分、bootstrap 置信区间 ----

def assign_splits(queries: list[dict]) -> list[str]:
    """按（库内/库外, style）分层，层内按问题文本的 sha256 排序后交替分配：calib, test, calib, ...
    只看问题文本，所以可复现、和文件里的顺序无关；新加一道题只会动它所在那一层后面的分配。
    阈值、RRF 权重、WEAK 这类要调的参数只在 calib 上选，test 只用来报数。"""
    import hashlib
    strata: dict[tuple, list[int]] = {}
    for i, q in enumerate(queries):
        strata.setdefault((bool(q["relevant"]), q["style"]), []).append(i)
    out = [""] * len(queries)
    for idx in strata.values():
        for n, i in enumerate(sorted(idx, key=lambda i: hashlib.sha256(queries[i]["q"].encode()).hexdigest())):
            out[i] = "calib" if n % 2 == 0 else "test"
    return out


def wilson_ci(values: list[float], z: float = 1.96) -> tuple[float, float] | None:
    """命中率（0/1）的 95% Wilson 区间。O8 复核⑦：初版用百分位 bootstrap，二值数据全对时塌成 (1.0, 1.0)、
    小样本覆盖不足；Wilson 在 10 道全对时给出约 0.72–1.0，更诚实。
    题少时区间很宽（10 道对 7 道约 0.40–0.89），这本身就是要报的事实：差一两道题的两个配置分不出高低。"""
    import math
    n = len(values)
    if not n:
        return None
    p = sum(values) / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, center - half), 3), round(min(1.0, center + half), 3)


def paired_diff_ci(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """两个配置在同一批题上的差（b - a）的 95% 区间：逐题配对、有放回重抽题目。分开算两个区间再看重不重叠，
    会把「同一道题两边都难」的相关性丢掉，区间偏宽；比较配置（改写开 / 关、权重）用这个。固定 seed。"""
    import random
    if not a or len(a) != len(b):
        return None
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    means = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
    return round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)


def terms_hit(hits: list[Fused], all_of: list) -> bool:
    """O8 第四步回放用：返回的块拼起来是否含 agent 评测题 answer_has 的全部关键词（每项是词或同义词列表）。
    agent 题没有片段标注，只能用答案关键词近似「答案在不在送进模型的内容里」。数字按完整数字比（15 不算命中 150）。"""
    text = norm("".join(f.hit.text for f in hits))

    def has(s: str) -> bool:
        s = norm(s)
        if s.replace(".", "").isdigit():
            return re.search(r"(?<![\d.])" + re.escape(s) + r"(?![\d])", text) is not None
        return s in text
    return all(any(has(s) for s in ([x] if isinstance(x, str) else x)) for x in all_of)


def bigram_jaccard(a: str, b: str) -> float:
    """两句话的字二元组 Jaccard，用来查评测集之间的近似重复（O8 复核⑥：只查完全相同会漏掉换个说法的同一道题）。"""
    g = lambda s: {s[i:i + 2] for i in range(len(s) - 1)}
    x, y = g(norm(a)), g(norm(b))
    return len(x & y) / len(x | y) if x | y else 0.0
