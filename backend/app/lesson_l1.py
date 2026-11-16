"""P4 第 6 步：L1 门禁的纯函数（挑题 + 判定）。真正跑模型的部分在 scripts/lesson_eval.py，这里不碰网络和库。

三类检验题（FGSE），都从固定题库里挑（用户选 A），题库每道题的基线结果只跑一次、各候选共用：
- 目标题 target：用到这条经验管的工具、基线做不对的。经验要让它变对。
- 回归题 regress：用到这条经验管的工具、基线已经会做的。经验不许把它弄坏。
- 证伪题 falsify：不用这条经验的工具、基线会做、但读取时最可能把这条经验挑进来的（前提文字最像）。
  这是 FGSE 的「诱饵 / 前提不成立」题：经验被注入了，但前提不成立，模型应该当没看见。
另加源任务回放：把源 run 的第一句用户原话再问一遍，用经验自己的「验证方式」判分。

提交规则（用户选 A，严）：至少 1 道变好；证伪题、回归题一道都不许变坏；挑不出证伪题就不提交（失败关闭）。
"""

from app.memory_search import terms

PER_ROLE = 2


def score(lesson: dict, text: str, sim: float = 0.0) -> float:
    """读取时「会不会被挑进来」的近似：向量相似度为主，关键词命中数做次序。"""
    hay = lesson["precondition"] + lesson["behavior"] + " ".join(lesson["tools"])
    return sim + 0.01 * sum(t in hay for t in terms(text))


def pick(lesson: dict, bank: list[dict], sims: dict[str, float]) -> dict[str, list[dict]]:
    """bank 每道题带 tools（要用到的工具）和 base（基线 3 次的多数：True / False / None 平票）。sims = 题 id → 相似度。"""
    mine = set(lesson["tools"])
    ok = [p for p in bank if p["base"] is not None]
    uses = [p for p in ok if mine & set(p["tools"])]
    others = sorted((p for p in ok if not mine & set(p["tools"]) and p["base"]),
                    key=lambda p: score(lesson, p["input"], sims.get(p["id"], 0.0)), reverse=True)
    return {"target": [p for p in uses if not p["base"]][:PER_ROLE],
            "regress": [p for p in uses if p["base"]][:PER_ROLE],
            "falsify": others[:PER_ROLE]}


def majority(flags: list) -> bool | None:
    got = [f for f in flags if f is not None]
    return None if not got or sum(got) * 2 == len(got) else sum(got) * 2 > len(got)


def decide(rows: list[dict], use_falsify: bool = True) -> dict:
    """rows = [{probe, role: source/target/regress/falsify, base: bool, with: [每次通过?]}]。
    use_falsify=False 是验收里的消融：不看证伪题（也不要求有证伪题），其余照旧。"""
    gain, broke, why = [], [], []
    for r in rows:
        w = majority(r["with"])
        if r["role"] in ("source", "target") and r["base"] is False and w is True:
            gain.append(r["probe"])
        if r["role"] == "regress" or (use_falsify and r["role"] == "falsify"):
            if r["base"] is True and w is not True:  # 平票也算没守住：宁严勿松
                broke.append(f"{r['role']}:{r['probe']}")
    n_falsify = sum(r["role"] == "falsify" for r in rows)
    if not gain:
        why.append("没有一道题因为它变好")
    if broke:
        why.append(f"变坏了：{'、'.join(broke)}")
    if use_falsify and n_falsify == 0:
        why.append("挑不出证伪题，无法检验前提")
    summary = f"收益 {len(gain)}、变坏 {len(broke)}、证伪题 {n_falsify if use_falsify else '不看'}"
    return {"commit": not why, "gain": gain, "broke": broke, "why": why, "summary": summary}
