"""O8 第一步：引用校验。不调模型，只用代码核对「回答里标的出处」和「本次 kb_search 真正返回过的块」。

忠实度评委回答的是「这句话原文里有没有依据」；这里回答的是「它标的那一段是不是依据所在的那一段」。
两者分开，才能把「检索对了、回答不对」拆成：没依据（生成编造）和有依据但引错了（归因错）。

出处格式由 kb_tool.CITE 规定：【标题 第N段】或【标题 · 小节 第N段】。模型也常写成「《标题》第N段」「第1、3段」
「【A 第1段；B 第3段】」，一并认。块头格式由 kb_tool.render 产生：「--- 《标题》第N段」或「--- 《标题》 · 小节 · 第N段」。

一条引用的判定（status）：
  ok              引的块是本次返回过的，且它挂着的那段话里的「锚点」都在这一块里
  wrong_block     锚点在本次返回的别的块里、不在它引的这一块里（引错段）
  wrong_para      标题对得上，但这个段号本次没返回过
  unknown_source  标题和本次返回的任何块都对不上
  unchecked       块对得上，但那段话里没有（能在证据里找到的）锚点，代码判断不了支持与否（不进正确率分母）
锚点 = 数字（200、22、0.5）和字母数字代号（E1024），按完整数字匹配（20 不算命中 200）；列表序号「1.」不算锚点。
锚点在所有返回块里都找不到时，那是忠实度的问题（数字没依据），不算引用错误，记 anchor_not_in_evidence。

一条引用挂哪段话（_claim_span，正确率和覆盖率都用它，O8 复核②：两处原来各算各的）：
  - 默认：出处跟在它支撑的话后面，取「句首或同句上一条引用之后」到这条引用
  - 紧跟在句末标点后面（「……600 元。【X 第3段】」）：挂前一句
  - 引用前只有「根据 / 依据 / 来源 / 按 / 见」这类词：挂后面；后面紧跟「：」换行的，挂到这一段结束（列表）；
    后面什么都没有（「来源：【A】【B】」放在末尾）：挂前一段
"""

import re
from dataclasses import dataclass, field

from app.kb_eval import norm

TITLE_CHARS, SECTION_CHARS = 60, 30  # 和 kb_tool 的块头截断一致（kb_tool 依赖本模块之外，这里不 import 它，避免环）
_CN = {c: i for i, c in enumerate("零一二三四五六七八九十")}
_NUMS = r"((?:\d+|[一二三四五六七八九十]{1,3})(?:\s*[、,，和及]\s*(?:\d+|[一二三四五六七八九十]{1,3}))*)"
_BRACKET = re.compile(r"【([^【】\n]{1,200}?)】")
_IN_BRACKET = re.compile(r"(.+?)\s*第\s*" + _NUMS + r"\s*段")
_GUILLEMET = re.compile(r"《([^《》\n]{1,80})》\s*(?:[·・]\s*([^·・\n第]{1,40}?)\s*[·・]?\s*)?第\s*" + _NUMS + r"\s*段")
_HEADER = re.compile(r"^--- 《([^《》\n]*)》\s*(?:·\s*(.*?)\s*·\s*)?第(\d+)段$", re.M)
_ANCHOR = re.compile(r"[A-Za-z]+\d+[A-Za-z\d]*|\d+(?:\.\d+)?")
_LIST_MARK = re.compile(r"^[ \t]*(?:[-*•·>#]+|\d+[.、)）]|[（(]\d+[)）])[ \t]*", re.M)
_SENT_END = "。！？!?\n；;"
_LEAD = ("根据", "依据", "按照", "按", "参见", "参考", "见", "据")  # 引出后面的内容
# 指回前面的内容：一段话之后单独一行「出处：【X 第3段】」说的是上面那段。真实评测里把它和「根据」混成一类，
# 这条引用挂到了下一段（另有出处的内容）上，判成引错段（agent-eval-20261112-193919，kb-hotel）
_BACK = ("来源", "出处", "引自", "摘自", "参考资料")


def _cn_int(s: str) -> int | None:
    s = s.strip()
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if len(s) == 1:
        return _CN.get(s)
    if s.startswith("十") and len(s) == 2:
        return 10 + _CN.get(s[1], 0)
    if len(s) in (2, 3) and s[1] == "十":
        return _CN[s[0]] * 10 + (_CN[s[2]] if len(s) == 3 else 0)
    return None


def _paras(s: str) -> list[int | None]:
    return [_cn_int(x) for x in re.split(r"[、,，和及]", s)]


def _key(s: str | None) -> str:
    return norm((s or "").strip("《》【】 ·・")).lower()


@dataclass
class Block:
    title: str
    section: str | None
    para: int  # 1 开始，和块头里的「第N段」一致
    text: str


@dataclass
class Citation:
    title: str
    section: str | None
    para: int | None
    start: int
    end: int
    status: str = ""
    claim: str = ""
    span: tuple[int, int] = (0, 0)
    detail: list[str] = field(default_factory=list)


def parse_blocks(results: list[str]) -> list[Block]:
    """kb_search 的返回文本 → 块。只认 render 产生的块头；not_found、错误文本里没有块头，自然为空。"""
    blocks = []
    for res in results:
        heads = list(_HEADER.finditer(res))
        for i, m in enumerate(heads):
            body_end = heads[i + 1].start() if i + 1 < len(heads) else len(res)
            blocks.append(Block(m.group(1), m.group(2) or None, int(m.group(3)), res[m.end():body_end].strip()))
    return blocks


def _split_head(head: str) -> tuple[str, str | None]:
    head = head.strip().strip("《》")
    for sep in ("·", "・"):
        if sep in head:
            t, _, sec = head.partition(sep)
            return t.strip().strip("《》 "), sec.strip() or None
    return head, None


def parse_citations(answer: str) -> list[Citation]:
    cites: list[Citation] = []
    for m in _BRACKET.finditer(answer):
        for part in re.split(r"[；;]", m.group(1)):  # 【A 第1段；B 第3段】
            pm = _IN_BRACKET.fullmatch(part.strip())
            if not pm:
                continue
            title, section = _split_head(pm.group(1))
            cites += [Citation(title, section, p, m.start(), m.end()) for p in _paras(pm.group(2))]
    taken = [(c.start, c.end) for c in cites]
    for m in _GUILLEMET.finditer(answer):
        if any(s <= m.start() < e for s, e in taken):  # 【《标题》 第1段】这种两种括号套着写的，只算一次
            continue
        cites += [Citation(m.group(1).strip(), (m.group(2) or "").strip() or None, p, m.start(), m.end())
                  for p in _paras(m.group(3))]
    return sorted(cites, key=lambda c: (c.start, c.para or 0))


def _sentence_start(answer: str, pos: int) -> int:
    return max((answer.rfind(ch, 0, pos) for ch in _SENT_END), default=-1) + 1


def _claim_span(answer: str, c: Citation, cites: list[Citation]) -> tuple[int, int]:
    marks = sorted({(o.start, o.end) for o in cites})
    sent = _sentence_start(answer, c.start)
    lo = max([sent] + [e for s, e in marks if e <= c.start])
    before = answer[lo:c.start]
    lead = before.strip("-*•·：:，, \t\n").strip()
    if lead in _BACK:  # 「出处：【X】」指回前一段
        para_start = answer.rfind("\n\n", 0, max(0, sent - 1))
        prev_end = len(answer[:sent].rstrip())
        if prev_end > 0:
            return (para_start + 2 if para_start >= 0 else 0), prev_end
    if lead and lead not in _LEAD and lead not in _BACK:
        return lo, c.start
    # 引用前面没有内容
    prev = answer[:c.start].rstrip(" \t")
    if not lead and prev and prev[-1] in "。！？!?；;":  # 「……600 元。【X 第3段】」：挂前一句
        end = len(prev) - 1
        return _sentence_start(answer, end), end
    # 句首引用（或「根据《X》第N段」）：挂后面
    after = c.end
    for s, e in marks:  # 连着写的几条引用（【A】【B】）一起跳过
        if s >= after and not answer[after:s].strip(" \t，,、"):
            after = max(after, e)
    if m := re.match(r"[ \t]*[：:]?[ \t]*\n", answer[after:]):  # 「依据《X》：\n- ……」挂整段列表，到空行或下一条引用
        start = after + m.end()
        para_end = answer.find("\n\n", start)
        nxt_cite = min([s for s, _ in marks if s >= start], default=len(answer))
        return start, min(para_end if para_end >= 0 else len(answer), nxt_cite)
    start = after + len(answer[after:]) - len(answer[after:].lstrip("：: \t"))
    stops = [i for ch in _SENT_END if (i := answer.find(ch, start)) >= 0] + [s for s, _ in marks if s >= start]
    end = min(stops, default=len(answer))
    if answer[start:end].strip():
        return start, end
    # 后面什么都没有（「来源：【A】【B】」在末尾）：挂前一段
    para_start = answer.rfind("\n\n", 0, max(0, sent - 1))
    return (para_start + 2 if para_start >= 0 else 0), max(0, sent - 1)


def _claim_text(answer: str, span: tuple[int, int], cites: list[Citation]) -> str:
    text = answer[span[0]:span[1]]
    for o in sorted({(o.start, o.end) for o in cites}, reverse=True):  # 段里别的引用标记去掉，段号不能当锚点
        s, e = max(o[0], span[0]) - span[0], min(o[1], span[1]) - span[0]
        if s < e:
            text = text[:s] + " " + text[e:]
    return _LIST_MARK.sub("", text)


def anchors(text: str) -> list[str]:
    """单个数字不算锚点：太容易在别的块里碰巧出现。真实评测（kb-sick）里模型自己举例算出的「剩下 3 天」，
    因为别的块里有个 3 被判成引错段。两位以上的数字（600、22、10）和代号（E1024）才有区分度。"""
    found = _ANCHOR.findall(_LIST_MARK.sub("", text))
    return list(dict.fromkeys(a for a in found if not (a.isdigit() and len(a) == 1)))


def _has(text: str, anchor: str) -> bool:
    """完整匹配：20 不算命中 200，0.5 不算命中 10.5；E1024 不算命中 E10245。"""
    return re.search(r"(?<![\d.A-Za-z])" + re.escape(norm(anchor)) + r"(?![\d.]\d|\d)", text) is not None


def _prefix_match(cited: str, block: str, cap: int) -> bool:
    """块头里的标题截到 cap 字：块头被截过时，按前缀比（模型可能抄截断的，也可能写全）；没截过就要求相等。"""
    if not cited:
        return False
    if len(block) >= cap:
        return cited.startswith(block) or block.startswith(cited)
    return cited == block


def _matches(c: Citation, b: Block) -> bool:
    if not _prefix_match(_key(c.title), _key(b.title), TITLE_CHARS):
        return False
    if c.section is None or b.section is None:
        return True
    return _prefix_match(_key(c.section), _key(b.section), SECTION_CHARS)


def check(answer: str, results: list[str]) -> dict:
    """返回 {citations: [...], n, ok, wrong, unchecked, correctness, coverage, anchor_not_in_evidence}。
    correctness = ok /（ok + wrong），分母是能判的引用数（不含 unchecked）；没有能判的引用时为 None。
    coverage = 带锚点、且锚点在证据里的句子中，被某条引用的「挂着的那段话」覆盖到的比例；没有这种句子时为 None。"""
    blocks = parse_blocks(results)
    cites = parse_citations(answer)
    evidence = [norm(b.text) for b in blocks]
    no_evidence = []
    for c in cites:
        c.span = _claim_span(answer, c, cites)
        c.claim = _claim_text(answer, c.span, cites)
        same_title = [b for b in blocks if _matches(c, b)]
        if not same_title:
            c.status = "unknown_source"
            continue
        cited = [b for b in same_title if b.para == c.para]
        if not cited:
            c.status = "wrong_para"
            continue
        found = [a for a in anchors(c.claim) if any(_has(t, a) for t in evidence)]
        no_evidence += [a for a in anchors(c.claim) if a not in found]
        if not found:
            c.status = "unchecked"
            continue
        missing = [a for a in found if not any(_has(norm(b.text), a) for b in cited)]
        c.status, c.detail = ("wrong_block", missing) if missing else ("ok", found)
    ok = sum(c.status == "ok" for c in cites)
    wrong = sum(c.status in ("wrong_block", "wrong_para", "unknown_source") for c in cites)
    sentences, covered = 0, 0
    for s, e, text in _sentences(answer):
        if any(_has(t, a) for a in anchors(_strip_cites(text)) for t in evidence):
            sentences += 1
            covered += any(cs < e and s < ce for cs, ce in (c.span for c in cites))
    return {"citations": [{"title": c.title, "section": c.section, "para": c.para, "status": c.status,
                           "claim": c.claim.strip()[:120], "detail": c.detail} for c in cites],
            "n": len(cites), "ok": ok, "wrong": wrong, "unchecked": sum(c.status == "unchecked" for c in cites),
            "correctness": round(ok / (ok + wrong), 3) if ok + wrong else None,
            "coverage": round(covered / sentences, 3) if sentences else None,
            "anchor_not_in_evidence": sorted(set(no_evidence))}


def _strip_cites(text: str) -> str:
    return _GUILLEMET.sub(" ", _BRACKET.sub(" ", text))


def _sentences(answer: str):
    start = 0
    for i, ch in enumerate(answer + "\n"):
        if ch in _SENT_END:
            if answer[start:i].strip():
                yield start, i, answer[start:i]
            start = i + 1
