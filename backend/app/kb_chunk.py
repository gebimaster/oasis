"""知识库切块（O5 第 2 步）：按小节、按条目切，边界只由内容决定；长条目再按句子拼，相邻块重叠。

M6 的旧切法把短段落按累计长度合并到 200 字：块边界由「前面攒了多少字」决定，E1024、E1025、E2001
被合进一块，标题行长度一变边界也跟着变。现在的规则（用户 O5 选的）：
- `## 标题` 行开一个新小节，块不跨小节；小节标题记进 Chunk.section（Q7 选 B，也拼进 embedding 文本）。
- 段落（空行分隔）是一个条目；超过一半行是列表项的段落，每个列表项是一个条目（Q5 选 C）。
- 少于 MIN_ENTRY 字的条目并进下一条（Q6 选 C）：只看这一条自己的长度，不看累计长度。
- 条目 ≤ SIZE 字自成一块；更长的（含短条目并出来的）按句子拼块，块之间重叠 OVERLAP 字，重叠只在条目内部。
增量保证（复核后用户选 A：规则不动，把保证说准；第二、三轮复核再收窄）：插入或修改一个条目，改动只限于
它所在的「合并组」：它前面连着的短条目、它自己、它（若也短）并进的那一条，再加上那串短条目原来并进的那一条
（新段落把短条目抢走后，那一条的块从「短条目\\n原条目」变成「原条目」）。组外的块 hash 不变。
组内要重算几块没有上限：
- 常见情况只变 1 块；插入点正前方是短条目（引导句、小节末尾的短残片）时，新段落会把它抢走，前面那块也要重算。
- 组里有超过 SIZE 的长条目时，短条目并进或离开它，整条长条目要重新走 _split_long。第一块还装得下短条目时，
  只有第一块变；装不下就把一句挤到第二块，后面贪心断句的边界可能一路挪位，最多这条长条目的块全部重算
  （5 字引导句插到 1000 字、每句 20 字的条目前面，重算 6 块；换成 950 字、每句 19 字，只重算第一块）。
  新段落本身很长也一样。
- 前提是不改变「这一段算不算列表」：_entries 按「超过一半的行是列表项」判定，给列表项加续行可能让比例
  跌到一半以下，整段变成一条，同段其他列表项的块也要重算。
"""

import hashlib
import re
from dataclasses import dataclass

SIZE, OVERLAP = 200, 30  # M6 用户选的 (a)：块约 200 字，长条目内部相邻块重叠 30 字；块最长 230 字（kb_tool 靠它守 4000 字上限）
MIN_ENTRY = 20  # O5 用户选：少于 20 字的条目（引导句、残片）并进下一条
_SENT_END = re.compile(r"(?<=[。！？；!?;])")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
_LIST_ITEM = re.compile(r"^(?:[-*+•]\s+|\d+[.)]\s+|\d+[、）]\s*)")  # 「1、」「1）」中文习惯后面不空格


@dataclass(frozen=True)
class Chunk:
    text: str
    section: str = ""    # 所在小节的标题，没有小节时为空
    para_index: int = 0  # 这块的第一条内容来自正文第几段（从 0 起，标题行不算段）；引用「第N段」用它（Q8 选 C）


def _sentences(paragraph: str, size: int) -> list[str]:
    """按句末标点断句；单句仍超过 size 时按字数硬切。"""
    out = []
    for s in _SENT_END.split(paragraph):
        s = s.strip()
        out.extend(s[i:i + size] for i in range(0, len(s), size) if s)
    return out


def _split_long(entry: str, size: int, overlap: int) -> list[str]:
    """长条目拆句后拼块，后一块带上前一块末尾 overlap 字。carry 只在这一条内部传。"""
    out: list[str] = []
    carry, piece = "", ""  # carry：上一块末尾的重叠部分；piece：本块的新内容（≤ size）
    for s in _sentences(entry, size):
        if piece and len(piece) + len(s) > size:
            out.append(carry + piece)
            carry = (carry + piece)[-overlap:]  # 重叠：防止关键句正好跨在边界上
            piece = ""
        piece += s
    if piece:
        out.append(carry + piece)
    return out


def _entries(paragraph: str) -> list[str]:
    """一个段落里的条目：列表段落按列表项切（不是列表项的行算上一项的续行），其余整段是一条。"""
    lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
    if len(lines) >= 2 and sum(1 for ln in lines if _LIST_ITEM.match(ln)) * 2 > len(lines):
        items: list[str] = []
        for ln in lines:
            if _LIST_ITEM.match(ln) or not items:
                items.append(ln)
            else:
                items[-1] += "\n" + ln
        return items
    return ["\n".join(lines)]


def _sections(text: str) -> list[tuple[str, list[tuple[int, str]]]]:
    """正文 → [(小节标题, [(段落序号, 条目)])]。标题行可以和正文挤在同一段里，逐行认。"""
    sections: list[tuple[str, list[tuple[int, str]]]] = [("", [])]
    para = 0
    for block in re.split(r"\n\s*\n", text):
        lines: list[str] = []

        def flush() -> None:
            nonlocal para
            if lines:
                sections[-1][1].extend((para, e) for e in _entries("\n".join(lines)))
                para += 1
                lines.clear()

        for ln in block.strip().splitlines():
            m = _HEADING.match(ln.strip())
            if m:
                flush()
                sections.append((m.group(1).strip(), []))
            else:
                lines.append(ln)
        flush()
    return [s for s in sections if s[1]]


def _group_short(entries: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """少于 MIN_ENTRY 字的条目并进下一条（引导句跟着它引出的内容走）。判断只看这一条自己的长度，
    不看攒了多少字，所以边界仍由内容决定。小节末尾的短条目没有下一条可并，自成一块。
    代价（复核发现）：只要跨段落合并，就有一块同时装着两个条目，往它们中间插段落，这一块必然要变；
    所以「插入点之前的块全不变」只对插入点前面不是短条目的情况成立；并进的是长条目时整条重新断句，边界可能一路挪位。见模块 docstring。"""
    out: list[tuple[int, str]] = []
    pending: list[tuple[int, str]] = []
    for idx, e in entries:
        pending.append((idx, e))
        if len(e) >= MIN_ENTRY:
            out.append((pending[0][0], "\n".join(t for _, t in pending)))
            pending = []
    if pending:
        out.append((pending[0][0], "\n".join(t for _, t in pending)))
    return out


def split_chunks(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, entries in _sections(text):
        for idx, e in _group_short(entries):
            pieces = [e] if len(e) <= size else _split_long(e, size, overlap)
            chunks.extend(Chunk(p, section, idx) for p in pieces)
    return chunks


def split_text(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[str]:
    """只要块文本时用（测试、评测）。"""
    return [c.text for c in split_chunks(text, size, overlap)]


def embed_text(title: str, section: str, text: str) -> str:
    """真正拿去算 embedding、分词、rerank 的文本（contextual embedding）：文档标题 + 小节标题 + 块。
    没有小节时和 M6 一样是「标题\\n块」，旧块的 id 不变。块 id 也按它算：改小节标题，这一节的块全部重算。"""
    return f"{title} > {section}\n{text}" if section else f"{title}\n{text}"


# 停用词：几乎每块都有、不区分内容的虚词和疑问词。只放很少几个，「不」「没有」这类会改变语义的词不放
STOPWORDS = frozenset("的 了 吗 呢 吧 啊 是 在 和 与 及 或 也 就 都 而 把 被 我 你 他 她 它 我们 怎么 怎样 如何 什么 哪些 多少 请问".split())
_WORD = re.compile(r"\w")  # 含字母、数字或汉字才算词，标点和空白丢掉


def tokenize(text: str) -> list[str]:
    """BM25 用的分词：jieba 搜索引擎模式（长词再切出短词，「交通费用」也能被「费用」命中），英文转小写。
    入库和查询必须用同一个函数，否则同一个词会被切成两样、互相匹配不上。"""
    import jieba  # 延迟导入：jieba 首次加载词典约 1 秒
    return [w for w in (t.strip().lower() for t in jieba.lcut_for_search(text))
            if w and _WORD.search(w) and w not in STOPWORDS]


def chunk_id(user_id: str, doc_id: str, content: str) -> str:
    """隔离维度写进 id：同一段内容在不同用户 / 不同文档下是不同的块，互不跳过、互不覆盖。"""
    return hashlib.sha256(f"{user_id}|{doc_id}|{content}".encode()).hexdigest()
