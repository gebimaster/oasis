"""O8 第一步：引用校验（纯函数，不调模型）。"""

from app import citations
from app.kb import Hit
from app.kb_search import Fused, SearchResult
from app.kb_tool import render

EXPENSE = [
    ("差旅报销制度", None, 0, "市内交通：打车单次上限 200 元；夜间 22 点后加班打车可全额报销。"),
    ("差旅报销制度", None, 2, "住宿标准：一线城市每晚不超过 600 元，其他城市每晚不超过 400 元。"),
    ("错误码手册", "设备类", 4, "E1024 设备离线：检查网线和电源后重启网关。"),
]


def result(rows=EXPENSE) -> str:
    """用真的 kb_tool.render 产生返回文本：块头格式和生产一致，改了 render 这里会跟着变。"""
    hits = [Fused(Hit(id=f"c{i}", text=x, doc_id="d", title=t, chunk_index=i, score=0.0, section=s or "", para_index=p), rerank=0.9)
            for i, (t, s, p, x) in enumerate(rows)]
    return render("q", SearchResult("ok", hits))


def statuses(answer, results=None):
    out = citations.check(answer, [result()] if results is None else results)
    return [c["status"] for c in out["citations"]], out


def test_parse_blocks_from_real_render():
    blocks = citations.parse_blocks([result()])
    assert [(b.title, b.section, b.para) for b in blocks] == [
        ("差旅报销制度", None, 1), ("差旅报销制度", None, 3), ("错误码手册", "设备类", 5)]
    assert blocks[2].text.startswith("E1024")


def test_correct_citation_is_ok():
    st, out = statuses("夜间 22 点后加班打车可以全额报销【差旅报销制度 第1段】。")
    assert st == ["ok"] and out["correctness"] == 1.0 and out["coverage"] == 1.0


def test_citing_the_wrong_paragraph_is_caught():
    """600 在第 3 段，却标了第 1 段：两段都是本次返回的，评委只看「原文里有没有」会放过。"""
    st, out = statuses("上海住宿每晚不超过 600 元【差旅报销制度 第1段】。")
    assert st == ["wrong_block"] and out["citations"][0]["detail"] == ["600"]
    assert out["correctness"] == 0.0


def test_citing_a_paragraph_that_was_not_returned():
    st, _ = statuses("加班要提前申请【差旅报销制度 第7段】。")
    assert st == ["wrong_para"]


def test_citing_an_unknown_document():
    st, _ = statuses("出差有 100 元饭补【员工手册 第1段】。")
    assert st == ["unknown_source"]


def test_sentence_without_anchor_is_unchecked_not_counted():
    st, out = statuses("打车需要在备注里写明加班事由【差旅报销制度 第1段】。")
    assert st == ["unchecked"] and out["correctness"] is None


def test_guillemet_style_and_leading_citation():
    """「根据《X》第N段：……」引用在句首，挂的是后面那句。"""
    st, out = statuses("根据《差旅报销制度》第3段：一线城市每晚不超过 600 元。")
    assert st == ["ok"] and "600" in out["citations"][0]["claim"]


def test_section_citation_and_codes_as_anchors():
    st, _ = statuses("E1024 要先检查网线和电源【错误码手册 · 设备类 第5段】。")
    assert st == ["ok"]
    st, _ = statuses("E1024 要先检查网线和电源【错误码手册 · 网络类 第5段】。")  # 小节写错：对不上
    assert st == ["unknown_source"]


def test_paragraph_number_in_citation_is_not_an_anchor():
    """「第3段」的 3 不能当锚点：否则任何引用都会因为段号在别的块里出现而判错。"""
    st, _ = statuses("一线城市住宿有上限【差旅报销制度 第3段】，22 点后打车全额【差旅报销制度 第1段】。")
    assert st == ["unchecked", "ok"]


def test_anchor_missing_from_all_evidence_is_faithfulness_not_citation():
    st, out = statuses("打车单次上限 300 元【差旅报销制度 第1段】。")
    assert st == ["unchecked"] and out["anchor_not_in_evidence"] == ["300"]


def test_coverage_counts_anchored_sentences_without_citation():
    _, out = statuses("22 点后打车全额报销【差旅报销制度 第1段】。一线城市每晚不超过 600 元。")
    assert out["coverage"] == 0.5


def test_citation_right_after_period_covers_previous_sentence():
    _, out = statuses("一线城市每晚不超过 600 元。【差旅报销制度 第3段】")
    assert out["coverage"] == 1.0


def test_no_kb_results_means_every_citation_is_unknown():
    st, _ = statuses("每晚不超过 600 元【差旅报销制度 第3段】。", results=["[KB_RESULT] 知识库里没有与「x」相关的内容。"])
    assert st == ["unknown_source"]


def test_chinese_numerals_and_malformed():
    st, _ = statuses("每晚不超过 600 元【差旅报销制度 第三段】。")
    assert st == ["ok"]


# ---------------- O8 复核补的用例（真实回答的写法） ----------------

def test_list_markers_are_not_anchors():
    """复核①：「1.」的 1 不能当锚点（真实回答 174 条里 72 条用了编号列表）。"""
    st, _ = statuses("住宿标准：\n1. 一线城市每晚不超过 600 元【差旅报销制度 第3段】")
    assert st == ["ok"]


def test_anchors_match_whole_numbers_only():
    """复核①：20 不能算命中 200，60 不能算命中 600。"""
    st, out = statuses("打车单次上限 20 元【差旅报销制度 第1段】。")
    assert st == ["unchecked"] and out["anchor_not_in_evidence"] == ["20"]


def test_realistic_answer_with_lead_in_list_and_trailing_sources():
    """复核②：模型真实的几种写法：「依据《X》第N段：」后跟列表、句末标点后再标出处、末尾「来源：【A】【B】」。"""
    answer = ("依据《差旅报销制度》第1段：\n- 打车单次上限 200 元\n- 夜间 22 点后加班打车可全额报销\n\n"
              "一线城市每晚不超过 600 元。【差旅报销制度 第3段】\n\n"
              "E1024 要先检查网线和电源后重启网关。\n\n来源：【错误码手册 · 设备类 第5段】")
    st, out = statuses(answer)
    assert st == ["ok", "ok", "ok"], out["citations"]
    assert out["correctness"] == 1.0 and out["coverage"] == 1.0


def test_citation_after_period_status_is_checked_too():
    """复核②：原来只断言了覆盖率；判定挂到了后一句。"""
    st, out = statuses("一线城市每晚不超过 600 元。【差旅报销制度 第3段】打车单次上限 200 元【差旅报销制度 第1段】。")
    assert st == ["ok", "ok"] and "600" in out["citations"][0]["claim"]


def test_multiple_paragraphs_in_one_citation():
    """复核⑩：【X 第1、3段】、【A 第1段；B 第5段】。"""
    st, _ = statuses("打车 200 元，住宿 600 元【差旅报销制度 第1、3段】。")
    assert st == ["wrong_block", "wrong_block"]  # 两个锚点分在两段：每一段单看都缺一个
    st, _ = statuses("E1024 断网先查网线，打车 200 元【错误码手册 · 设备类 第5段；差旅报销制度 第1段】。")
    assert len(st) == 2


def test_long_titles_truncated_in_header_still_match():
    """复核④：块头标题截到 60 字，模型写全名也要对得上；没截过的短标题要求相等（员工手册 ≠ 员工手册附录）。"""
    long = "某某科技有限公司员工差旅费用报销管理制度及实施细则（二〇二六年修订版，适用于全体正式员工与实习生以及外包驻场人员和临时借调人员）"
    assert len(long) > 60
    r = [result([(long, None, 0, "打车单次上限 200 元。")])]
    assert statuses(f"打车上限 200 元【{long} 第1段】。", r)[0] == ["ok"]
    assert statuses(f"打车上限 200 元【{long[:60]} 第1段】。", r)[0] == ["ok"]
    r2 = [result([("员工手册附录", None, 0, "病假 10 天以内全额发放。")])]
    assert statuses("病假 10 天以内全额发放【员工手册 第1段】。", r2)[0] == ["unknown_source"]


def test_list_marker_digit_does_not_make_a_sentence_countable_for_coverage():
    """覆盖率的分母：「1. 需要部门负责人审批」的 1 是序号，不是锚点，不能因为证据里有「1 天」就算进分母。"""
    r = [result([("请假制度", None, 0, "病假需提前 1 天在系统里申请，部门负责人审批。")])]
    out = citations.check("1. 需要部门负责人审批。", r)
    assert out["coverage"] is None



def test_source_line_refers_back_to_the_paragraph_above():
    """真实评测（kb-hotel）里的写法：一段话之后单独一行「出处：【X 第3段】」，后面再跟别的带出处的列表。
    「出处 / 来源」指回上一段，「根据 / 依据」才引出后面。"""
    answer = ("上海属于一线城市，住宿费每晚最多报销 600 元。\n\n出处：【差旅报销制度 第3段】\n\n补充一点：\n"
              "- 打车单次上限 200 元【差旅报销制度 第1段】")
    st, out = statuses(answer)
    assert st == ["ok", "ok"] and "600" in out["citations"][0]["claim"]



def test_single_digits_are_not_anchors():
    """真实评测（kb-sick）：模型举例算出的「剩下 3 天」，别的块里碰巧有个 3，不能因此判引错段。"""
    st, _ = statuses("一线城市住宿每晚不超过 600 元，住 3 晚就是 1800 元。【差旅报销制度 第3段】")
    assert st == ["ok"]
