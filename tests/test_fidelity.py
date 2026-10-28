"""tests/test_fidelity.py

Standalone text-fidelity check (extraction/fidelity.py).
"""

from csr_lab.extraction.fidelity import check_fragments, format_report
from csr_lab.schemas import PageSpan

SOURCE = """\
1 Scope
1.1 This document applies to suppliers.

2 Requirements
2.1 The supplier shall package all parts in accordance with Specification CVM-PKG-210.
2.2 The supplier shall keep records
for ten years after the end of production.
2.3 Labels shall conform to CVM-LBL-220.
"""


def test_match_kinds_and_differences():
    report = check_fragments(
        SOURCE,
        [
            "2.1 The supplier shall package all parts in accordance with Specification CVM-PKG-210.",
            "2.2 The supplier shall keep records for ten years after the end of production.",  # line break removed
            "2.1 The supplier must package all parts per Specification CVM-PKG-210.",  # reworded
            "Completely unrelated text about something else entirely here.",
        ],
    )
    kinds = [f.match for f in report.fragments]
    assert kinds == ["exact", "whitespace_normalized", "fuzzy", "not_found"]
    s = report.summary
    assert (s.exact, s.whitespace_normalized, s.fuzzy, s.not_found) == (1, 1, 1, 1)
    assert s.verbatim_rate == 0.5
    reworded = report.fragments[2]
    assert 0.6 <= reworded.fragment_ratio < 1.0
    changes = {(d.source_text, d.fragment_text) for d in reworded.differences}
    assert ("shall", "must") in changes and ("in accordance with", "per") in changes
    # The reworded fragment is located on the right source span.
    assert SOURCE[reworded.start_offset : reworded.end_offset].startswith("2.1 The supplier shall")


def test_gaps_flag_uncovered_requirement_text():
    report = check_fragments(SOURCE, ["2.1 The supplier shall package all parts in accordance with Specification CVM-PKG-210."])
    assert report.summary.source_coverage < 0.5
    assert any(g.looks_like_requirement and "2.3 Labels shall" in g.preview for g in report.gaps)
    assert "REQUIREMENT?" in format_report(report)


def test_reported_pages_are_compared_with_located_pages():
    half = SOURCE.index("2 Requirements")
    pages = [PageSpan(page_number=1, start_offset=0, end_offset=half - 1), PageSpan(page_number=2, start_offset=half, end_offset=len(SOURCE))]
    report = check_fragments(
        SOURCE,
        [{"id": "a", "text": "2.3 Labels shall conform to CVM-LBL-220.", "source_pages": [2]}, {"id": "b", "text": "1.1 This document applies to suppliers.", "source_pages": [2]}],
        pages=pages,
    )
    a, b = report.fragments
    assert (a.page_start, a.pages_agree) == (2, True)
    assert (b.page_start, b.pages_agree) == (1, False)
    assert report.summary.pages_checked == 2 and report.summary.pages_agree == 1


def test_dropped_words_lower_span_ratio():
    report = check_fragments(SOURCE, ["2.2 The supplier shall keep records after the end of production."])
    frag = report.fragments[0]
    assert frag.match == "fuzzy" and frag.fragment_ratio == 1.0 and frag.span_ratio < 1.0
    assert any(d.op == "delete" and "ten years" in d.source_text for d in frag.differences)
