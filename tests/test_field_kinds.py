"""Field kinds: extracted (strict citation), calculated (cites inputs, recomputed),
inherited (checked against the chunk's section context). No field names in the code, so
the tests use made-up ones."""

import re
from datetime import date

import pytest

from app import chunking, normalize as N, pipeline
from app.chunking import Section, split_rows
from app.pipeline import JobContext

# A T12-like sheet: section headings (no amounts) above GL lines with monthly cells.
SHEET = Section("sheet", "S", [
    "Row 6: [A6] GL | [B6] Name | [C6] Jan | [D6] Feb | [E6] Mar | [F6] Total",
    "Row 7: [A7] 500-000 | [B7] REVENUE",
    "Row 8: [A8] 510-500 | [B8] Rent | [C8] 100 | [D8] 0 | [E8] 50 | [F8] 150",
    "Row 9: [A9] 600-000 | [B9] EXPENSES",
    "Row 10: [A10] 610-100 | [B10] Repairs | [C10] 0 | [D10] 0 | [E10] 20 | [F10] 20",
])
PAGES = Section("pages", None, [
    "[Page 3]",
    "for the term of Ten (10) years Seven (7) months commencing on the 1st day of August,",
    "2010, and terminating on the 28th day of February, 2021",
    "[Page 18]",
    "| 05/01/97 - 04/30/98 | 26,088 | 2,174 | 12 | 26,088 |",
    "| 60 | $ 123,084 |",
])
SOURCE = N.Source([SHEET, PAGES])


def spec(type_="number", kind="extracted", calc="other", **kw):
    return {**N.default_spec("k", {"description": ""}, True), "type": type_, "kind": kind,
            "calc": calc, **kw}


def fin(value, citation, s, chunk_norm=None):
    return N.finalize_field([value, citation, 0.95], s, SOURCE, sheet="S", chunk_norm=chunk_norm)


# --------------------------------------------------------------------------- field prep


def test_clean_spec_reads_kind_and_calc():
    s = N.clean_spec("x", {"description": "d"}, {"kind": "calculated", "calc": "sum"}, True)
    assert (s["kind"], s["calc"], s["computed"]) == ("calculated", "sum", True)
    old = N.clean_spec("x", {"description": "d"}, {"computed": True}, True)
    assert old["kind"] == "calculated" and old["calc"] == "other"
    bad = N.clean_spec("x", {"description": "d"}, {"kind": "magic", "calc": "sqrt"}, True)
    assert (bad["kind"], bad["calc"]) == ("extracted", "other")
    pk = N.clean_spec("x", {"description": "d", "primary": True}, {"kind": "inherited"}, True)
    assert pk["kind"] == "extracted"


# --------------------------------------------------------------------------- extracted


def test_extracted_stays_strict():
    assert fin(150, "[F8] 150", spec())["value"] == 150
    assert fin(2, "Computed from [C8] to [E8]", spec())["value"] is None
    assert fin(150, "[F8] 999", spec())["value"] is None


def test_page_tag_before_a_table_cell_applies_to_the_text_after_it():
    f = fin(123084, "[Page 18] | 60 | $ 123,084", spec())
    assert f["value"] == 123084 and f["page_numbers"] == [18]


# --------------------------------------------------------------------------- calculated


@pytest.mark.parametrize("citation", [
    "Computed from [C8] to [E8]", "[C8:E8]", "[S!C8:E8]", "[C8]-[E8]",
    "[C8] 100 ... [D8] 0 ... [E8] 50",
])
def test_count_is_recomputed_from_the_cited_cells(citation):
    f = fin(3, citation, spec(kind="calculated", calc="count_nonzero"))
    assert f["value"] == 2  # the model said 3; Feb is 0
    assert f["source_locations"] == ["S!C8", "S!D8", "S!E8"]
    assert f["confidence"] <= 0.9


def test_count_the_model_got_right_keeps_its_confidence():
    f = fin(2, "[C8:E8]", spec(kind="calculated", calc="count_nonzero"))
    assert f["value"] == 2 and f["confidence"] == 0.95


def test_first_and_last_cell_joined_by_ellipsis_can_mean_the_range():
    s = spec(kind="calculated", calc="count_nonzero")
    # "[C8] 100 ... [E8] 50" may mean C8 and E8 (2) or C8 through E8 (2 non-zero of 3).
    assert fin(2, "[C8] 100 ... [E8] 50", s)["value"] == 2
    wide = Section("sheet", "W", ["Row 1: [A1] x | " + " | ".join(
        f"[{c}1] {v}" for c, v in zip("BCDEFG", [5, 0, 7, 8, 0, 9]))])
    src = N.Source([wide])
    run = lambda v: N.finalize_field([v, "[B1] 5 ... [G1] 9", 0.9], s, src, sheet="W")["value"]
    assert run(4) == 4  # the model meant B1..G1: matches the range reading, so it stands
    assert run(2) == 2  # the model meant the two cells: matches the list reading
    assert run(6) is None  # matches neither, and the readings disagree: ambiguous -> null


def test_count_over_only_the_nonzero_cells_still_counts_right():
    f = fin(1, "[E10] 20", spec(kind="calculated", calc="count_nonzero"))
    assert f["value"] == 1


def test_sum_and_difference_are_recomputed():
    assert fin(140, "[C8:E8]", spec(kind="calculated", calc="sum"))["value"] == 150
    assert fin(0, "[F8] 150 ... [F10] 20", spec(kind="calculated", calc="difference"))["value"] == 130


def test_calculated_needs_real_inputs():
    s = spec(kind="calculated", calc="count_nonzero")
    assert fin(12, "Computed from income line", s)["value"] is None
    assert fin(12, "[X500:Z500]", s)["value"] is None  # names no existing cell
    assert fin(12, "", s)["value"] is None


def test_classification_keeps_the_model_value_when_its_input_is_cited():
    s = spec("string", kind="calculated", calc="other")
    assert fin("Revenue", "[B7] REVENUE", s)["value"] == "Revenue"
    assert fin("Revenue", "the income line", s)["value"] is None


def test_term_in_months_is_recomputed_from_the_cited_dates():
    s = spec(kind="calculated", calc="months_between")
    f = fin(120, "[Page 3] commencing on the 1st day of August, 2010, and terminating on "
                 "the 28th day of February, 2021", s)
    assert f["value"] == 127 and f["page_numbers"] == [3]


def test_calculated_value_the_code_cannot_recompute_stands_if_inputs_verify():
    s = spec(kind="calculated", calc="months_between")
    f = fin(60, "[Page 18] | 60 | $ 123,084", s)  # no dates to recompute from
    assert f["value"] == 60


@pytest.mark.parametrize("start,end,months", [
    (date(1993, 6, 1), date(2003, 5, 31), 120),
    (date(2010, 8, 1), date(2021, 2, 28), 127),
    (date(2025, 10, 1), date(2027, 4, 30), 19),
    (date(2024, 11, 1), date(2025, 10, 31), 12),
])
def test_months_between(start, end, months):
    assert N.months_between(start, end) == months


def test_recompute_gives_up_on_inputs_it_cannot_use():
    assert N.recompute("sum", ["100", "Rent"]) is None
    assert N.recompute("months_between", ["no dates here"]) is None
    assert N.recompute("other", ["1", "2"]) is None


# --------------------------------------------------------------------------- inherited

CHUNK_WITH_HEADING = N.norm_text("Row 7: [A7] 500-000 | [B7] REVENUE\n"
                                 "Row 8: [A8] 510-500 | [B8] Rent | [C8] 100")
CHUNK_WITHOUT = N.norm_text("Row 10: [A10] 610-100 | [B10] Repairs | [C10] 0")


def test_inherited_value_checked_against_the_chunks_section_context():
    s = spec("string", kind="inherited")
    assert fin("REVENUE", "[B7] REVENUE", s, CHUNK_WITH_HEADING)["value"] == "REVENUE"
    # The heading exists in the document, but not above this chunk.
    assert fin("REVENUE", "[B7] REVENUE", s, CHUNK_WITHOUT)["value"] is None
    # The cited heading must be where the value came from.
    assert fin("EXPENSES", "[B7] REVENUE", s, CHUNK_WITH_HEADING)["value"] is None


def test_chunks_carry_the_headings_above_them():
    lines = ["Row 1: [A1] GL | [B1] Name | [C1] Amount"]
    r = 2
    for sec_no in range(3):
        lines.append(f"Row {r}: [A{r}] {sec_no}00-000 | [B{r}] SECTION {sec_no}")
        r += 1
        for _ in range(20):
            lines.append(f"Row {r}: [A{r}] {sec_no}{r:02d}-1 | [B{r}] Line | [C{r}] {r}")
            r += 1
    chunks = split_rows(Section("sheet", "S", lines), ["gl", "GL"], target=12)
    for c in chunks[1:]:
        first_row = int(re.match(r"Row (\d+)", c.lines[0]).group(1))
        expected = (first_row - 2) // 21  # which section the chunk starts in
        assert any(f"SECTION {expected}" in h for h in c.header), (first_row, c.header)
    a, b = chunks[3].halves()
    assert any("SECTION" in h for h in b.header)


# --------------------------------------------------------------------------- end to end

SCHEMA = {
    "name": "t12", "description": "One row per GL line with an amount.",
    "fields": {
        "gl": {"name": "GL", "description": "GL code.", "primary": True},
        "section": {"name": "Section", "description": "Heading above the line."},
        "active": {"name": "Active months", "description": "Months with a non-zero value."},
    },
}


def test_pipeline_uses_kinds_end_to_end(tmp_path, fake_llm):
    rows = ["GL,Name,Jan,Feb,Mar"]
    for s in range(2):
        rows.append(f"{s}00-000,HEAD {s},,,")
        rows += [f"{s}{i:02d}-1,Line,{i},0,{i % 2}" for i in range(1, 40)]
    path = tmp_path / "t12.csv"
    path.write_text("\n".join(rows) + "\n")
    fake_llm.prep = lambda m: {"fields": {
        "gl": {"type": "string", "scope": "row", "kind": "extracted"},
        "section": {"type": "string", "scope": "row", "kind": "inherited"},
        "active": {"type": "number", "scope": "row", "kind": "calculated",
                   "calc": "count_nonzero"},
    }}
    prompts = []

    def extract(messages):
        user = messages[1]["content"]
        prompts.append(user)
        context, body = user.split("ROWS (")
        heads = re.findall(r"Row (\d+): \[A\d+\] \d00-000 \| \[B\d+\] (HEAD \d)",
                           context + body)
        out = []
        for r, gl in re.findall(r"Row (\d+): \[A\d+\] (\d\d\d-1) \|", body):
            head_row, head = [h for h in heads if int(h[0]) < int(r)][-1]
            out.append({"gl": [gl, f"[A{r}] {gl}", 1],
                        "section": [head, f"[B{head_row}] {head}", 0.9],
                        "active": [3, f"Computed from [C{r}] to [E{r}]", 0.9]})
        return {"rows": out}

    fake_llm.rows = extract
    out = pipeline.run(JobContext("j", path, "csv", "t12.csv", SCHEMA, lambda s: None))

    assert out.status == "SUCCESS", out.error
    assert len(out.entries) == 78 and len(prompts) > 1
    by_gl = {e["gl"]["value"]: e for e in out.entries}
    assert by_gl["001-1"]["active"]["value"] == 2  # 1, 0, 1 -> recomputed from 3
    assert by_gl["002-1"]["active"]["value"] == 1  # 2, 0, 0
    # Lines deep in a section, in a later chunk, still get their heading.
    assert by_gl["139-1"]["section"]["value"] == "HEAD 1"
    assert by_gl["035-1"]["section"]["value"] == "HEAD 0"
    assert '"kind": "calculated"' in prompts[0] and '"calc": "count_nonzero"' in prompts[0]
