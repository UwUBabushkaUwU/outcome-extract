"""Compact sheet rendering for row chunks, and citations built from cell references."""

from app import chunking, normalize as N
from app.chunking import Section

LINES = [
    "Row 7: [A7] Unit | [B7] Type | [H7] Charge | [I7] Amount",
    "Row 8: [A8] 100-101 | [B8] C5 | [H8] Rent | [I8] 2842",
    "Row 9: [H9] Storage Fee | [I9] 25",
    "Row 10: [G10] Charge Total: | [I10] 2867",
]
SHEET = Section("sheet", "Willow", LINES)
SRC = N.Source([SHEET])


def spec(type_="number", kind="extracted", **kw):
    return {**N.default_spec("k", {"description": ""}, True), "type": type_, "kind": kind, **kw}


def test_columns_span_leftmost_to_rightmost():
    assert chunking.columns_for(LINES) == list("ABCDEFGHI")
    assert chunking.columns_for(["...", "(headings above this part:)"]) == []


def test_render_compact_keeps_positions_and_drops_trailing_empties():
    cols = chunking.columns_for(LINES)
    out = chunking.render_compact(LINES + ["..."], cols)
    assert out[1] == "row8: 100-101 | C5||||||Rent | 2842"
    assert out[2] == "row9: |||||||Storage Fee | 25"
    assert out[-1] == "..."  # non-row lines pass through
    # Every value lands at its column's position: the n-th "|"-separated slot.
    for ln, orig in zip(out[:4], LINES):
        values = [v.strip() for v in ln.split(": ", 1)[1].split("|")]
        for col, text in chunking.parse_cells(orig).items():
            assert values[cols.index(col)] == text


def test_compact_is_shorter_than_tagged_cells():
    cols = chunking.columns_for(LINES)
    assert sum(map(len, chunking.render_compact(LINES, cols))) < sum(map(len, LINES))


def test_citation_built_from_references():
    assert N.citation_from_refs("I8", SRC, "Willow") == "[I8] 2842"
    assert N.citation_from_refs("Willow!I8", SRC, None) == "[Willow!I8] 2842"
    assert N.citation_from_refs("I8, I9", SRC, "Willow") == "[I8] 2842 ... [I9] 25"
    assert N.citation_from_refs(["I8", "I9"], SRC, "Willow") == "[I8] 2842 ... [I9] 25"
    assert N.citation_from_refs("I8:I9", SRC, "Willow") == "[I8:I9]"
    assert N.citation_from_refs("Z99", SRC, "Willow") == "[Z99]"  # no such cell: no text
    # An old-style quoted citation passes through for the normal check.
    assert N.citation_from_refs("[I8] 2842", SRC, "Willow") == "[I8] 2842"
    assert N.citation_from_refs("", SRC, "Willow") == ""


def test_value_must_agree_with_the_cell_it_cites():
    cite = lambda ref: N.citation_from_refs(ref, SRC, "Willow")
    ok = N.finalize_field([2842, cite("I8"), 0.9], spec(), SRC, "Willow")
    assert ok["value"] == 2842 and ok["citation"] == "[Willow!I8] 2842"
    assert N.finalize_field([9999, cite("I8"), 0.9], spec(), SRC, "Willow")["value"] is None
    assert N.finalize_field([2842, cite("Z99"), 0.9], spec(), SRC, "Willow")["value"] is None
    s = N.finalize_field(["C5", cite("B8"), 0.9], spec("string"), SRC, "Willow")
    assert s["value"] == "C5"


def test_a_value_found_twice_in_the_row_is_not_guessed():
    lines = ["Row 1: [A1] x | [B1] 0 | [C1] 0 | [D1] 5"]
    src = N.Source([Section("sheet", "S", lines)])
    f = N.finalize_field([0, N.citation_from_refs("D1", src, "S"), 0.9], spec(), src, "S")
    assert f["value"] is None  # D1 holds 5; two cells hold 0, so no correction


def test_calculated_and_inherited_by_reference():
    calc = spec(kind="calculated", calc="sum")
    f = N.finalize_field([0, N.citation_from_refs("I8:I9", SRC, "Willow"), 0.9], calc, SRC,
                         "Willow")
    assert f["value"] == 2867 and f["source_locations"] == ["Willow!I8", "Willow!I9"]
    inh = spec("string", kind="inherited")
    chunk = N.norm_text("\n".join(LINES))
    g = N.finalize_field(["Charge Total:", N.citation_from_refs("G10", SRC, "Willow"), 0.9],
                         inh, SRC, "Willow", chunk)
    assert g["value"] == "Charge Total:"
