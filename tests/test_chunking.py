import json
import re
from pathlib import Path

import pytest

from app import chunking, parsers
from app.chunking import Section, compact_line, split_rows
from app.parsers.base import Chunk, Document

REF = Path(__file__).resolve().parent.parent / "ref" / "sample"


def _rent_roll(units: int, charges: int = 9) -> Section:
    """Header, then per unit: a unit line (key in column A) and charge lines (A empty)."""
    lines = ["Row 1: [A1] Rent Roll", "Row 2: [A2] Unit | [B2] Type | [H2] Charge | [I2] Amount"]
    r = 3
    for u in range(units):
        lines.append(f"Row {r}: [A{r}] 100-{u:03d} | [B{r}] C5 | [H{r}] Rent | [I{r}] 2000")
        r += 1
        for c in range(charges):
            lines.append(f"Row {r}: [H{r}] Fee {c} | [I{r}] 25")
            r += 1
        lines.append(f"Row {r}: [G{r}] Charge Total: | [I{r}] 2225")
        r += 1
    return Section("sheet", "RR", lines)


def _first_key(chunk) -> str | None:
    cells = chunking.parse_cells(chunk.lines[0])
    return cells.get("A")


def test_rent_roll_chunks_never_cut_a_unit_block():
    sec = _rent_roll(units=40)  # 11 lines per unit, 442 lines
    chunks = split_rows(sec, ["unit_number", "Unit"], target=60)
    assert len(chunks) > 5
    for c in chunks[1:]:
        assert re.fullmatch(r"100-\d{3}", _first_key(c)), c.lines[0]
        assert c.lines[-1].endswith("2225")  # ends on the unit's Charge Total line
    # Every line appears exactly once across chunks.
    assert sum(len(c.lines) for c in chunks) == len(sec.lines)
    # Header context goes on every chunk but the first.
    assert chunks[0].header == []
    assert all("[A2] Unit" in c.header[-1] or any("[A2] Unit" in h for h in c.header)
               for c in chunks[1:])


def test_repeated_key_from_merged_cells_does_not_start_a_block():
    lines = ["Row 1: [A1] Code | [B1] Amount"]
    for r in range(2, 200):
        key = f"K{r // 10}"  # the same key on 10 consecutive lines, as merged cells produce
        lines.append(f"Row {r}: [A{r}] {key} | [B{r}] {r}")
    chunks = split_rows(Section("sheet", "S", lines), ["code", "Code"], target=25)
    for c in chunks[1:]:
        prev_chunk = chunks[c.index - 1]
        assert _first_key(c) != chunking.parse_cells(prev_chunk.lines[-1]).get("A")


def test_key_column_found_by_header_name_not_position():
    lines = ["Row 1: [A1] Seq | [B1] GL Code | [C1] Amount"]
    for r in range(2, 100):
        gl = f"{6000 + r // 3}-0000"  # three lines per GL code; column A changes every line
        lines.append(f"Row {r}: [A{r}] {r} | [B{r}] {gl} | [C{r}] {r * 10}")
    chunks = split_rows(Section("sheet", "S", lines), ["gl_code", "GL Code"], target=20)
    for c in chunks[1:]:
        assert chunking.parse_cells(c.lines[0])["B"].endswith("-0000")
        r = int(re.match(r"Row (\d+)", c.lines[0]).group(1))
        assert r % 3 == 0  # starts on the first line of a GL code's group


def test_pages_split_on_record_code_shape_and_keep_wrapped_lines():
    lines = ["[Page 1]", "Budget Analysis", "GL Line Item Budget"]
    for i in range(30):
        lines.append(f"61{i:02d}-0000 Item {i} 1,000 2,000 Explanation starts")
        lines += ["which wraps onto", "several lines"]
        if i == 15:
            lines.append("[Page 2]")
    sec = Section("pages", None, lines)
    chunks = split_rows(sec, ["gl_code", "GL Code"], target=20)
    for c in chunks:
        body = [ln for ln in c.lines if not chunking.PAGE_MARKER.match(ln)]
        assert re.match(r"\d{4}-0000 ", body[0]) or c.index == 0
        assert body[-1] in ("several lines",)
    later = [c for c in chunks if any("6116-0000" in ln for ln in c.lines)][0]
    assert later.lines[0] in ("[Page 1]", "[Page 2]")  # chunk knows its page


def test_empty_cells_are_dropped():
    assert compact_line("Row 9: [A9] | [B9]  | [H9] Rent | [I9] 25 | [J9] ") == \
        "Row 9: [H9] Rent | [I9] 25"
    assert compact_line("Row 15: [G15] Charge Total: | [I15] 2942") == \
        "Row 15: [G15] Charge Total: | [I15] 2942"
    doc = Document([Chunk("Row 1: [A1] x | [B1] \nRow 2: [A2] \n", "[S!A1:B2]")])
    assert chunking.build_sections(doc)[0].lines == ["Row 1: [A1] x"]


@pytest.mark.skipif(not (REF / "sample-data-documents" / "rentroll_001.xlsx").exists(),
                    reason="reference sample not present")
def test_sample_rent_roll_every_chunk_starts_on_a_unit():
    doc = parsers.parse(REF / "sample-data-documents" / "rentroll_001.xlsx")
    schema = json.loads((REF / "schemas" / "rent_roll_units.json").read_text())
    sec = chunking.build_sections(doc)[0]
    chunks = split_rows(sec, ["unit_number", schema["fields"]["unit_number"]["name"]])
    assert len(chunks) > 20
    units = [c for c in chunks[1:] if re.fullmatch(r"\d{3}-\d{3}", _first_key(c) or "")]
    # Every chunk starts on a unit line, except the trailing summary tables.
    assert len(units) >= len(chunks) - 4
    assert all(len(c.lines) <= chunking.MAX_BLOCK_LINES for c in chunks)
