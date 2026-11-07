import pytest

from app import normalize as N
from app.chunking import Section


@pytest.mark.parametrize("raw,expected", [
    ("$4,250.00", 4250), ("(1,234.50)", -1234.5), ("-322,938", -322938), ("12.5%", 12.5),
    ("339,050 RSF", 339050), (7, 7), (2.0, 2), ("N/A", None), ("-", None), ("", None),
    ("1 to 2", None), (True, None), ("−46.78%", -46.78),
])
def test_parse_number(raw, expected):
    assert N.parse_number(raw) == expected


@pytest.mark.parametrize("raw,fmt,expected", [
    ("2022-12-03", "YYYY-MM-DD", "2022-12-03"),
    ("02/09/2025", "YYYY-MM-DD", "2025-02-09"),
    ("2/9/25", "YYYY-MM-DD", "2025-02-09"),
    ("June 1, 1993", "YYYY-MM-DD", "1993-06-01"),
    ("the 1st day of June, 1993", "YYYY-MM-DD", "1993-06-01"),
    ("1-Jun-93", "YYYY-MM-DD", "1993-06-01"),
    ("Feb 2023", "YYYY-MM", "2023-02"),
    ("Feb 2023", "YYYY-MM-DD", None),  # no day given: don't invent one
    ("2023-02-31", "YYYY-MM-DD", None),
    ("soon", "YYYY-MM-DD", None),
])
def test_parse_date(raw, fmt, expected):
    assert N.parse_date(raw, fmt) == expected


def _spec(type_="string", **kw):
    return {**N.default_spec("k", {"description": ""}, False), "type": type_, **kw}


def test_list_coercion_keeps_alignment():
    spec = _spec("list", item_type="number")
    assert N.coerce(["$1,000", "n/a", 3], spec) == [1000, None, 3]
    assert N.coerce("5", spec) == [5]


SHEET = Section("sheet", "Units", [
    "Row 7: [A7] Unit | [F7] Market Rent",
    "Row 8: [A8] 100-101 | [E8] Rainey, Brenda | [F8] 3197",
    "Row 10: [H10] Rent | [I10] 2842",
])
PAGES = Section("pages", None, ["[Page 1]", "LEASE", "[Page 3]",
                                "Base Rent: $4,250.00 per month, payable in advance."])
SOURCE = N.Source([SHEET, PAGES])


def test_cell_citation_is_verified_and_sheet_qualified():
    f = N.finalize_field([3197, "[F8] 3197", 0.9], _spec("number"), SOURCE, sheet="Units")
    assert f == {"value": 3197, "citation": "[Units!F8] 3197", "confidence": 0.9,
                 "source_locations": ["Units!F8"]}


def test_page_citation_gets_page_number():
    f = N.finalize_field({"value": "$4,250.00", "citation": "[Page 3] Base Rent: $4,250.00 per month",
                          "confidence": 0.95}, _spec("number"), SOURCE)
    assert f["value"] == 4250 and f["page_numbers"] == [3]


def test_hallucinated_citation_nulls_the_value():
    f = N.finalize_field({"value": 5000, "citation": "[Page 3] Base Rent: $5,000.00",
                          "confidence": 0.99}, _spec("number"), SOURCE)
    assert f["value"] is None and f["citation"] == ""


def test_value_without_citation_is_nulled_unless_it_is_the_absent_default():
    zero_if_none = _spec("number", default_if_absent=0)
    assert N.finalize_field([0, "", 0.9], zero_if_none, SOURCE)["value"] == 0
    assert N.finalize_field([25, "", 0.9], zero_if_none, SOURCE)["value"] is None
    empty_list = _spec("list", item_type="string", default_if_absent=[])
    assert N.finalize_field([[], "", 0.9], empty_list, SOURCE)["value"] == []


def test_computed_value_citing_several_snippets():
    f = N.finalize_field([6039, "[F8] 3197 ... [I10] 2842", 0.8], _spec("number"), SOURCE,
                         sheet="Units")
    assert f["value"] == 6039
    assert f["source_locations"] == ["Units!F8", "Units!I10"]
    bad = N.finalize_field([6039, "[F8] 3197 ... [I10] 9999", 0.8], _spec("number"), SOURCE,
                           sheet="Units")
    assert bad["value"] is None


def test_wrong_cell_but_real_text_keeps_value_and_drops_location():
    f = N.finalize_field(["Rainey, Brenda", "[A8] Rainey, Brenda", 0.9], _spec(), SOURCE,
                         sheet="Units")
    assert f["value"] == "Rainey, Brenda" and "source_locations" not in f


def test_assemble_copies_document_fields_and_fills_every_key():
    schema = {"fields": {"prop": {}, "unit": {"primary": True}, "rent": {}, "extra": {}}}
    doc = {"prop": {"value": "Willow", "citation": "[Units!A1] Willow", "confidence": 1.0}}
    rows = [{"unit": {"value": "1", "citation": "[Units!A8] 1", "confidence": 1.0},
             "rent": {"value": None, "citation": "junk", "confidence": 0.2}},
            {"unit": {"value": "2", "citation": "[Units!A9] 2", "confidence": 1.0}}]
    entries = N.assemble(schema, doc, rows)
    assert len(entries) == 2
    for e in entries:
        assert set(e) == {"prop", "unit", "rent", "extra"}
        assert e["prop"]["value"] == "Willow"
    assert entries[0]["rent"] == {"value": None, "citation": "", "confidence": 0.2}
    assert entries[1]["extra"]["value"] is None


def test_dedupe_keeps_first_occurrence():
    rows = [{"u": {"value": "100-312", "x": 1}}, {"u": {"value": None}},
            {"u": {"value": " 100-312 ", "x": 2}}, {"u": {"value": "100-313"}}]
    out = N.dedupe_rows(rows, "u")
    assert [r["u"]["value"] for r in out] == ["100-312", "100-313"]
    assert out[0]["u"]["x"] == 1


def test_clean_spec_falls_back_and_forces_primary_to_row():
    spec = N.clean_spec("code", {"description": "d", "primary": True},
                        {"type": "bogus", "scope": "document", "default_if_absent": "x"}, True)
    assert spec["type"] == "string" and spec["scope"] == "row"
    num = N.clean_spec("n", {"description": "d"}, {"type": "number", "default_if_absent": "zero"},
                       True)
    assert num["default_if_absent"] is None  # not a valid number, so ignored
