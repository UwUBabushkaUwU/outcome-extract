"""pipeline.run end to end against the fake LLM: no network, no spend."""

import json
import re

import pymupdf

from app import llm, pipeline
from app.pipeline import JobContext

SCHEMA = {
    "name": "units", "description": "One row per unit.",
    "fields": {
        "property": {"name": "Property", "description": "Header name. Same for every row."},
        "unit": {"name": "Unit", "description": "Unit id.", "primary": True},
        "rent": {"name": "Rent", "description": "Rent amount."},
    },
}
PREP = {"fields": {
    "property": {"type": "string", "scope": "document"},
    "unit": {"type": "string", "scope": "row"},
    "rent": {"type": "number", "scope": "row"},
}}


def _csv(tmp_path, units=150):
    lines = ["Willow Creek", "Unit,Charge,Amount"]
    for u in range(units):
        lines += [f"U{u:03d},Rent,{1000 + u}", ",Fee,25", ",Fee,10"]
    p = tmp_path / "rr.csv"
    p.write_text("\n".join(lines) + "\n")
    return p


def _ctx(path, kind="csv", schema=SCHEMA):
    return JobContext("job1", path, kind, path.name, schema, set_stage=lambda s: None)


def _rows_from_prompt(messages):
    """Fake row extractor: return every unit whose line is in ROWS, citing real cells."""
    rows_text = messages[1]["content"].split("ROWS (")[1]
    out = []
    for m in re.finditer(r"Row (\d+): \[A\d+\] (U\d{3}) \| \[B\d+\] Rent \| \[C\d+\] (\d+)",
                         rows_text):
        r, unit, rent = m.groups()
        out.append({"unit": [unit, f"[A{r}] {unit}", 0.99],
                    "rent": [int(rent), f"[C{r}] {rent}", 0.95]})
    return {"rows": out}


def test_rows_are_extracted_per_chunk_merged_and_deduped(tmp_path, fake_llm):
    fake_llm.prep = lambda m: PREP
    fake_llm.doc = lambda m: {"fields": {"property": {
        "value": "Willow Creek", "citation": "[Sheet1!A1] Willow Creek", "confidence": 1}}}
    fake_llm.rows = _rows_from_prompt
    out = pipeline.run(_ctx(_csv(tmp_path)))

    assert out.status == "SUCCESS", out.error
    assert fake_llm.count("rows") > 5  # 452 lines split into ~60-line chunks
    assert [e["unit"]["value"] for e in out.entries] == [f"U{u:03d}" for u in range(150)]
    first = out.entries[0]
    assert first["property"]["value"] == "Willow Creek"  # copied into every row
    assert first["rent"] == {"value": 1000, "citation": "[Sheet1!C3] 1000", "confidence": 0.95,
                             "source_locations": ["Sheet1!C3"]}
    # Row prompts carry only row-scope fields; the document prompt only document ones.
    _, row_msgs = next(c for c in fake_llm.calls if c[0] == "rows")
    assert '"key": "rent"' in row_msgs[1]["content"]
    assert '"key": "property"' not in row_msgs[1]["content"]


def test_a_failed_chunk_gives_partial_not_failed(tmp_path, fake_llm):
    fake_llm.prep = lambda m: PREP
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] == 2:
            return llm.LLMError("all models failed: 429")
        return _rows_from_prompt(messages)

    fake_llm.rows = flaky
    out = pipeline.run(_ctx(_csv(tmp_path)))
    assert out.status == "PARTIAL"
    assert out.error["code"] == "PARTIAL_EXTRACTION" and "rows chunk" in out.error["message"]
    assert 0 < len(out.entries) < 150


def test_every_call_failing_still_returns_valid_partial_result(tmp_path, fake_llm):
    boom = llm.LLMError("down")
    fake_llm.prep = fake_llm.doc = fake_llm.rows = lambda m: boom
    out = pipeline.run(_ctx(_csv(tmp_path, units=3)))
    assert out.status == "PARTIAL"
    assert len(out.entries) == 1 and set(out.entries[0]) == set(SCHEMA["fields"])
    assert all(f["value"] is None and f["citation"] == "" for f in out.entries[0].values())


def test_hallucinated_rows_are_nulled(tmp_path, fake_llm):
    fake_llm.prep = lambda m: PREP
    fake_llm.rows = lambda m: {"rows": [{"unit": ["U000", "[A3] U000", 1],
                                         "rent": [99999, "[C3] 99999", 0.9]}]}
    out = pipeline.run(_ctx(_csv(tmp_path, units=2)))
    assert out.entries[0]["unit"]["value"] == "U000"
    assert out.entries[0]["rent"]["value"] is None


def test_no_primary_means_one_document_call_and_one_entry(tmp_path, fake_llm):
    schema = {"name": "lease", "fields": {"tenant": {"name": "Tenant", "description": "t"}}}
    p = tmp_path / "lease.csv"
    p.write_text("Tenant,Acme Corp\n")
    fake_llm.doc = lambda m: {"fields": {"tenant": {"value": "Acme Corp",
                                                    "citation": "[Sheet1!B1] Acme Corp",
                                                    "confidence": 0.9}}}
    out = pipeline.run(_ctx(p, schema=schema))
    assert out.status == "SUCCESS"
    assert fake_llm.count("rows") == 0 and fake_llm.count("doc") == 1
    assert out.entries == [{"tenant": {"value": "Acme Corp", "citation": "[Sheet1!B1] Acme Corp",
                                       "confidence": 0.9, "source_locations": ["Sheet1!B1"]}}]


def test_field_prep_is_one_call_per_schema(tmp_path, fake_llm):
    p = _csv(tmp_path, units=2)
    fake_llm.prep = lambda m: PREP
    pipeline.run(_ctx(p))
    pipeline.run(_ctx(p))
    assert fake_llm.count("prep") == 1


def test_scanned_page_is_transcribed_and_cited(tmp_path, fake_llm):
    doc = pymupdf.open()
    doc.new_page()  # no text layer: the parser renders it as an image
    path = tmp_path / "scan.pdf"
    doc.save(path)
    schema = {"name": "lease", "fields": {"rent": {"name": "Rent", "description": "Base rent."}}}
    fake_llm.prep = lambda m: {"fields": {"rent": {"type": "number"}}}
    fake_llm.transcribe = lambda m: {"text": "ARTICLE 3\nBase Rent: $4,250.00 per month"}

    def doc_call(messages):
        assert "Base Rent: $4,250.00" in messages[1]["content"]
        return {"fields": {"rent": {"value": 4250, "citation": "[Page 1] Base Rent: $4,250.00",
                                    "confidence": 0.9}}}

    fake_llm.doc = doc_call
    out = pipeline.run(_ctx(path, "pdf", schema))
    assert fake_llm.count("transcribe") == 1
    _, msgs = next(c for c in fake_llm.calls if c[0] == "transcribe")
    assert msgs[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert out.entries[0]["rent"]["value"] == 4250
    assert out.entries[0]["rent"]["page_numbers"] == [1]
