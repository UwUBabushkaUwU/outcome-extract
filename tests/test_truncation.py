"""A row-chunk reply cut off at max_tokens is retried as two halves, split at a record
boundary, recursively down to one record. Real llm.chat_json on a mocked client."""

import json
import re
from types import SimpleNamespace

import pytest

from app import chunking, llm, pipeline
from app.chunking import Section, split_rows
from app.pipeline import JobContext

SCHEMA = {
    "name": "units", "description": "One row per unit.",
    "fields": {
        "unit": {"name": "Unit", "description": "Unit id.", "primary": True},
        "rent": {"name": "Rent", "description": "Rent amount."},
    },
}
PREP = {"fields": {"unit": {"type": "string", "scope": "row"},
                   "rent": {"type": "number", "scope": "row"}}}
UNIT_LINE = re.compile(r"Row (\d+): \[A\d+\] (U\d{3}) \| \[B\d+\] Rent \| \[C\d+\] (\d+)")


def _csv(path, units):
    lines = ["Unit,Charge,Amount"]
    for u in range(units):
        lines += [f"U{u:03d},Rent,{1000 + u}", ",Fee,25", ",Fee,10"]
    path.write_text("\n".join(lines) + "\n")
    return path


def _reply(content, finish="stop"):
    usage = SimpleNamespace(prompt_tokens=500, completion_tokens=100, cost=0.0)
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage, model_extra={})


class TruncatingClient:
    """Answers row calls with every unit in ROWS, but stops at max_tokens (finish=length)
    whenever ROWS holds more than `fits` units, as a model with a long reply would."""

    def __init__(self, fits, always_truncate=()):
        self.fits = fits
        self.always_truncate = set(always_truncate)  # units too big to ever fit
        self.row_requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        system, user = params["messages"][0]["content"], params["messages"][1]["content"]
        if system == pipeline.PREP_SYSTEM:
            return _reply(json.dumps(PREP))
        if system != pipeline.ROW_SYSTEM:
            return _reply('{"fields": {}}')
        rows_text = user.split("ROWS (", 1)[1]
        units = UNIT_LINE.findall(rows_text)
        self.row_requests.append(rows_text)
        if len(units) > self.fits or self.always_truncate & {u for _, u, _ in units}:
            return _reply('{"rows": [{"unit": ["U0', finish="length")
        rows = [{"unit": [u, f"[A{r}] {u}", 0.99], "rent": [int(rent), f"[C{r}] {rent}", 0.9]}
                for r, u, rent in units]
        return _reply(json.dumps({"rows": rows}))


@pytest.fixture
def install(monkeypatch):
    monkeypatch.setattr(llm, "CACHE_ENABLED", False)
    pipeline._prep_memo.clear()

    def go(client):
        monkeypatch.setattr(llm, "_get_client", lambda: client)
        return client
    return go


def _run(path):
    return pipeline.run(JobContext("job", path, "csv", path.name, SCHEMA, lambda s: None))


def test_truncated_chunk_is_split_until_it_fits(tmp_path, install):
    client = install(TruncatingClient(fits=2))
    out = _run(_csv(tmp_path / "rr.csv", units=40))

    assert out.status == "SUCCESS", out.error
    assert [e["unit"]["value"] for e in out.entries] == [f"U{u:03d}" for u in range(40)]
    assert out.entries[7]["rent"]["value"] == 1007
    # Every request (whole chunk or half) starts on a unit line: no record was ever cut.
    for rows_text in client.row_requests:
        first = rows_text.split("<<<\n", 1)[1].splitlines()[0]
        assert UNIT_LINE.match(first) or first.startswith("Row 1:"), first
    # Some requests were truncated and retried as halves of at most 2 units that succeeded.
    sizes = [len(UNIT_LINE.findall(t)) for t in client.row_requests]
    assert max(sizes) > 2 and min(sizes) >= 1


def test_a_single_record_that_never_fits_gives_partial_and_keeps_the_rest(tmp_path, install):
    client = install(TruncatingClient(fits=3, always_truncate={"U005"}))
    out = _run(_csv(tmp_path / "rr.csv", units=12))

    assert out.status == "PARTIAL"
    assert "single record" in out.error["message"] and "truncated" in out.error["message"]
    got = [e["unit"]["value"] for e in out.entries]
    assert "U005" not in got
    assert got == [f"U{u:03d}" for u in range(12) if u != 5]
    # It went all the way down: one request held U005 alone.
    alone = [t for t in client.row_requests if [u for _, u, _ in UNIT_LINE.findall(t)] == ["U005"]]
    assert alone


def test_halves_split_at_record_boundaries_with_context():
    lines = ["Row 1: [A1] Unit | [B1] Charge | [C1] Amount"]
    for u in range(6):
        r = 2 + u * 3
        lines += [f"Row {r}: [A{r}] U{u} | [B{r}] Rent | [C{r}] 1000",
                  f"Row {r + 1}: [B{r + 1}] Fee | [C{r + 1}] 25",
                  f"Row {r + 2}: [B{r + 2}] Fee | [C{r + 2}] 10"]
    chunk = split_rows(Section("sheet", "S", lines), ["unit", "Unit"], target=1000)[0]
    assert chunk.records == 6

    a, b = chunk.halves()
    assert (a.records, b.records) == (3, 3)
    assert a.lines + b.lines == chunk.lines  # nothing lost or duplicated
    assert b.lines[0].startswith("Row 11: [A11] U3")
    assert b.header[0] == "Row 1: [A1] Unit | [B1] Charge | [C1] Amount"  # column header kept
    assert "Row 10: [B10] Fee | [C10] 10" in b.header  # the lines just before, as context

    leaf = chunk
    while (h := leaf.halves()) is not None:
        leaf = h[1]
    assert leaf.records == 1 and leaf.label.startswith("0.2")
    assert all(chunking.parse_cells(leaf.lines[s]).get("A") for s in leaf.starts)


def test_halves_of_a_page_chunk_keep_their_page():
    lines = ["[Page 1]"]
    for i in range(8):
        if i == 2:
            lines.append("[Page 2]")
        lines += [f"61{i:02d}-0000 Item {i} 1,000", "wrapped commentary"]
    chunk = split_rows(Section("pages", None, lines), ["gl"], target=1000)[0]
    a, b = chunk.halves()
    assert b.lines[0] == "[Page 2]" and b.lines[1].startswith("6104-0000")
    assert a.lines + b.lines[1:] == chunk.lines
