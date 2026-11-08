"""Extraction pipeline. Works for any field list; nothing here knows a field name.

    parse -> transcribe scanned pages (flash, vision) ┐
          -> field prep (flash, cached per schema)    ┘-> document-level call (full text)
                                                        -> row calls, one per ~60-line chunk
          -> normalize: coerce, verify citations, dedupe rows, assemble

Every model call goes through app.llm (retries, fallback, disk cache, usage accounting).
A call that still fails after its retries doesn't fail the job: the job returns what it has
with status PARTIAL and an error saying which steps were lost.
"""

import base64
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from app import chunking, llm, normalize, parsers
from app.llm import Usage

log = logging.getLogger("app.pipeline")

MODEL = os.environ.get("LLM_MODEL", llm.DEFAULT_MODEL)  # flash
VISION_MODEL = os.environ.get("LLM_VISION_MODEL", llm.DEFAULT_MODEL)
# Pro / sonnet only when someone sets this on purpose, never by fallback.
ALLOW_EXPENSIVE = os.environ.get("LLM_ALLOW_EXPENSIVE", "0") == "1"
CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", 4))  # parallel calls per job
DOC_TEXT_MAX_CHARS = int(os.environ.get("DOC_TEXT_MAX_CHARS", 150_000))

# Output ceilings per call type. Hidden reasoning counts against these, so a runaway reply
# is cut off instead of billed in full.
PREP_MAX_TOKENS = 2000
ROWS_MAX_TOKENS = 8000
DOC_MAX_TOKENS = 6000
TRANSCRIBE_MAX_TOKENS = 4000
PREP_BATCH = 15  # fields per field-prep call, so each reply fits in PREP_MAX_TOKENS
# Rough output tokens per field in a row record ([value, "[F12] text", 0.9] plus key).
_TOKENS_PER_ROW_FIELD = 20


@dataclass
class JobContext:
    job_id: str
    file_path: Path
    file_type: str  # pdf | docx | xlsx | csv
    filename: str | None
    schema: dict
    set_stage: Callable[[str], None]
    usage: Usage = field(default_factory=Usage)


@dataclass
class PipelineOutput:
    status: str  # SUCCESS or PARTIAL
    entries: list[dict[str, Any]]
    error: dict | None = None  # which steps were lost, for PARTIAL


class _Errors:
    def __init__(self):
        self._items: list[str] = []
        self._budget: str | None = None
        self._lock = threading.Lock()

    def add(self, step: str, exc: Exception) -> None:
        with self._lock:
            if isinstance(exc, llm.BudgetExceeded):
                if self._budget is None:
                    log.error("step %s stopped: %s", step, exc)
                    self._budget = str(exc)
                self._items.append(f"{step}: skipped, budget spent")
                return
            log.error("step %s failed: %s", step, exc)
            self._items.append(f"{step}: {str(exc)[:300]}")

    def __bool__(self) -> bool:
        return bool(self._items)

    def as_error(self) -> dict:
        detail = f"{len(self._items)} step(s) failed: " + " | ".join(self._items)
        if self._budget:
            return {"code": "BUDGET_EXCEEDED", "message": f"{self._budget}; {detail}"[:2000]}
        return {"code": "PARTIAL_EXTRACTION", "message": detail[:2000]}


# --------------------------------------------------------------------------- prompts

TRANSCRIBE_SYSTEM = """You transcribe scanned document pages.
Write out all text on the page exactly as printed, in reading order. Keep numbers, dates,
punctuation and capitalisation exactly. Put each table row on its own line with cells
separated by " | ". Include handwritten text and marginal notes, marked [handwritten: ...].
Do not summarise, correct or add anything. Return JSON: {"text": "<transcription>"}"""

PREP_SYSTEM = """You prepare a field list for a document-extraction model. For every field key
return an object with:
- "type": one of "number", "percent", "date", "string", "boolean", "list".
  "percent" is a number the document prints as a percentage, kept as printed (5.26 for 5.26%).
  Counts, money, years and areas are "number".
- "item_type": for "list" fields, the type of each item (same choices, except "list").
- "date_format": "YYYY-MM-DD" unless the description asks for another format ("YYYY-MM").
- "scope": "document" if the value is the same for the whole document (header values,
  anything described as "same for every row"), "row" if it differs per table row. If the
  field list has no primary field, every field is "document". The primary field is "row".
- "kind": how the value relates to the document.
  "extracted": printed in the document and read as is.
  "calculated": computed, counted, summed, classified or otherwise derived from other
  values (including choosing a category from a fixed list).
  "inherited": copied from a heading or header above the record, e.g. the section a line
  sits under. The primary field is always "extracted".
- "calc": for "calculated" fields, the operation: "count_nonzero" (how many of the inputs
  are non-zero), "count" (how many inputs), "sum", "difference" (first minus second),
  "months_between" (whole months from a start date to an end date), or "other" (anything
  else, e.g. a classification or a schedule total). For other kinds use "other".
- "default_if_absent": the value the description says to use when the thing is absent
  (0 for "0 if none", [] for "empty list if none", false for "false otherwise"); null if the
  description says null or says nothing.
- "description": the description rewritten for an extraction model, in at most 40 words:
  precise about where the value is, its unit, sign and format, and what not to confuse it
  with. Keep every instruction of the original. Do not invent facts about any document.
Return JSON: {"fields": {"<key>": {...}, ...}} with exactly the keys given."""

_RULES = """Rules:
- Use only what the document explicitly says. If a value is not stated, return null. Never
  guess, never fill in from typical values or outside knowledge. A null is better than a
  plausible guess.
- citation: a short verbatim copy (under 150 characters) of the document text that holds
  the value, starting with its location tag exactly as it appears in the text: "[Page 3]
  Base Rent: $4,250.00" for pages, {cell_hint}. Several sources: join them with " ... ".
  Empty string when the value is null. Never describe the source in your own words
  ("computed from ...", "see above"): quote it, or name its cells.
- Each field has a kind:
  extracted: cite the text that holds the value.
  calculated: compute only from values stated in the document, and cite every input you
  used, each with its location tag, joined by " ... ". A run of spreadsheet cells may be
  cited as a range, e.g. "[C13:N13]". For a count, cite all the cells you counted over,
  zeros included. For a date span, cite both dates.
  inherited: the value comes from a heading above the record (e.g. its section); cite that
  heading verbatim with its location tag. Headings above this part are given in CONTEXT.
- confidence: your probability (0 to 1) that the value is correct.
- Formats: number and percent as JSON numbers (no $, %, or thousands separators; parentheses
  or a minus sign mean negative); dates in the field's date_format; boolean as true/false;
  list as a JSON array."""

DOC_SYSTEM = ("You extract document-level fields from a document.\n"
              + _RULES.format(cell_hint='"[SheetName!B7] 4250" for spreadsheet cells')
              + '\nReturn JSON: {"fields": {"<key>": {"value": ..., "citation": "...", '
                '"confidence": 0.9}, ...}} with every requested key.')

ROW_SYSTEM = ("You extract table records from part of a document.\n"
              + _RULES.format(cell_hint='"[F12] 4250" (cell reference, then the cell text) '
                                        'for spreadsheet cells')
              + """
- Return one record per distinct value of the primary field that starts in ROWS. A record
  may span several lines (e.g. a unit line followed by its charge lines): fold them into one.
- CONTEXT (column headers, the headings above this part, and the lines just before ROWS)
  is for reference only: never return a record that starts in CONTEXT.
- Follow the document description to decide what is a record. Totals, subtotals, section
  headers and summary tables are not records unless the description says so.
Return JSON: {"rows": [{"<key>": [value, "citation", confidence], ...}, ...]} with every
requested key in every record, or {"rows": []} if ROWS holds no records.""")


def _field_list(schema: dict, specs: dict, keys: list[str]) -> str:
    items = []
    for k in keys:
        s = specs[k]
        item = {"key": k, "name": schema["fields"][k].get("name", k), "type": s["type"],
                "description": s["description"]}
        if s["type"] == "list":
            item["item_type"] = s["item_type"]
        if s["type"] == "date" or s.get("item_type") == "date":
            item["date_format"] = s["date_format"]
        item["kind"] = s.get("kind", "extracted")
        if item["kind"] == "calculated":
            item["calc"] = s.get("calc", "other")
        if schema["fields"][k].get("primary"):
            item["primary"] = True
        items.append(item)
    return json.dumps(items, ensure_ascii=False, indent=1)


def _doc_header(schema: dict) -> str:
    return (f"DOCUMENT TYPE: {schema.get('name', '')}\n"
            f"DESCRIPTION: {schema.get('description', '')}")


# --------------------------------------------------------------------------- steps


def transcribe_pages(doc, usage: Usage, errors: _Errors, pool: ThreadPoolExecutor) -> None:
    """Replace the text of each page that has an image (no usable text layer) with a
    transcription. The llm disk cache makes re-runs free."""
    by_page = {}
    for i, chunk in enumerate(doc.chunks):
        if chunk.location.startswith("[Page "):
            by_page[chunk.location] = i

    def one(page_no: int, png: bytes) -> tuple[int, str | None]:
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        messages = [
            {"role": "system", "content": TRANSCRIBE_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": f"Transcribe page {page_no}."},
                {"type": "image_url", "image_url": {"url": url}},
            ]},
        ]
        try:
            out = llm.chat_json(messages, usage=usage, model=VISION_MODEL,
                                max_tokens=TRANSCRIBE_MAX_TOKENS, allow_expensive=ALLOW_EXPENSIVE,
                                tag=f"transcribe p{page_no}")
        except llm.LLMError as exc:
            errors.add(f"transcribe page {page_no}", exc)
            return page_no, None
        text = out.get("text") if isinstance(out, dict) else None
        return page_no, text if isinstance(text, str) else None

    for page_no, text in pool.map(lambda pi: one(*pi), doc.images):
        idx = by_page.get(f"[Page {page_no}]")
        if text and idx is not None and len(text.strip()) > len(doc.chunks[idx].text.strip()):
            doc.chunks[idx].text = text


_prep_memo: dict[str, dict] = {}
_prep_lock = threading.Lock()


def prepare_fields(schema: dict, usage: Usage, errors: _Errors) -> dict[str, dict]:
    """One flash call per distinct field list: type, scope and a sharpened description per
    field. Memoised in-process and cached on disk (the llm cache key is the schema itself)."""
    fields = schema["fields"]
    has_primary = normalize.primary_key(schema) is not None
    canonical = json.dumps(schema, sort_keys=True, ensure_ascii=False)
    key = hashlib.sha256(canonical.encode()).hexdigest()
    with _prep_lock:
        if key in _prep_memo:
            return _prep_memo[key]

    # Large field lists go in batches, so each reply fits in PREP_MAX_TOKENS.
    raw: dict = {}
    ok = True
    keys = list(fields)
    header = {k: v for k, v in schema.items() if k != "fields"}
    for start in range(0, len(keys), PREP_BATCH):
        batch = {k: fields[k] for k in keys[start:start + PREP_BATCH]}
        part = json.dumps({**header, "has_primary_field": has_primary, "fields": batch},
                          sort_keys=True, ensure_ascii=False)
        try:
            out = llm.chat_json([
                {"role": "system", "content": PREP_SYSTEM},
                {"role": "user", "content": f"FIELD LIST:\n{part}"},
            ], usage=usage, model=MODEL, max_tokens=PREP_MAX_TOKENS,
                allow_expensive=ALLOW_EXPENSIVE, tag=f"prep {start // PREP_BATCH + 1}")
            got = out.get("fields", {}) if isinstance(out, dict) else {}
            raw.update(got if isinstance(got, dict) else {})
        except llm.LLMError as exc:
            errors.add(f"field prep batch {start // PREP_BATCH + 1}", exc)
            ok = False
    specs = {k: normalize.clean_spec(k, f, raw.get(k), has_primary) for k, f in fields.items()}
    if ok:
        with _prep_lock:
            _prep_memo[key] = specs
    return specs


def extract_document(schema, specs, keys, full_text, usage, errors) -> dict:
    user = (f"{_doc_header(schema)}\n\nFIELDS:\n{_field_list(schema, specs, keys)}\n\n"
            f"DOCUMENT TEXT:\n<<<\n{full_text}\n>>>")
    try:
        out = llm.chat_json([{"role": "system", "content": DOC_SYSTEM},
                             {"role": "user", "content": user}], usage=usage, model=MODEL,
                            max_tokens=DOC_MAX_TOKENS, allow_expensive=ALLOW_EXPENSIVE,
                            tag="document")
    except llm.LLMError as exc:
        errors.add("document fields", exc)
        return {}
    fields = out.get("fields", out) if isinstance(out, dict) else {}
    return fields if isinstance(fields, dict) else {}


def extract_rows(schema, specs, keys, primary, chunk: chunking.RowChunk, usage, errors) -> list:
    """Extract the records of one chunk. If the reply is cut off at max_tokens, split the
    chunk in half at a record boundary and extract each half, down to a single record."""
    context = "\n".join(chunk.header) if chunk.header else "(none)"
    where = f"sheet '{chunk.sheet}'" if chunk.sheet else "pages"
    user = (f"{_doc_header(schema)}\n\nPRIMARY FIELD: {primary}\n"
            f"FIELDS:\n{_field_list(schema, specs, keys)}\n\n"
            f"CONTEXT ({where}):\n<<<\n{context}\n>>>\n\n"
            f"ROWS ({where}):\n<<<\n{chunk.text}\n>>>")
    try:
        out = llm.chat_json([{"role": "system", "content": ROW_SYSTEM},
                             {"role": "user", "content": user}],
                            usage=usage, model=MODEL, max_tokens=ROWS_MAX_TOKENS,
                            allow_expensive=ALLOW_EXPENSIVE, tag=f"rows {chunk.label}")
    except llm.Truncated as exc:
        halves = chunk.halves()
        if halves is None:
            errors.add(f"rows chunk {chunk.label} ({where}), a single record", exc)
            return []
        log.warning("rows chunk %s (%d records) truncated; splitting into %d + %d records",
                    chunk.label, chunk.records, halves[0].records, halves[1].records)
        return [row for half in halves
                for row in extract_rows(schema, specs, keys, primary, half, usage, errors)]
    except llm.LLMError as exc:
        errors.add(f"rows chunk {chunk.label} ({where})", exc)
        return []
    rows = out.get("rows", []) if isinstance(out, dict) else out
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


# --------------------------------------------------------------------------- run


def run(ctx: JobContext) -> PipelineOutput:
    schema = ctx.schema
    errors = _Errors()
    primary = normalize.primary_key(schema)

    ctx.set_stage("PARSING")
    doc = parsers.parse(ctx.file_path, ctx.file_type)  # detected type, never the filename

    with ThreadPoolExecutor(CONCURRENCY, thread_name_prefix=f"job-{ctx.job_id[:8]}") as pool:
        ctx.set_stage("PREPARING")
        prep = pool.submit(prepare_fields, schema, ctx.usage, errors)
        if doc.images:
            transcribe_pages(doc, ctx.usage, errors, pool)
        specs = prep.result()

        sections = chunking.build_sections(doc)
        source = normalize.Source(sections)
        keys = list(schema["fields"])
        row_keys = [k for k in keys if primary and specs[k]["scope"] == "row"]
        doc_keys = [k for k in keys if k not in row_keys]

        ctx.set_stage("EXTRACTING")
        doc_future = None
        if doc_keys:
            full = chunking.render_full(sections, DOC_TEXT_MAX_CHARS)
            doc_future = pool.submit(extract_document, schema, specs, doc_keys, full,
                                     ctx.usage, errors)
        chunks: list[chunking.RowChunk] = []
        if row_keys:
            labels = [primary, schema["fields"][primary].get("name", "")]
            # Few enough records per chunk that the reply fits in ROWS_MAX_TOKENS (with
            # headroom): a wide table gets smaller chunks than a narrow one.
            max_records = max(3, int(ROWS_MAX_TOKENS * 0.7)
                              // (len(row_keys) * _TOKENS_PER_ROW_FIELD + 10))
            for sec in sections:
                chunks.extend(chunking.split_rows(sec, labels, max_records=max_records))
        row_futures = [pool.submit(extract_rows, schema, specs, row_keys, primary, c,
                                   ctx.usage, errors) for c in chunks]
        raw_doc = doc_future.result() if doc_future else {}
        raw_rows = [f.result() for f in row_futures]

    ctx.set_stage("NORMALIZING")
    doc_fields = {k: normalize.finalize_field(raw_doc.get(k), specs[k], source) for k in doc_keys}
    rows = []
    for chunk, chunk_rows in zip(chunks, raw_rows):
        chunk_norm = normalize.norm_text("\n".join(chunk.header + chunk.lines))
        for r in chunk_rows:
            rows.append({k: normalize.finalize_field(r.get(k), specs[k], source, chunk.sheet,
                                                     chunk_norm) for k in row_keys})
    if primary:
        rows = normalize.dedupe_rows(rows, primary)
        log.info("job=%s %d chunks -> %d rows", ctx.job_id, len(chunks), len(rows))
    entries = normalize.assemble(schema, doc_fields, rows)

    if errors:
        return PipelineOutput("PARTIAL", entries, errors.as_error())
    if primary and row_keys and not rows:
        return PipelineOutput("PARTIAL", entries, {"code": "NO_ROWS",
                                                   "message": "no table rows were found"})
    return PipelineOutput("SUCCESS", entries)
