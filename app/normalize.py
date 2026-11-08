"""Turn raw model output into contract-shaped results. Nothing here calls a model.

* coerce: numbers ($, commas stripped, (x) = negative, % dropped), dates (YYYY-MM-DD, or
  YYYY-MM when the field asks for it), booleans, strings, lists of those.
* verify: a value survives only if its citation is found in the source text. The one
  exception is the field's own "default when absent" (0 for "0 if none", [] for "empty list
  if none", false for "false otherwise"), which by definition has nothing to cite.
* assemble: copy document-level fields into every row, make sure every schema field is
  present, empty citation for nulls.
* build_metadata: model_calls, tokens, cost, latency for the result.
"""

import logging
import math
import re
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import date, datetime, timedelta
from typing import Any

from app.chunking import PAGE_MARKER, Section, parse_cells
from app.llm import Usage

log = logging.getLogger("app.normalize")

TYPES = ("number", "percent", "date", "string", "boolean", "list")
# How a field's value relates to the document, from field prep:
#   extracted  - printed in the document; strict citation check.
#   calculated - computed, counted, classified or derived; must cite its inputs, and is
#                recomputed from them when CALC_OPS covers it.
#   inherited  - copied from a heading above the record (its section); checked against
#                the chunk's section context.
KINDS = ("extracted", "calculated", "inherited")
CALC_OPS = ("count_nonzero", "count", "sum", "difference", "months_between", "other")
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "-", "--", "—", "–", "not stated",
                 "not found", "not applicable", "unknown"}

# --------------------------------------------------------------------------- field specs


def default_spec(key: str, field: dict, has_primary: bool) -> dict:
    return {
        "type": "string",
        "item_type": "string",
        "date_format": "YYYY-MM-DD",
        "scope": "row" if has_primary else "document",
        "kind": "extracted",
        "calc": "other",
        "computed": False,
        "default_if_absent": None,
        "description": field.get("description", ""),
    }


def clean_spec(key: str, field: dict, raw: Any, has_primary: bool) -> dict:
    """Validate one field spec from the field-prep model, falling back field by field."""
    spec = default_spec(key, field, has_primary)
    if isinstance(raw, dict):
        if raw.get("type") in TYPES:
            spec["type"] = raw["type"]
        if raw.get("item_type") in TYPES and raw["item_type"] != "list":
            spec["item_type"] = raw["item_type"]
        if raw.get("date_format") in ("YYYY-MM-DD", "YYYY-MM", "YYYY"):
            spec["date_format"] = raw["date_format"]
        if raw.get("scope") in ("document", "row") and has_primary:
            spec["scope"] = raw["scope"]
        if raw.get("kind") in KINDS:
            spec["kind"] = raw["kind"]
        elif raw.get("computed"):  # older field-prep replies
            spec["kind"] = "calculated"
        if raw.get("calc") in CALC_OPS:
            spec["calc"] = raw["calc"]
        if "default_if_absent" in raw:
            spec["default_if_absent"] = raw["default_if_absent"]
        if isinstance(raw.get("description"), str) and raw["description"].strip():
            spec["description"] = raw["description"].strip()
    if field.get("primary"):
        spec["scope"] = "row"
        spec["kind"] = "extracted"  # the row's identity is always read, never derived
    spec["computed"] = spec["kind"] == "calculated"
    # The default must itself be a valid value for the type, or it is ignored.
    if spec["default_if_absent"] is not None:
        spec["default_if_absent"] = coerce(spec["default_if_absent"], spec)
    return spec


# --------------------------------------------------------------------------- coercion


def _is_nullish(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return isinstance(v, str) and v.strip().lower() in _NULL_STRINGS


_NUM_TOKEN = re.compile(r"\(?\s*[-−–]?\s*\$?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*\)?")


def parse_number(v: Any) -> int | float | None:
    if isinstance(v, bool) or _is_nullish(v):
        return None
    if isinstance(v, (int, float)):
        num = float(v)
    else:
        tokens = _NUM_TOKEN.findall(str(v))
        if len(tokens) != 1:
            return None
        t = tokens[0].strip()
        digits = re.sub(r"[^\d.]", "", t)
        try:
            num = float(digits)
        except ValueError:
            return None
        if (t.startswith("(") and t.endswith(")")) or re.search(r"[-−–]", t):
            num = -num
    if math.isnan(num) or math.isinf(num):
        return None
    return int(num) if num.is_integer() else num


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _month(name: str) -> int | None:
    return _MONTHS.get(name[:3].lower()) if len(name) >= 3 else None


def _year(y: str) -> int:
    n = int(y)
    if len(y) == 2:
        n += 2000 if n < 50 else 1900
    return n


def parse_date(v: Any, fmt: str = "YYYY-MM-DD") -> str | None:
    if isinstance(v, bool) or _is_nullish(v):
        return None
    if isinstance(v, (datetime, date)):
        y, m, d = v.year, v.month, v.day
    else:
        s = str(v).strip()
        y = m = d = None
        pats = [
            (r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ].*)?", "ymd"),
            (r"(\d{4})/(\d{1,2})/(\d{1,2})", "ymd"),
            (r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4}|\d{2})", "mdy"),
            (r"(\d{4})-(\d{1,2})", "ym"),
            (r"(\d{1,2})-([A-Za-z]{3,9})-(\d{4}|\d{2})", "dMy"),
            (r".*?([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4}).*", "Mdy"),
            (r".*?(\d{1,2})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?([A-Za-z]{3,9})\.?,?\s+(\d{4}).*", "dMy"),
            (r".*?([A-Za-z]{3,9})\.?[\s,-]+(\d{4}).*", "My"),
            (r"(\d{4})", "y"),
        ]
        for pat, order in pats:
            mt = re.fullmatch(pat, s)
            if not mt:
                continue
            g = mt.groups()
            if order == "ymd":
                y, m, d = int(g[0]), int(g[1]), int(g[2])
            elif order == "mdy":
                m, d, y = int(g[0]), int(g[1]), _year(g[2])
            elif order == "ym":
                y, m = int(g[0]), int(g[1])
            elif order == "dMy":
                d, m, y = int(g[0]), _month(g[1]), _year(g[2])
            elif order == "Mdy":
                m, d, y = _month(g[0]), int(g[1]), int(g[2])
            elif order == "My":
                m, y = _month(g[0]), int(g[1])
            else:
                y = int(g[0])
            if m is None and order not in ("y",):
                y = m = d = None
                continue
            break
        if y is None:
            return None
    try:
        date(y, m or 1, d or 1)
    except (TypeError, ValueError):
        return None
    if fmt == "YYYY":
        return f"{y:04d}"
    if m is None:
        return None
    if fmt == "YYYY-MM":
        return f"{y:04d}-{m:02d}"
    if d is None:
        return None  # "Feb 2023" is not a full date; don't invent the day
    return f"{y:04d}-{m:02d}-{d:02d}"


def parse_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "y"):
            return True
        if s in ("false", "no", "n"):
            return False
    return None


def _string(v: Any) -> str | None:
    if _is_nullish(v):
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (list, tuple)):
        parts = [p for p in (_string(x) for x in v) if p]
        return "; ".join(parts) or None
    if isinstance(v, dict):
        return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s or None


def _scalar(v: Any, kind: str, fmt: str) -> Any:
    if kind in ("number", "percent"):
        return parse_number(v)
    if kind == "date":
        return parse_date(v, fmt)
    if kind == "boolean":
        return parse_bool(v)
    return _string(v)


def coerce(v: Any, spec: dict) -> Any:
    kind = spec["type"]
    if kind == "list":
        if v is None:
            return None
        if not isinstance(v, (list, tuple)):
            if _is_nullish(v):
                return None
            v = [v]
        # Keep None in place so parallel lists (start dates / end dates / rents) stay aligned.
        return [_scalar(x, spec["item_type"], spec["date_format"]) for x in v]
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) == 1 else (v if kind == "string" else None)
    return _scalar(v, kind, spec["date_format"])


def equals_default(value: Any, spec: dict) -> bool:
    default = spec.get("default_if_absent")
    if default is None or value is None:
        return False
    if isinstance(default, list):
        return value == [] and default == []
    return type(value) is type(default) and value == default


# --------------------------------------------------------------------------- source text

_TAG = re.compile(r"^\s*\[([^\[\]]+)\]\s*")
_CELL_TAG = re.compile(r"\[[A-Z]{1,3}\d+\]")
_REF = re.compile(r"^(?:(.+)!)?([A-Z]{1,3}\d+)$")
_PAGE = re.compile(r"^page\s+(\d+)$", re.IGNORECASE)
_SPLIT = re.compile(r"\s*(?:\.\.\.|…|;|\s\|\s)\s*")


def norm_text(s: str) -> str:
    """Comparison form: no cell tags or row labels, no $ or thousands commas, one space."""
    s = _CELL_TAG.sub(" ", s)
    s = re.sub(r"(?m)^Row \d+:", " ", s).lower()
    s = s.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-",
                                   "—": "-", "−": "-", "|": " ", "$": "", ",": "",
                                   " ": " "}))
    return re.sub(r"\s+", " ", s).strip()


class Source:
    """Searchable view of the document text the model saw."""

    def __init__(self, sections: list[Section]):
        self.cells: dict[tuple[str, str], str] = {}
        self.pages: set[int] = set()
        parts = []
        for sec in sections:
            for ln in sec.lines:
                if sec.kind == "sheet":
                    row = re.match(r"Row (\d+):", ln)
                    if row:
                        for col, text in parse_cells(ln).items():
                            self.cells[(sec.name, f"{col}{row.group(1)}")] = text
                elif m := PAGE_MARKER.match(ln):
                    self.pages.add(int(m.group(1)))
            parts.append("\n".join(sec.lines))
        self.sheets = [s.name for s in sections if s.kind == "sheet"]
        self.full_norm = norm_text("\n".join(parts))

    def cell(self, sheet: str | None, ref: str) -> tuple[str, str] | None:
        if sheet and (sheet, ref) in self.cells:
            return sheet, self.cells[(sheet, ref)]
        for name in self.sheets:
            if (name, ref) in self.cells:
                return name, self.cells[(name, ref)]
        return None


_RANGE_WORDS = re.compile(
    r"\[((?:[^\[\]!]+!)?)([A-Z]{1,3})(\d+)\]\s*(?:to|through|thru|-|–|—|:)\s*"
    r"\[(?:[^\[\]!]+!)?([A-Z]{1,3})(\d+)\]", re.IGNORECASE)
_ANY_TAG = re.compile(r"\[([^\[\]]+)\]")
_RANGE_REF = re.compile(r"^(?:(.+)!)?([A-Z]{1,3})(\d+):([A-Z]{1,3})(\d+)$")
_MAX_RANGE_CELLS = 500


@dataclass
class Check:
    """Result of checking a citation. `inputs` holds the text of every cited source, in
    order (cell text for cell references, the quoted text otherwise): what a calculated
    field is recomputed from and what an inherited value must appear in."""
    ok: bool
    citation: str = ""
    pages: list[int] = dc_field(default_factory=list)
    locations: list[str] = dc_field(default_factory=list)
    inputs: list[str] = dc_field(default_factory=list)
    input_cells: list[tuple[str, str] | None] = dc_field(default_factory=list)  # per input
    ranged: bool = False  # an explicit range (C13:N13, "C13 to N13") was cited


def _col_name(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _col_num(col: str) -> int:
    n = 0
    for ch in col.upper():
        n = n * 26 + ord(ch) - 64
    return n


def _cells_for(tag: str, source: Source, sheet: str | None) -> list[tuple[str, str, str]] | None:
    """Cells named by a tag: 'F12', 'Sheet!F12', 'C13:N13'. None if the tag isn't a cell
    reference at all; [] if it is one but names no existing cell."""
    if m := _REF.match(tag):
        found = source.cell(m.group(1) or sheet, m.group(2))
        return [(found[0], m.group(2), found[1])] if found else []
    if m := _RANGE_REF.match(tag):
        sh, c1, r1, c2, r2 = m.groups()
        cols = range(min(_col_num(c1), _col_num(c2)), max(_col_num(c1), _col_num(c2)) + 1)
        rows = range(min(int(r1), int(r2)), max(int(r1), int(r2)) + 1)
        if len(cols) * len(rows) > _MAX_RANGE_CELLS:
            return []
        out = []
        for r in rows:
            for c in cols:
                ref = f"{_col_name(c)}{r}"
                if found := source.cell(sh or sheet, ref):
                    out.append((found[0], ref, found[1]))
        return out  # blank cells are absent from the source: no value, nothing to add
    return None


def _only_page_tags(part: str) -> bool:
    rest, saw = part.strip(), False
    while m := _TAG.match(rest):
        if not _PAGE.match(m.group(1).strip()):
            return False
        saw, rest = True, rest[m.end():].strip()
    return saw and not rest


def verify_citation(citation: str, source: Source, sheet: str | None = None,
                    chunk_norm: str | None = None, mode: str = "extracted") -> Check:
    """Check every part of a citation (parts are joined by ' ... ', ';' or ' | ').

    mode "extracted": each part is location tag(s) plus text that must be verbatim in the
      source (in the named cell, or else in the chunk or document text).
    mode "calculated": as extracted, but a part may also name its inputs by cell reference
      or range ("[C13:N13]", "[C13] to [N13]") amid prose; the named cells must exist and
      are the inputs. The prose is not checked and not kept.
    mode "inherited": text must be in the chunk's own text and context (its section
      headings), not merely somewhere in the document.
    """
    if not isinstance(citation, str) or not citation.strip():
        return Check(False)
    text = _RANGE_WORDS.sub(lambda m: f"[{m.group(1)}{m.group(2)}{m.group(3)}:"
                                      f"{m.group(4)}{m.group(5)}]", citation.strip())
    parts, carry = [], ""
    for p in _SPLIT.split(text):
        if not p or not p.strip():
            continue
        if _only_page_tags(p):  # "[Page 18] | 60 | $ 123,084": the tag is for what follows
            carry += p.strip() + " "
            continue
        parts.append(carry + p)
        carry = ""
    if not parts:
        return Check(False)

    if mode == "inherited" and chunk_norm is not None:
        haystacks = [chunk_norm]
    else:
        haystacks = [h for h in (chunk_norm, source.full_norm) if h is not None]
    pages, locations, out_parts, inputs, input_cells = [], [], [], [], []
    ranged = False
    current_page = None
    for part in parts:
        tags, rest = [], part
        while m := _TAG.match(rest):
            tags.append(m.group(1).strip())
            rest = rest[m.end():]
        rest = rest.strip()
        if mode == "calculated":
            inline = [t for t in _ANY_TAG.findall(rest)
                      if _cells_for(t, source, sheet) is not None]
            if inline:  # "Computed from [C13] to [N13]": the cells are the evidence
                tags += inline
                rest = ""
        cells: list[tuple[str, str, str]] = []
        shown_tags = []
        for tag in tags:
            if pm := _PAGE.match(tag):
                current_page = int(pm.group(1))
                shown_tags.append(f"Page {current_page}")
                continue
            found = _cells_for(tag, source, sheet)
            if found is None or (not found and mode != "calculated"):
                shown_tags.append(tag)  # not a cell, or a cell that's empty: the text decides
            elif not found:
                return Check(False)  # a calculated input naming no existing cell
            else:
                cells.extend(found)
                ranged = ranged or _RANGE_REF.match(tag) is not None
                shown_tags.append(f"{found[0][0]}!{tag.split('!')[-1]}")
        snippet = norm_text(rest)
        if not snippet:
            if not cells:
                return Check(False)  # a bare page tag proves nothing
            inputs.extend(c[2] for c in cells)
            input_cells.extend((c[0], c[1]) for c in cells)
        elif (len(cells) == 1 and (cn := norm_text(cells[0][2]))
              and (snippet in cn or cn in snippet)):
            if mode == "inherited" and chunk_norm is not None and cn not in chunk_norm:
                return Check(False)  # a real heading, but not one above this chunk
            inputs.append(cells[0][2])
            input_cells.append((cells[0][0], cells[0][1]))
        elif any(snippet in h for h in haystacks):
            if cells:  # the text is real but the named cell doesn't hold it: drop the ref
                shown_tags = [t for t in shown_tags if "!" not in t]
                cells = []
            inputs.append(rest)
            input_cells.append(None)
        else:
            return Check(False)
        locations.extend(f"{c[0]}!{c[1]}" for c in cells)
        if current_page and current_page in source.pages:
            pages.append(current_page)
        out_parts.append(("".join(f"[{t}] " for t in shown_tags) + rest).strip())
    return Check(True, " ... ".join(out_parts), sorted(set(pages)),
                 list(dict.fromkeys(locations)), inputs, input_cells, ranged)


def span_inputs(check: Check, source: Source) -> list[str] | None:
    """The other reading of "[C13] x ... [N13] y": every cell from the first cited cell to
    the last, when all cited cells sit in one row or one column. None if not applicable."""
    cells = check.input_cells
    if check.ranged or len(cells) < 2 or any(c is None for c in cells):
        return None
    if len({c[0] for c in cells}) != 1:
        return None
    sheet = cells[0][0]
    pos = [(_col_num(re.match(r"[A-Z]+", r).group()), int(re.search(r"\d+", r).group()))
           for _, r in cells]
    cols, rows = {p[0] for p in pos}, {p[1] for p in pos}
    if len(rows) == 1:
        r = rows.pop()
        refs = [f"{_col_name(c)}{r}" for c in range(min(cols), max(cols) + 1)]
    elif len(cols) == 1:
        c = _col_name(cols.pop())
        refs = [f"{c}{r}" for r in range(min(rows), max(rows) + 1)]
    else:
        return None
    if len(refs) <= len(cells) or len(refs) > _MAX_RANGE_CELLS:
        return None  # nothing between the cited cells: both readings are the same
    return [found[1] for ref in refs if (found := source.cell(sheet, ref))]


# --------------------------------------------------------------------------- recompute

_DATE_IN_TEXT = re.compile(
    r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?[A-Za-z]{3,9}\.?,?\s+\d{4}")


def find_dates(text: str) -> list[date]:
    out = []
    for m in _DATE_IN_TEXT.finditer(text):
        if iso := parse_date(m.group(0)):
            out.append(date.fromisoformat(iso))
    return out


def months_between(start: date, end: date) -> int:
    """Whole months of a term from its first day to its last day, both inclusive:
    2010-08-01 .. 2021-02-28 is 127 months."""
    after = end + timedelta(days=1)
    return round((after.year - start.year) * 12 + (after.month - start.month)
                 + (after.day - start.day) / 30.44)


def recompute(op: str, inputs: list[str]) -> int | float | None:
    """The value a calculated field should have, from its cited inputs; None if the op
    can't be applied to them (then the model's value stands, its inputs being verified)."""
    if not inputs:
        return None
    if op == "count":
        return len(inputs)
    if op in ("count_nonzero", "sum", "difference"):
        nums = [parse_number(x) for x in inputs]
        if any(n is None for n in nums):
            return None
        if op == "count_nonzero":
            return sum(1 for n in nums if n != 0)
        if op == "difference":
            return parse_number(round(nums[0] - nums[1], 6)) if len(nums) == 2 else None
        return parse_number(round(sum(nums), 6))
    if op == "months_between":
        dates = [d for x in inputs for d in find_dates(x)]
        if len(dates) < 2 or dates[-1] <= dates[0]:
            return None
        return months_between(dates[0], dates[-1])
    return None


# --------------------------------------------------------------------------- fields


def null_field(confidence: float = 0.5) -> dict:
    return {"value": None, "citation": "", "confidence": round(_clamp(confidence), 3)}


def _clamp(x: Any, default: float = 0.5) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return 0.0 if math.isnan(x) else min(1.0, max(0.0, x))


def unpack(raw: Any) -> tuple[Any, str, float]:
    """Accept {"value","citation","confidence"}, [value, citation, confidence] or a bare value."""
    if isinstance(raw, dict):
        return raw.get("value"), raw.get("citation") or "", _clamp(raw.get("confidence"))
    if isinstance(raw, (list, tuple)) and len(raw) == 3 and isinstance(raw[1], str):
        return raw[0], raw[1], _clamp(raw[2])
    return raw, "", 0.5


def finalize_field(raw: Any, spec: dict, source: Source, sheet: str | None = None,
                   chunk_norm: str | None = None) -> dict:
    """Coerce, check the citation the way the field's kind requires, recompute calculated
    values from their cited inputs where possible. Unsupported values become null."""
    value, citation, conf = unpack(raw)
    value = coerce(value, spec)
    if value is None or (isinstance(value, list) and value and all(x is None for x in value)):
        return null_field(conf)
    kind = spec.get("kind", "extracted")
    check = verify_citation(citation, source, sheet, chunk_norm, mode=kind)
    if check.ok and kind == "inherited" and isinstance(value, str):
        # The heading cited must actually be where the value comes from.
        if norm_text(value) not in norm_text(" ".join(check.inputs)):
            check = Check(False)
    if not check.ok:
        if equals_default(value, spec):
            return {"value": value, "citation": "", "confidence": round(min(conf, 0.8), 3)}
        return null_field(0.3)  # unverifiable: a null costs less than a hallucination
    if kind == "calculated" and spec["type"] in ("number", "percent"):
        op = spec.get("calc", "other")
        listed = recompute(op, check.inputs)
        span = span_inputs(check, source)
        spanned = recompute(op, span) if span is not None else None
        if listed is not None and value not in (listed, spanned):
            if spanned is not None and spanned != listed:
                # "[C13] ... [N13]" can mean those two cells or everything between them;
                # the readings disagree and the model's value matches neither.
                log.info("%s ambiguous: model %r, cited cells %r, cells between them %r",
                         op, value, listed, spanned)
                return null_field(0.3)
            log.info("recomputed %s: model said %r, cited inputs give %r", op, value, listed)
            value, conf = listed, min(conf, 0.9)
    out = {"value": value, "citation": check.citation, "confidence": round(conf, 3)}
    if check.pages:
        out["page_numbers"] = check.pages
    if check.locations:
        out["source_locations"] = check.locations
    return out


# --------------------------------------------------------------------------- assembly


def primary_key(schema: dict) -> str | None:
    return next((k for k, f in schema["fields"].items() if f.get("primary")), None)


def _key_norm(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def dedupe_rows(rows: list[dict], primary: str) -> list[dict]:
    """Drop rows without a primary value, keep the first row for each primary value."""
    seen, out = set(), []
    for row in rows:
        v = row.get(primary, {}).get("value")
        if v is None or v == []:
            continue
        k = _key_norm(v)
        if k in seen:
            continue
        seen.add(k)
        out.append(row)
    return out


def assemble(schema: dict, doc_fields: dict[str, dict], rows: list[dict]) -> list[dict]:
    keys = list(schema["fields"])
    base_rows = rows or [{}]
    entries = []
    for row in base_rows:
        entry = {}
        for k in keys:
            f = row.get(k) or doc_fields.get(k) or null_field()
            if f["value"] is None:
                f = {"value": None, "citation": "", "confidence": f.get("confidence", 0.5)}
            entry[k] = dict(f)
        entries.append(entry)
    return entries


def latency_ms(created_at: str) -> int:
    created = datetime.fromisoformat(created_at)
    return max(0, int((time.time() - created.timestamp()) * 1000))


def build_metadata(usage: Usage, created_at: str) -> dict:
    return {**usage.to_metadata(), "latency_ms": latency_ms(created_at)}
