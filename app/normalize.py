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

import math
import re
import time
from datetime import date, datetime
from typing import Any

from app.chunking import PAGE_MARKER, Section, parse_cells
from app.llm import Usage

TYPES = ("number", "percent", "date", "string", "boolean", "list")
_NULL_STRINGS = {"", "null", "none", "n/a", "na", "-", "--", "—", "–", "not stated",
                 "not found", "not applicable", "unknown"}

# --------------------------------------------------------------------------- field specs


def default_spec(key: str, field: dict, has_primary: bool) -> dict:
    return {
        "type": "string",
        "item_type": "string",
        "date_format": "YYYY-MM-DD",
        "scope": "row" if has_primary else "document",
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
        spec["computed"] = bool(raw.get("computed", False))
        if "default_if_absent" in raw:
            spec["default_if_absent"] = raw["default_if_absent"]
        if isinstance(raw.get("description"), str) and raw["description"].strip():
            spec["description"] = raw["description"].strip()
    if field.get("primary"):
        spec["scope"] = "row"
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


def verify_citation(citation: str, source: Source, sheet: str | None = None,
                    chunk_norm: str | None = None) -> tuple[bool, str, list[int], list[str]]:
    """Check every snippet of the citation against the source.
    Returns (verified, citation with sheet-qualified cell tags, page_numbers, source_locations)."""
    if not isinstance(citation, str) or not citation.strip():
        return False, "", [], []
    pages: list[int] = []
    locations: list[str] = []
    out_parts: list[str] = []
    current_page = None
    for part in _SPLIT.split(citation.strip()):
        if not part:
            continue
        tags, rest = [], part
        while m := _TAG.match(rest):
            tags.append(m.group(1).strip())
            rest = rest[m.end():]
        rest = rest.strip()
        cell_hit: tuple[str, str, str] | None = None  # (sheet, ref, cell text)
        new_tags = []
        for tag in tags:
            if pm := _PAGE.match(tag):
                current_page = int(pm.group(1))
                new_tags.append(f"Page {current_page}")
            elif rm := _REF.match(tag):
                found = source.cell(rm.group(1) or sheet, rm.group(2))
                if found:
                    cell_hit = (found[0], rm.group(2), found[1])
                    new_tags.append(f"{found[0]}!{rm.group(2)}")
                else:
                    new_tags.append(tag)
            else:
                new_tags.append(tag)
        snippet = norm_text(rest)
        cell_norm = norm_text(cell_hit[2]) if cell_hit else ""
        if not snippet:
            ok = cell_hit is not None  # a bare cell reference: the cell must exist
        elif cell_norm and (snippet in cell_norm or cell_norm in snippet):
            ok = True
        else:
            ok = (chunk_norm is not None and snippet in chunk_norm) or snippet in source.full_norm
            if cell_hit and ok:
                # The text is real but the named cell doesn't hold it: drop the wrong ref.
                new_tags = [t for t in new_tags if t != f"{cell_hit[0]}!{cell_hit[1]}"]
                cell_hit = None
        if not ok:
            return False, "", [], []
        if cell_hit:
            locations.append(f"{cell_hit[0]}!{cell_hit[1]}")
        if current_page and current_page in source.pages:
            pages.append(current_page)
        out_parts.append("".join(f"[{t}] " for t in new_tags) + rest)
    if not out_parts:
        return False, "", [], []
    return True, " ... ".join(out_parts).strip(), sorted(set(pages)), list(dict.fromkeys(locations))


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
    value, citation, conf = unpack(raw)
    value = coerce(value, spec)
    if value is None or (isinstance(value, list) and value and all(x is None for x in value)):
        return null_field(conf)
    ok, cit, pages, locs = verify_citation(citation, source, sheet, chunk_norm)
    if not ok:
        if equals_default(value, spec):
            return {"value": value, "citation": "", "confidence": round(min(conf, 0.8), 3)}
        return null_field(0.3)  # unverifiable: a null costs less than a hallucination
    out = {"value": value, "citation": cit, "confidence": round(conf, 3)}
    if pages:
        out["page_numbers"] = pages
    if locs:
        out["source_locations"] = locs
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
