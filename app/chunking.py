"""Turn a parsed Document into text for the model.

* Sections: one per spreadsheet sheet; all pages of a PDF/DOCX together as one section, with
  a "[Page N]" marker line before each page.
* Lines are compacted: empty cells and runs of whitespace are dropped to save tokens.
* Row chunks (for table fields) hold about TARGET_LINES lines and are only ever cut where a
  new record starts, so a record that spans several lines (a rent-roll unit followed by its
  charge lines, a GL line followed by wrapped commentary) is never split across two calls.
  - Spreadsheets: a record starts where the primary-key column takes a new value. The key
    column is the header cell matching the primary field's name, else the leftmost data
    column. Repeats of the same value (merged cells) don't start a new record.
  - Pages: a record starts at a line whose first token has the document's dominant
    "code shape" (6101-0000 -> 9999-9999). With no such shape, pages are the units.
  Each chunk after the first gets context: the column-header lines and the last few lines of
  the previous chunk.
"""

import os
import re
from collections import Counter
from dataclasses import dataclass, field

from app.parsers.base import Document

TARGET_LINES = int(os.environ.get("ROW_CHUNK_LINES", 60))
HEADER_MAX_LINES = 15
CONTEXT_TAIL_LINES = 8
# A record larger than this is split anyway, so a mis-detected key column can't produce
# a single giant chunk.
MAX_BLOCK_LINES = TARGET_LINES * 4

_SHEET_LOC = re.compile(r"^\[(.+)![A-Z]+\d+:[A-Z]+\d+\]$")
_PAGE_LOC = re.compile(r"^\[Page (\d+)\]$")
PAGE_MARKER = re.compile(r"^\[Page (\d+)\]$")
_CELL = re.compile(r"\[([A-Z]{1,3})(\d+)\] ?")
_EMPTY_CELL = re.compile(r"\[[A-Z]{1,3}\d+\]\s*(?=\||$)")


@dataclass
class Section:
    kind: str  # "sheet" or "pages"
    name: str | None  # sheet name for sheets
    lines: list[str]

    def render(self) -> str:
        head = [f"=== Sheet: {self.name} ==="] if self.kind == "sheet" else []
        return "\n".join(head + self.lines)


@dataclass
class RowChunk:
    section: Section
    header: list[str]  # context lines, empty for the first chunk of a section
    lines: list[str]
    index: int = 0
    starts: list[int] = field(default_factory=lambda: [0])  # where each record begins
    label: str = ""
    columns: list[str] = field(default_factory=list)  # the section's column-header lines

    def __post_init__(self):
        self.label = self.label or str(self.index)

    @property
    def sheet(self) -> str | None:
        return self.section.name if self.section.kind == "sheet" else None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def records(self) -> int:
        return len(self.starts)

    def halves(self) -> tuple["RowChunk", "RowChunk"] | None:
        """Split at the record boundary nearest the middle. None if this is one record."""
        if len(self.starts) < 2:
            return None
        k = len(self.starts) // 2
        mid = self.starts[k]
        first_lines, rest = self.lines[:mid], self.lines[mid:]
        second_lines = list(rest)
        offset = -mid
        page = next((ln for ln in reversed(first_lines) if PAGE_MARKER.match(ln)), None)
        if page and not PAGE_MARKER.match(rest[0]):
            second_lines.insert(0, page)  # the second half still knows its page
            offset += 1
        tail = [ln for ln in first_lines[-CONTEXT_TAIL_LINES:]
                if ln not in self.columns and not PAGE_MARKER.match(ln)]
        first = RowChunk(self.section, self.header, first_lines, self.index,
                         self.starts[:k], f"{self.label}.1", self.columns)
        second = RowChunk(self.section, self.columns + (["..."] + tail if tail else []),
                          second_lines, self.index, [s + offset for s in self.starts[k:]],
                          f"{self.label}.2", self.columns)
        return first, second


# --------------------------------------------------------------------------- sections


def compact_line(line: str) -> str:
    line = _EMPTY_CELL.sub("", line)
    line = re.sub(r"\|(\s*\|)+", "|", line)  # pipes left behind by removed cells
    line = re.sub(r"^(Row \d+:)\s*\|", r"\1", line)
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line.rstrip("| ").strip()


def _is_blank_row(line: str) -> bool:
    return re.fullmatch(r"Row \d+:?", line) is not None


def build_sections(doc: Document) -> list[Section]:
    sections: list[Section] = []
    pages: list[str] = []
    for chunk in doc.chunks:
        loc = chunk.location.strip()
        lines = [compact_line(ln) for ln in chunk.text.splitlines()]
        lines = [ln for ln in lines if ln and not _is_blank_row(ln)]
        if m := _SHEET_LOC.match(loc):
            sections.append(Section("sheet", m.group(1), lines))
        else:
            m = _PAGE_LOC.match(loc)
            pages.append(f"[Page {m.group(1)}]" if m else loc)
            pages.extend(lines)
    if pages:
        sections.append(Section("pages", None, pages))
    return sections


def render_full(sections: list[Section], max_chars: int) -> str:
    """All sections as one text. Over max_chars, each section keeps its head and tail
    in proportion to its size, with a note where lines were left out."""
    rendered = [s.render() for s in sections]
    total = sum(len(r) for r in rendered)
    if total <= max_chars:
        return "\n\n".join(rendered)
    out = []
    for sec, text in zip(sections, rendered):
        budget = max(2000, int(max_chars * len(text) / total))
        if len(text) <= budget:
            out.append(text)
            continue
        head_n, tail_n = int(budget * 0.75), int(budget * 0.25)
        head = text[:head_n].rsplit("\n", 1)[0]
        tail = text[-tail_n:].split("\n", 1)[-1]
        omitted = text.count("\n") - head.count("\n") - tail.count("\n")
        out.append(f"{head}\n[... {omitted} lines omitted ...]\n{tail}")
    return "\n\n".join(out)


# --------------------------------------------------------------------------- cells


def parse_cells(line: str) -> dict[str, str]:
    """'Row 8: [A8] 100-101 | [B8] C5' -> {'A': '100-101', 'B': 'C5'}."""
    matches = list(_CELL.finditer(line))
    cells = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
        cells[m.group(1)] = line[m.end():end].strip().rstrip("|").strip()
    return cells


def _col_index(col: str) -> int:
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - 64)
    return n


def _is_number(text: str) -> bool:
    t = re.sub(r"[$,%()\s]", "", text)
    try:
        float(t)
        return True
    except ValueError:
        return False


def norm_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


# --------------------------------------------------------------------------- row chunks


def _group(blocks: list[list[str]], target: int,
           max_records: int | None = None) -> list[tuple[list[str], list[int]]]:
    """Pack whole blocks into groups of about `target` lines and at most `max_records`
    blocks (records), so the model's reply for a group stays within its token limit.
    Returns (lines, record start offsets) per group."""
    groups, cur, starts = [], [], []
    for block in blocks:
        for piece in (block[i:i + MAX_BLOCK_LINES] for i in range(0, len(block), MAX_BLOCK_LINES)):
            full = len(cur) + len(piece) > target or (max_records and len(starts) >= max_records)
            if cur and full:
                groups.append((cur, starts))
                cur, starts = [], []
            starts.append(len(cur))
            cur.extend(piece)
    if cur:
        groups.append((cur, starts))
    return groups


def _blocks_from_starts(lines: list[str], starts: list[int]) -> list[list[str]]:
    if not starts:
        return [[ln] for ln in lines]
    # Lines before the first record (titles, column headers, a page marker) are not a
    # record of their own: they go with the first one.
    bounds = [0] + starts[1:] + [len(lines)]
    return [lines[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]


def _sheet_blocks(sec: Section, primary_labels: list[str]) -> tuple[list[list[str]], list[str]]:
    lines = sec.lines
    cells = [parse_cells(ln) for ln in lines]
    data_start = next((i for i, c in enumerate(cells)
                       if len(c) >= 2 and any(_is_number(v) for v in c.values())), 0)
    header = lines[:data_start][-HEADER_MAX_LINES:]

    wanted = {norm_label(x) for x in primary_labels if norm_label(x)}
    key_col = None
    for c in cells[:data_start]:
        key_col = next((col for col, v in c.items() if norm_label(v) in wanted), None)
        if key_col:
            break
    if key_col is None:
        cols = [col for c in cells[data_start:] for col in c]
        key_col = min(cols, key=_col_index) if cols else None

    starts, prev = [], None
    for i in range(data_start, len(lines)):
        v = cells[i].get(key_col) if key_col else None
        if v and v != prev:
            starts.append(i)
            prev = v
    return _blocks_from_starts(lines, starts), header


def _shape(token: str) -> str:
    return re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", token))


def _page_blocks(sec: Section) -> tuple[list[list[str]], list[str]]:
    lines = sec.lines
    firsts = [ln.split(" ", 1)[0] if not PAGE_MARKER.match(ln) else "" for ln in lines]
    shapes = Counter(_shape(t) for t in firsts if re.search(r"\d", t) and len(t) >= 3)
    shape, count = shapes.most_common(1)[0] if shapes else ("", 0)
    if count >= max(3, len(lines) // 50):
        starts = [i for i, t in enumerate(firsts) if t and _shape(t) == shape]
    else:  # no record pattern: each page is a unit
        starts = [i for i, ln in enumerate(lines) if PAGE_MARKER.match(ln)]
    first = starts[0] if starts else 0
    header = [ln for ln in lines[:first] if not PAGE_MARKER.match(ln)][-HEADER_MAX_LINES:]
    return _blocks_from_starts(lines, starts), header


def split_rows(sec: Section, primary_labels: list[str], target: int | None = None,
               max_records: int | None = None) -> list[RowChunk]:
    target = target or TARGET_LINES
    if not sec.lines:
        return []
    if sec.kind == "sheet":
        blocks, header = _sheet_blocks(sec, primary_labels)
    else:
        blocks, header = _page_blocks(sec)

    chunks, page, prev = [], None, []
    for i, (group, starts) in enumerate(_group(blocks, target, max_records)):
        body = list(group)
        if sec.kind == "pages" and page and not PAGE_MARKER.match(body[0]):
            body.insert(0, page)  # so the chunk knows which page it starts on
            starts = [s + 1 for s in starts]
        for ln in group:
            if PAGE_MARKER.match(ln):
                page = ln
        context: list[str] = []
        if i > 0:
            # Column headers, plus the lines just before this chunk so that values like
            # "the nearest section header above the line" can still be read.
            tail = [ln for ln in prev[-CONTEXT_TAIL_LINES:] if ln not in header]
            context = header + (["..."] + tail if tail else [])
        chunks.append(RowChunk(sec, context, body, index=len(chunks), starts=starts,
                               columns=header))
        prev = group
    return chunks
