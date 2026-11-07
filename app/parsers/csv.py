"""CSV parser rendering cells with A1 references and row-numbered text."""

import csv as std_csv
from pathlib import Path

from openpyxl.utils import get_column_letter

from app.parsers.base import Chunk, Document


def parse(path: str | Path) -> Document:
    """Parse a CSV document into a Document Chunk.

    Cells are rendered with A1 references and row-numbered text under Sheet1.

    Args:
        path: Path to the CSV file.

    Returns:
        Document containing a Chunk for the CSV sheet.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read content with encoding fallback
    try:
        content = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="latin-1")

    if not content.strip():
        return Document(chunks=[Chunk(text="", location="[Sheet1!A1:A1]")], images=[])

    # Sniff dialect / delimiter
    sample = content[:4096]
    try:
        dialect = std_csv.Sniffer().sniff(sample, delimiters=",\t;|")
    except Exception:
        dialect = std_csv.excel

    import io
    reader = std_csv.reader(io.StringIO(content, newline=""), dialect=dialect)
    grid: dict[tuple[int, int], str] = {}

    for r, row in enumerate(reader, start=1):
        for c, val in enumerate(row, start=1):
            val_str = str(val).strip().replace("\r\n", " ").replace("\n", " ")
            if val_str:
                grid[(r, c)] = val_str

    if not grid:
        return Document(chunks=[Chunk(text="", location="[Sheet1!A1:A1]")], images=[])

    min_r = min(r for r, c in grid.keys())
    max_r = max(r for r, c in grid.keys())
    min_c = min(c for r, c in grid.keys())
    max_c = max(c for r, c in grid.keys())

    min_col_letter = get_column_letter(min_c)
    max_col_letter = get_column_letter(max_c)
    location = f"[Sheet1!{min_col_letter}{min_r}:{max_col_letter}{max_r}]"

    row_lines: list[str] = []
    for r in range(min_r, max_r + 1):
        row_cells: list[str] = []
        for c in range(min_c, max_c + 1):
            if (r, c) in grid:
                col_letter = get_column_letter(c)
                row_cells.append(f"[{col_letter}{r}] {grid[(r, c)]}")
        if row_cells:
            row_lines.append(f"Row {r}: " + " | ".join(row_cells))

    sheet_text = "\n".join(row_lines)
    return Document(chunks=[Chunk(text=sheet_text, location=location)], images=[])
