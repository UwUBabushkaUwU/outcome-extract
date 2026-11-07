"""XLSX parser processing every sheet with A1 references, row-numbered text, and merged cells filled in."""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import openpyxl
from openpyxl.utils import get_column_letter

from app.parsers.base import Chunk, Document


def _format_cell_value(val: Any) -> str:
    """Format a cell value into a clean string representation."""
    if val is None:
        return ""
    if isinstance(val, (datetime, date)):
        if isinstance(val, datetime) and (val.hour != 0 or val.minute != 0 or val.second != 0):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val.strftime("%Y-%m-%d")
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return str(val)
    if isinstance(val, bool):
        return str(val)
    return str(val).strip().replace("\r\n", " ").replace("\n", " ")


def parse(path: str | Path) -> Document:
    """Parse an XLSX document into Chunks for every sheet.

    Cells are rendered with A1 references, row-numbered text, and merged cells are filled in.

    Args:
        path: Path to the XLSX file.

    Returns:
        Document containing a Chunk for every sheet in the workbook.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Pass a file object: given a path, openpyxl rejects anything not named .xlsx/.xlsm.
    with file_path.open("rb") as fh:
        wb = openpyxl.load_workbook(fh, data_only=True)
    chunks: list[Chunk] = []

    for ws in wb.worksheets:
        grid: dict[tuple[int, int], Any] = {}

        # 1. Populate raw cell values efficiently with iter_rows
        for r, row in enumerate(ws.iter_rows(), start=1):
            for c, cell in enumerate(row, start=1):
                if cell.value is not None:
                    grid[(r, c)] = cell.value

        # 2. Fill in merged cells with top-left value
        merged_ranges = list(ws.merged_cells.ranges)
        for rng in merged_ranges:
            top_left_val = ws.cell(rng.min_row, rng.min_col).value
            if top_left_val is not None:
                for r in range(rng.min_row, rng.max_row + 1):
                    for c in range(rng.min_col, rng.max_col + 1):
                        grid[(r, c)] = top_left_val

        # 3. Determine active range bounds
        active_cells = {
            (r, c): _format_cell_value(v)
            for (r, c), v in grid.items()
            if _format_cell_value(v) != ""
        }

        if not active_cells:
            chunks.append(Chunk(text="", location=f"[{ws.title}!A1:A1]"))
            continue

        min_r = min(r for r, c in active_cells.keys())
        max_r = max(r for r, c in active_cells.keys())
        min_c = min(c for r, c in active_cells.keys())
        max_c = max(c for r, c in active_cells.keys())

        min_col_letter = get_column_letter(min_c)
        max_col_letter = get_column_letter(max_c)
        location = f"[{ws.title}!{min_col_letter}{min_r}:{max_col_letter}{max_r}]"

        # 4. Render row-numbered text with A1 cell references
        row_lines: list[str] = []
        for r in range(min_r, max_r + 1):
            row_cells: list[str] = []
            for c in range(min_c, max_c + 1):
                if (r, c) in active_cells:
                    val_str = active_cells[(r, c)]
                    col_letter = get_column_letter(c)
                    row_cells.append(f"[{col_letter}{r}] {val_str}")
            if row_cells:
                row_lines.append(f"Row {r}: " + " | ".join(row_cells))

        sheet_text = "\n".join(row_lines)
        chunks.append(Chunk(text=sheet_text, location=location))

    wb.close()
    return Document(chunks=chunks, images=[])
