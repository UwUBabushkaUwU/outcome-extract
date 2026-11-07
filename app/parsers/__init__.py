"""Document parsers for PDF, DOCX, XLSX, and CSV files."""

from pathlib import Path

from app.parsers.base import Chunk, Document
from app.parsers.csv import parse as parse_csv
from app.parsers.docx import parse as parse_docx
from app.parsers.pdf import parse as parse_pdf
from app.parsers.xlsx import parse as parse_xlsx


_PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "xlsx": parse_xlsx,
    "csv": parse_csv,
}


def parse(path: str | Path, file_type: str | None = None) -> Document:
    """Parse a PDF, DOCX, XLSX or CSV document.

    Args:
        path: Path to the document file.
        file_type: "pdf", "docx", "xlsx" or "csv". The service passes the type it detected
            from the file's bytes at upload, so the filename doesn't matter. If omitted
            (e.g. the CLI), the type comes from the file extension.

    Returns:
        Document instance containing chunks and optional images.

    Raises:
        ValueError: If file format is not supported.
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path)
    kind = (file_type or file_path.suffix.lstrip(".")).lower()
    parser = _PARSERS.get(kind)
    if parser is None:
        raise ValueError(
            f"Unsupported file format '{kind}' for {file_path}. Expected pdf, docx, xlsx, or csv")
    return parser(file_path)


__all__ = [
    "Chunk",
    "Document",
    "parse",
    "parse_pdf",
    "parse_docx",
    "parse_xlsx",
    "parse_csv",
]
