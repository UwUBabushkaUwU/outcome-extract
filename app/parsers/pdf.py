"""PDF parser using pdfplumber for text extraction and PyMuPDF for page rendering."""

from pathlib import Path
import pdfplumber
import pymupdf

from app.parsers.base import Chunk, Document


def parse(path: str | Path) -> Document:
    """Parse a PDF document into Chunks per page and render pages with <50 chars as PNG images.

    Args:
        path: Path to the PDF file.

    Returns:
        Document containing chunks tagged [Page N] and rendered images for pages with <50 chars.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    chunks: list[Chunk] = []
    images: list[tuple[int, bytes]] = []

    with pdfplumber.open(file_path) as pdf, pymupdf.open(file_path) as mupdf_doc:
        for page_num, plumber_page in enumerate(pdf.pages, start=1):
            text = plumber_page.extract_text() or ""
            chunks.append(Chunk(text=text, location=f"[Page {page_num}]"))

            if len(text.strip()) < 50:
                mupdf_page = mupdf_doc[page_num - 1]
                pix = mupdf_page.get_pixmap(dpi=150)
                images.append((page_num, pix.tobytes("png")))

    return Document(chunks=chunks, images=images)
