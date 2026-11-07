"""DOCX parser extracting paragraphs and tables in reading order."""

from pathlib import Path
import docx
from docx.text.paragraph import Paragraph
from docx.table import Table

from app.parsers.base import Chunk, Document


def _format_table(table: Table) -> str:
    """Format a docx Table into pipe-separated markdown-like rows."""
    lines: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\r\n", " ").replace("\n", " ") for cell in row.cells]
        if any(cells):
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _has_page_break(element) -> bool:
    """Check if element contains a page break or section break."""
    for n in element.iter():
        if n.tag.endswith("lastRenderedPageBreak"):
            return True
        if n.tag.endswith("sectPr"):
            return True
        if n.tag.endswith("br") and n.attrib.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type") == "page":
            return True
    return False


def _iter_body_elements(container, doc):
    """Yield tuples of (type, content, has_break) in reading order."""
    for child in container:
        tag = child.tag
        if tag.endswith("p"):
            p = Paragraph(child, doc)
            text = p.text.strip()
            yield ("p", text, _has_page_break(child))
        elif tag.endswith("tbl"):
            t = Table(child, doc)
            tbl_text = _format_table(t)
            yield ("tbl", tbl_text, _has_page_break(child))
        elif tag.endswith("sdt"):
            sdt_content = child.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sdtContent")
            if sdt_content is not None:
                yield from _iter_body_elements(sdt_content, doc)
        elif tag.endswith("sectPr"):
            yield ("sectPr", "", True)


def parse(path: str | Path) -> Document:
    """Parse a DOCX document into Chunks preserving reading order of paragraphs and tables.

    Args:
        path: Path to the DOCX file.

    Returns:
        Document containing chunks in reading order.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc = docx.Document(str(file_path))
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    page_num = 1

    for elem_type, text, is_break in _iter_body_elements(doc.element.body, doc):
        if text:
            current_texts.append(text)

        if is_break:
            if current_texts:
                chunks.append(Chunk(text="\n\n".join(current_texts), location=f"[Page {page_num}]"))
                current_texts = []
            page_num += 1

    if current_texts:
        chunks.append(Chunk(text="\n\n".join(current_texts), location=f"[Page {page_num}]"))

    # If document was completely empty
    if not chunks:
        chunks.append(Chunk(text="", location="[Page 1]"))

    return Document(chunks=chunks, images=[])
