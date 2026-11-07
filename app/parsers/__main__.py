"""CLI for document parsers: python -m app.parsers <file>."""

import argparse
import sys
from pathlib import Path

from app.parsers import parse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a document (PDF, DOCX, XLSX, CSV) and print its chunks and rendered images."
    )
    parser.add_argument("file", help="Path to the document to parse")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.is_file():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        doc = parse(file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"--- Document: {file_path.name} ---")
    print(f"Total Chunks: {len(doc.chunks)}")
    print(f"Total Images: {len(doc.images)}\n")

    for i, chunk in enumerate(doc.chunks, start=1):
        print(f"=== Chunk {i}: {chunk.location} ===")
        if chunk.text:
            print(chunk.text)
        else:
            print("(empty text)")
        print()

    if doc.images:
        print(f"=== Rendered Images ({len(doc.images)}) ===")
        for page_no, img_bytes in doc.images:
            print(f"Page {page_no}: PNG ({len(img_bytes)} bytes)")
        print()


if __name__ == "__main__":
    main()
