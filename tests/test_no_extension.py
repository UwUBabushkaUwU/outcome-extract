"""A file uploaded with no extension (or a misleading one) is typed from its bytes and parsed."""

import io
import json

import docx
import openpyxl
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import config, db, parsers, worker
from app.main import app

SCHEMA = json.dumps({"name": "t", "fields": {"a": {"name": "A", "description": "a"}}})
MARKER = "Cedar Ridge Apartments"


def _pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), f"{MARKER} rent roll, a page with enough text on it.")
    return doc.tobytes()


def _docx() -> bytes:
    d = docx.Document()
    d.add_paragraph(MARKER)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _xlsx() -> bytes:
    wb = openpyxl.Workbook()
    wb.active.append(["Property", MARKER])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _csv() -> bytes:
    return f"Property,Units\n{MARKER},24\n".encode()


SAMPLES = {"pdf": _pdf, "docx": _docx, "xlsx": _xlsx, "csv": _csv}


@pytest.mark.parametrize("file_type", SAMPLES)
@pytest.mark.parametrize("upload_name", ["upload", "document.bin"])
def test_upload_without_extension_is_parsed_end_to_end(file_type, upload_name, monkeypatch,
                                                        fake_llm):
    parsed = {}
    real_parse = parsers.parse

    def spy(path, kind=None):
        parsed["doc"] = real_parse(path, kind)
        parsed["kind"] = kind
        return parsed["doc"]

    monkeypatch.setattr(parsers, "parse", spy)

    with TestClient(app) as client:
        r = client.post("/jobs", files={"file": (upload_name, SAMPLES[file_type]())},
                        data={"schema": SCHEMA})
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]

        job = db.claim_job("test-worker")
        assert job["file_type"] == file_type
        worker.process_job(job, "test-worker")
        done = client.get(f"/jobs/{job_id}").json()

    assert done["status"] == "SUCCESS", done["error"]
    assert parsed["kind"] == file_type
    assert MARKER in "\n".join(c.text for c in parsed["doc"].chunks)


@pytest.mark.parametrize("file_type", SAMPLES)
def test_parse_uses_explicit_type_not_the_filename(file_type, tmp_path):
    path = tmp_path / "no_extension"
    path.write_bytes(SAMPLES[file_type]())
    with pytest.raises(ValueError):
        parsers.parse(path)  # nothing to go on without an extension...
    doc = parsers.parse(path, file_type)  # ...but the detected type is enough
    assert MARKER in "\n".join(c.text for c in doc.chunks)


def test_stored_upload_keeps_detected_extension(tmp_path):
    with TestClient(app) as client:
        r = client.post("/jobs", files={"file": ("upload", _xlsx())}, data={"schema": SCHEMA})
    job = db.get_job(r.json()["job_id"], "")
    assert (config.DATA_DIR / job["file_path"]).suffix == ".xlsx"
