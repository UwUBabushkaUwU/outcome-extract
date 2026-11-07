"""HTTP API. Matches ref/contract/openapi.yaml: POST /jobs, GET /jobs/{job_id}, GET /health."""

import hashlib
import json
import logging
import os
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import config, db

log = logging.getLogger("app.api")

SUPPORTED_TYPES = ("pdf", "docx", "xlsx", "csv")
_CHUNK = 1024 * 1024


@asynccontextmanager
async def _lifespan(_: FastAPI):
    config.FILES_DIR.mkdir(parents=True, exist_ok=True)
    config.TMP_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    yield


app = FastAPI(title="Document Extraction Service", version="1.0", lifespan=_lifespan)


def _error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"code": code, "message": message}, status_code=status, headers=headers)


# The contract says malformed requests are 400 with an Error body. FastAPI would send 422.
@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return _error(400, "BAD_REQUEST", "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()))


@app.exception_handler(StarletteHTTPException)
async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error(exc.status_code, "HTTP_ERROR", str(exc.detail))


def _job_status(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "result": job["result"] if job["status"] in ("SUCCESS", "PARTIAL") else None,
        "error": job["error"],
    }


def _org(x_organization_id: str | None) -> str:
    return (x_organization_id or "").strip()


def detect_file_type(filename: str | None, path: Path) -> str | None:
    """Decide the type from the bytes first and the extension second.
    Returns None for anything that isn't PDF, DOCX, XLSX or CSV."""
    with path.open("rb") as f:
        head = f.read(4096)
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except zipfile.BadZipFile:
            return None
        if any(n.startswith("word/") for n in names):
            return "docx"
        if any(n.startswith("xl/") for n in names):
            return "xlsx"
        return None
    if not head or b"\x00" in head:
        return None
    # Text with no other signature: treat it as CSV whatever the extension says.
    for enc in ("utf-8-sig", "cp1252"):
        try:
            head.decode(enc)
            return "csv"
        except UnicodeDecodeError:
            continue
    return None


def _parse_schema(raw: str) -> dict:
    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"schema is not valid JSON: {e}") from e
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON object")
    fields = schema.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("schema must have a non-empty 'fields' object")
    if not all(isinstance(v, dict) for v in fields.values()):
        raise ValueError("each entry in schema.fields must be an object")
    return schema


def _canonical_sha256(obj: Any) -> str:
    """Hash of the parsed schema, so whitespace and key order don't defeat idempotency."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@app.get("/health")
def health() -> dict[str, str]:
    db.ping()
    return {"status": "ok", "version": config.VERSION}


@app.post("/jobs", status_code=202)
def create_job(
    file: UploadFile | None = File(None),
    schema_raw: str | None = Form(None, alias="schema"),  # "schema" shadows a pydantic attr
    webhook_url: str | None = Form(None),
    x_organization_id: str | None = Header(None),
):
    schema = schema_raw
    if file is None:
        return _error(400, "MISSING_FILE", "multipart field 'file' is required")
    if schema is None or not schema.strip():
        return _error(400, "MISSING_SCHEMA", "multipart field 'schema' is required")
    try:
        schema_obj = _parse_schema(schema)
    except ValueError as e:
        return _error(400, "INVALID_SCHEMA", str(e))
    if webhook_url:
        parsed = urlparse(webhook_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return _error(400, "INVALID_WEBHOOK_URL", "webhook_url must be an http(s) URL")

    # Stream the upload to a temp file, hashing as we go, so the 50 MB limit is enforced
    # without holding the file in memory.
    tmp = config.TMP_DIR / f"{uuid.uuid4().hex}.part"
    sha = hashlib.sha256()
    size = 0
    try:
        with tmp.open("wb") as out:
            while chunk := file.file.read(_CHUNK):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    out.close()
                    tmp.unlink(missing_ok=True)
                    return _error(413, "FILE_TOO_LARGE",
                                  f"file exceeds {config.MAX_UPLOAD_BYTES} bytes")
                sha.update(chunk)
                out.write(chunk)
        if size == 0:
            tmp.unlink(missing_ok=True)
            return _error(400, "EMPTY_FILE", "uploaded file is empty")

        file_type = detect_file_type(file.filename, tmp)
        if file_type is None:
            tmp.unlink(missing_ok=True)
            return _error(400, "UNSUPPORTED_FILE_TYPE",
                          f"supported types: {', '.join(SUPPORTED_TYPES)}")

        # Files are content-addressed, so a duplicate upload reuses the stored copy.
        file_sha = sha.hexdigest()
        final = config.FILES_DIR / f"{file_sha}.{file_type}"
        if final.exists():
            tmp.unlink(missing_ok=True)
        else:
            os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    org = _org(x_organization_id)
    job, created = db.create_job(
        org_id=org,
        idem_key=db.idempotency_key(file_sha, _canonical_sha256(schema_obj), org),
        filename=file.filename,
        file_path=str(final.relative_to(config.DATA_DIR)),
        file_type=file_type,
        schema=schema_obj,
        webhook_url=webhook_url or None,
    )
    log.info("job=%s %s org=%r file=%r type=%s size=%d",
             job["id"], "created" if created else "duplicate", org, file.filename, file_type, size)
    return JSONResponse(_job_status(job), status_code=202)


@app.get("/jobs/{job_id}")
def get_job(job_id: str, x_organization_id: str | None = Header(None)):
    job = db.get_job(job_id, _org(x_organization_id))
    if job is None:
        return _error(404, "NOT_FOUND", f"job {job_id} not found")
    return _job_status(job)
