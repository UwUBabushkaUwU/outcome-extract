"""SQLite job table (WAL mode), shared by the API process and the worker processes.

Lifecycle: PENDING -> RUNNING -> SUCCESS | PARTIAL | FAILED.

* A worker claims the oldest PENDING job under BEGIN IMMEDIATE, so two workers never get the
  same job. The claim sets a lease (lease_owner, lease_expires) that the worker renews.
* A RUNNING job whose lease has expired (its worker crashed or hung) goes back to PENDING.
  After MAX_ATTEMPTS claims it is FAILED instead, so a document that kills workers can't loop.
* Terminal states are written only with WHERE status='RUNNING' AND lease_owner=<me>. Once a
  job is terminal, nothing can change it, and a worker that lost its lease can't overwrite
  the result of the worker that took the job over.
"""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from app import config

TERMINAL = ("SUCCESS", "PARTIAL", "FAILED")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    org_id        TEXT NOT NULL,
    idem_key      TEXT NOT NULL UNIQUE,
    status        TEXT NOT NULL,
    stage         TEXT,
    filename      TEXT,
    file_path     TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    schema_json   TEXT NOT NULL,
    webhook_url   TEXT,
    result_json   TEXT,
    error_json    TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    lease_owner   TEXT,
    lease_expires REAL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs (status, created_at);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def connect() -> sqlite3.Connection:
    # Autocommit mode; transactions are opened explicitly where they are needed.
    conn = sqlite3.connect(config.DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def _write_tx(conn: sqlite3.Connection) -> Iterator[None]:
    """Take the write lock up front, so read-then-write sequences can't race."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def init_db() -> None:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)


def _row_to_job(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    job = dict(row)
    job["schema"] = json.loads(job.pop("schema_json"))
    job["result"] = json.loads(r) if (r := job.pop("result_json")) else None
    job["error"] = json.loads(e) if (e := job.pop("error_json")) else None
    return job


def idempotency_key(file_sha256: str, schema_sha256: str, org_id: str) -> str:
    return f"{file_sha256}:{schema_sha256}:{org_id}"


def create_job(
    *,
    org_id: str,
    idem_key: str,
    filename: str | None,
    file_path: str,
    file_type: str,
    schema: dict,
    webhook_url: str | None,
) -> tuple[dict[str, Any], bool]:
    """Insert a PENDING job. Returns (job, created); created is False if an identical
    submission (same idempotency key) already exists, in which case that job is returned."""
    ts = now_iso()
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO jobs (id, org_id, idem_key, status, filename, file_path, file_type,
                                 schema_json, webhook_url, created_at, updated_at)
               VALUES (?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (idem_key) DO NOTHING""",
            (uuid.uuid4().hex, org_id, idem_key, filename, file_path, file_type,
             json.dumps(schema), webhook_url, ts, ts),
        )
        created = cur.rowcount == 1
        row = conn.execute("SELECT * FROM jobs WHERE idem_key = ?", (idem_key,)).fetchone()
    return _row_to_job(row), created


def get_job(job_id: str, org_id: str) -> dict[str, Any] | None:
    """Org-scoped read: a job from another organization looks exactly like a missing one."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ? AND org_id = ?", (job_id, org_id)
        ).fetchone()
    return _row_to_job(row)


def _reclaim_expired(conn: sqlite3.Connection, now: float) -> None:
    ts = now_iso()
    error = json.dumps({
        "code": "WORKER_LOST",
        "message": f"worker lease expired {config.MAX_ATTEMPTS} times; giving up",
    })
    conn.execute(
        """UPDATE jobs SET status = 'FAILED', stage = NULL, error_json = ?, lease_owner = NULL,
                          lease_expires = NULL, updated_at = ?
           WHERE status = 'RUNNING' AND lease_expires < ? AND attempts >= ?""",
        (error, ts, now, config.MAX_ATTEMPTS),
    )
    conn.execute(
        """UPDATE jobs SET status = 'PENDING', stage = NULL, lease_owner = NULL,
                          lease_expires = NULL, updated_at = ?
           WHERE status = 'RUNNING' AND lease_expires < ?""",
        (ts, now),
    )


def claim_job(worker_id: str, lease_seconds: float | None = None) -> dict[str, Any] | None:
    """Requeue expired leases, then atomically move the oldest PENDING job to RUNNING."""
    lease = config.LEASE_SECONDS if lease_seconds is None else lease_seconds
    now = time.time()
    with _conn() as conn:
        with _write_tx(conn):
            _reclaim_expired(conn, now)
            row = conn.execute(
                """UPDATE jobs SET status = 'RUNNING', stage = 'QUEUED', lease_owner = ?,
                                  lease_expires = ?, attempts = attempts + 1, updated_at = ?
                   WHERE id = (SELECT id FROM jobs WHERE status = 'PENDING'
                               ORDER BY created_at LIMIT 1)
                   RETURNING *""",
                (worker_id, now + lease, now_iso()),
            ).fetchone()
    return _row_to_job(row)


def renew_lease(job_id: str, worker_id: str, lease_seconds: float | None = None) -> bool:
    """Extend the lease. False means this worker no longer owns the job."""
    lease = config.LEASE_SECONDS if lease_seconds is None else lease_seconds
    with _conn() as conn:
        cur = conn.execute(
            """UPDATE jobs SET lease_expires = ?
               WHERE id = ? AND status = 'RUNNING' AND lease_owner = ?""",
            (time.time() + lease, job_id, worker_id),
        )
    return cur.rowcount == 1


def set_stage(job_id: str, worker_id: str, stage: str) -> bool:
    with _conn() as conn:
        cur = conn.execute(
            """UPDATE jobs SET stage = ?, updated_at = ?
               WHERE id = ? AND status = 'RUNNING' AND lease_owner = ?""",
            (stage, now_iso(), job_id, worker_id),
        )
    return cur.rowcount == 1


def finish_job(
    job_id: str,
    worker_id: str,
    status: str,
    result: dict | None = None,
    error: dict | None = None,
) -> bool:
    """Move a RUNNING job to a terminal state. Returns False (and changes nothing) if the job
    is already terminal or the lease now belongs to someone else."""
    if status not in TERMINAL:
        raise ValueError(f"not a terminal status: {status}")
    with _conn() as conn:
        cur = conn.execute(
            """UPDATE jobs SET status = ?, stage = NULL, result_json = ?, error_json = ?,
                              lease_owner = NULL, lease_expires = NULL, updated_at = ?
               WHERE id = ? AND status = 'RUNNING' AND lease_owner = ?""",
            (status, json.dumps(result) if result is not None else None,
             json.dumps(error) if error is not None else None, now_iso(), job_id, worker_id),
        )
    return cur.rowcount == 1


def ping() -> None:
    with _conn() as conn:
        conn.execute("SELECT 1").fetchone()
