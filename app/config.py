"""Runtime settings, read from the environment once per process."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("DB_PATH", DATA_DIR / "jobs.db"))
FILES_DIR = DATA_DIR / "files"
TMP_DIR = DATA_DIR / "tmp"
CACHE_DIR = Path(os.environ.get("CACHE_DIR", ROOT / "cache"))

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = _int("PORT", 8080)
WORKERS = _int("WORKERS", 4)
VERSION = os.environ.get("GIT_SHA", "dev")

MAX_UPLOAD_BYTES = _int("MAX_UPLOAD_BYTES", 50 * 1024 * 1024)

# A worker holds a job for LEASE_SECONDS and renews the lease every LEASE_SECONDS / 3.
# If the worker dies, the lease runs out and another worker picks the job up again.
LEASE_SECONDS = _float("LEASE_SECONDS", 60)
MAX_ATTEMPTS = _int("MAX_ATTEMPTS", 3)  # claims per job before it is FAILED as crash-looping
POLL_INTERVAL = _float("POLL_INTERVAL", 0.5)
