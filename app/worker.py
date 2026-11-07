"""Worker processes: poll the job table, run the pipeline, write a terminal state.

Each worker holds a lease on its job and renews it from a heartbeat thread. If the process
dies, the lease expires and another worker picks the job up (see app/db.py).
"""

import logging
import multiprocessing as mp
import os
import random
import threading
import time
import traceback
from datetime import datetime

import httpx

from app import config, db, pipeline

log = logging.getLogger("app.worker")


class _Heartbeat(threading.Thread):
    def __init__(self, job_id: str, worker_id: str):
        super().__init__(daemon=True, name=f"heartbeat-{job_id[:8]}")
        self.job_id, self.worker_id = job_id, worker_id
        self.stop = threading.Event()
        self.lost = False

    def run(self) -> None:
        while not self.stop.wait(config.LEASE_SECONDS / 3):
            try:
                if not db.renew_lease(self.job_id, self.worker_id):
                    self.lost = True
                    log.error("job=%s lease lost by %s", self.job_id, self.worker_id)
                    return
            except Exception:  # noqa: BLE001 - a failed renewal is retried next tick
                log.exception("job=%s lease renewal failed", self.job_id)


def _latency_ms(created_at: str) -> int:
    created = datetime.fromisoformat(created_at)
    return max(0, int((time.time() - created.timestamp()) * 1000))


def _send_webhook(url: str, body: dict, job_id: str) -> None:
    def post() -> None:
        for attempt in range(3):
            try:
                r = httpx.post(url, json=body, timeout=10)
                if r.status_code < 500:
                    log.info("job=%s webhook %s -> %d", job_id, url, r.status_code)
                    return
            except httpx.HTTPError as e:
                log.warning("job=%s webhook attempt %d failed: %s", job_id, attempt + 1, e)
            time.sleep(2 ** attempt)
        log.error("job=%s webhook %s gave up", job_id, url)

    threading.Thread(target=post, daemon=True, name=f"webhook-{job_id[:8]}").start()


def process_job(job: dict, worker_id: str) -> None:
    job_id = job["id"]
    log.info("job=%s claimed by %s (attempt %d) file=%r type=%s",
             job_id, worker_id, job["attempts"], job["filename"], job["file_type"])
    hb = _Heartbeat(job_id, worker_id)
    hb.start()
    ctx = pipeline.JobContext(
        job_id=job_id,
        file_path=config.DATA_DIR / job["file_path"],
        file_type=job["file_type"],
        filename=job["filename"],
        schema=job["schema"],
        set_stage=lambda stage: db.set_stage(job_id, worker_id, stage),
    )
    result = error = None
    try:
        out = pipeline.run(ctx)
        status, error = out.status, out.error
        result = {
            "entries": out.entries,
            "metadata": {**ctx.usage.to_metadata(), "latency_ms": _latency_ms(job["created_at"])},
        }
    except Exception as exc:  # noqa: BLE001 - any pipeline failure becomes FAILED with a reason
        status = "FAILED"
        error = {"code": type(exc).__name__, "message": str(exc)[:1000] or repr(exc)}
        log.error("job=%s FAILED: %s\n%s", job_id, error["message"], traceback.format_exc())
    finally:
        hb.stop.set()

    if not db.finish_job(job_id, worker_id, status, result, error):
        log.error("job=%s result discarded: lease no longer held by %s", job_id, worker_id)
        return
    meta = (result or {}).get("metadata", {})
    log.info("job=%s %s in %sms, %s calls, $%s", job_id, status, meta.get("latency_ms"),
             meta.get("model_calls"), meta.get("cost_usd"))
    if job["webhook_url"]:
        final = db.get_job(job_id, job["org_id"])
        _send_webhook(job["webhook_url"], {
            "job_id": job_id, "status": status, "created_at": final["created_at"],
            "updated_at": final["updated_at"], "result": result, "error": error,
        }, job_id)


def _parent_alive() -> bool:
    parent = mp.parent_process()
    return parent is None or parent.is_alive()


def worker_loop(index: int, stop: "mp.synchronize.Event") -> None:
    """Entry point of one worker process."""
    from app.logging_setup import setup_logging
    import signal

    setup_logging()
    # Ctrl+C goes to the whole console on Windows; let the parent coordinate shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker_id = f"w{index}-{os.getpid()}"
    log.info("worker %s started", worker_id)
    while not stop.is_set() and _parent_alive():
        try:
            job = db.claim_job(worker_id)
        except Exception:  # noqa: BLE001 - e.g. a locked DB; back off and poll again
            log.exception("claim failed")
            job = None
        if job is None:
            stop.wait(config.POLL_INTERVAL * random.uniform(0.5, 1.5))
            continue
        process_job(job, worker_id)
    log.info("worker %s stopped", worker_id)


class Supervisor(threading.Thread):
    """Starts N worker processes and restarts any that die."""

    def __init__(self, n: int):
        super().__init__(daemon=True, name="supervisor")
        self.ctx = mp.get_context("spawn")  # same behaviour on Windows and Linux
        self.stop_event = self.ctx.Event()
        self.n = n
        self.procs: list = [None] * n
        self._halt = threading.Event()

    def _spawn(self, i: int) -> None:
        p = self.ctx.Process(target=worker_loop, args=(i, self.stop_event),
                             name=f"worker-{i}", daemon=False)
        p.start()
        self.procs[i] = p

    def run(self) -> None:
        for i in range(self.n):
            self._spawn(i)
        while not self._halt.wait(2):
            for i, p in enumerate(self.procs):
                if p is not None and not p.is_alive():
                    log.error("worker-%d exited with code %s; restarting", i, p.exitcode)
                    self._spawn(i)

    def shutdown(self, timeout: float = 10) -> None:
        self._halt.set()
        self.stop_event.set()
        deadline = time.time() + timeout
        for p in self.procs:
            if p is not None:
                p.join(max(0.1, deadline - time.time()))
        for p in self.procs:
            if p is not None and p.is_alive():
                log.warning("terminating %s (its job will be re-queued when the lease expires)",
                            p.name)
                p.terminate()
