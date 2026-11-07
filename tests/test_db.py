import threading

from app import config, db


def _create(key="f:s:", org=""):
    return db.create_job(org_id=org, idem_key=key, filename="a.csv", file_path="files/a.csv",
                         file_type="csv", schema={"fields": {"x": {}}}, webhook_url=None)


def test_duplicate_submission_returns_same_job():
    a, created_a = _create()
    b, created_b = _create()
    assert created_a and not created_b
    assert a["id"] == b["id"]


def test_same_file_other_org_is_a_different_job():
    a, _ = _create(db.idempotency_key("f", "s", "org1"), "org1")
    b, created = _create(db.idempotency_key("f", "s", "org2"), "org2")
    assert created and a["id"] != b["id"]


def test_get_job_is_org_scoped():
    job, _ = _create(db.idempotency_key("f", "s", "org1"), "org1")
    assert db.get_job(job["id"], "org1") is not None
    assert db.get_job(job["id"], "org2") is None
    assert db.get_job(job["id"], "") is None


def test_claim_is_exclusive_under_concurrency():
    for i in range(20):
        _create(f"k{i}")
    claimed, lock = [], threading.Lock()

    def worker(n):
        while (job := db.claim_job(f"w{n}")) is not None:
            with lock:
                claimed.append(job["id"])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == 20 and len(set(claimed)) == 20


def test_expired_lease_goes_back_to_pending_and_old_owner_cannot_finish():
    job, _ = _create()
    first = db.claim_job("dead-worker", lease_seconds=-1)  # already expired
    assert first["status"] == "RUNNING"
    second = db.claim_job("live-worker")
    assert second["id"] == job["id"] and second["attempts"] == 2
    assert not db.finish_job(job["id"], "dead-worker", "SUCCESS", {"entries": []})
    assert not db.renew_lease(job["id"], "dead-worker")
    assert db.finish_job(job["id"], "live-worker", "SUCCESS", {"entries": []})


def test_terminal_state_never_changes():
    job, _ = _create()
    db.claim_job("w")
    assert db.finish_job(job["id"], "w", "SUCCESS", {"entries": []})
    assert not db.finish_job(job["id"], "w", "FAILED", error={"code": "X", "message": "y"})
    assert not db.set_stage(job["id"], "w", "EXTRACTING")
    assert db.get_job(job["id"], "")["status"] == "SUCCESS"
    assert db.claim_job("w2") is None


def test_crash_looping_job_fails_after_max_attempts(monkeypatch):
    monkeypatch.setattr(config, "MAX_ATTEMPTS", 2)
    job, _ = _create()
    db.claim_job("w1", lease_seconds=-1)
    db.claim_job("w2", lease_seconds=-1)
    assert db.claim_job("w3") is None  # reclaim marks it FAILED instead of re-queueing
    final = db.get_job(job["id"], "")
    assert final["status"] == "FAILED" and final["error"]["code"] == "WORKER_LOST"
