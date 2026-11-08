"""Spend guards at pipeline and worker level, with the real llm.chat_json on a mocked client."""

import json
import re
from types import SimpleNamespace

import pytest

from app import config, db, llm, pipeline, worker
from app.pipeline import JobContext
from conftest import rows_section

SCHEMA = {
    "name": "units", "description": "One row per unit.",
    "fields": {
        "property": {"name": "Property", "description": "Header name. Same for every row."},
        "unit": {"name": "Unit", "description": "Unit id.", "primary": True},
        "rent": {"name": "Rent", "description": "Rent amount."},
    },
}


def _csv(path, units):
    lines = ["Willow Creek", "Unit,Charge,Amount"]
    for u in range(units):
        lines += [f"U{u:03d},Rent,{1000 + u}", ",Fee,25"]
    path.write_text("\n".join(lines) + "\n")
    return path


class CountingClient:
    """Answers every request with a valid, cheap-looking reply and counts them."""

    def __init__(self, cost=0.01):
        self.requests = []
        self.cost = cost
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.requests.append(params)
        system = params["messages"][0]["content"]
        rows_call = system in (pipeline.ROW_SYSTEM, pipeline.SHEET_ROW_SYSTEM)
        body = {"rows": []} if rows_call else {"fields": {}}
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, cost=self.cost)
        choice = SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)),
                                 finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model_extra={})


@pytest.fixture
def client(monkeypatch):
    c = CountingClient()
    monkeypatch.setattr(llm, "_get_client", lambda: c)
    monkeypatch.setattr(llm, "CACHE_ENABLED", False)  # every call is a real (fake) request
    pipeline._prep_memo.clear()
    return c


def _ctx(path, usage):
    return JobContext("job1", path, "csv", path.name, SCHEMA, set_stage=lambda s: None,
                      usage=usage)


def test_cost_cap_stops_the_job_with_partial(tmp_path, client):
    path = _csv(tmp_path / "rr.csv", units=400)  # ~800 lines -> many row chunks
    out = pipeline.run(_ctx(path, llm.Usage(max_cost_usd=0.05)))
    assert out.status == "PARTIAL"
    assert out.error["code"] == "BUDGET_EXCEEDED"
    # $0.05 at $0.01 per call: 5 calls, plus at most the ones already in flight.
    assert len(client.requests) <= 5 + pipeline.CONCURRENCY
    assert len(out.entries) >= 1 and set(out.entries[0]) == set(SCHEMA["fields"])


def test_call_cap_stops_the_job_with_partial(tmp_path, client):
    client.cost = 0.0
    path = _csv(tmp_path / "rr.csv", units=400)
    out = pipeline.run(_ctx(path, llm.Usage(max_calls=6)))
    assert out.status == "PARTIAL" and out.error["code"] == "BUDGET_EXCEEDED"
    assert len(client.requests) == 6


def test_every_request_has_minimal_reasoning_and_a_per_type_max_tokens(tmp_path, client):
    client.cost = 0.0
    pipeline.run(_ctx(_csv(tmp_path / "rr.csv", units=50), llm.Usage()))
    by_type = {}
    for r in client.requests:
        by_type.setdefault(r["messages"][0]["content"], set()).add(r["max_tokens"])
        assert r["model"] == llm.DEFAULT_MODEL
        assert r["extra_body"]["reasoning"] == {"effort": "minimal"}
    assert by_type[pipeline.PREP_SYSTEM] == {2000}
    assert by_type[pipeline.SHEET_ROW_SYSTEM] == {8000}
    assert max(max(v) for v in by_type.values()) <= 8000


def test_field_prep_is_batched_to_fit_its_token_limit(tmp_path, fake_llm):
    fields = {f"f{i}": {"name": f"F{i}", "description": "d"} for i in range(40)}
    schema = {"name": "big", "fields": fields}
    p = tmp_path / "x.csv"
    p.write_text("a,b\n")
    pipeline.run(JobContext("j", p, "csv", "x.csv", schema, lambda s: None))
    preps = [kw for kind, kw in fake_llm.kwargs if kind == "prep"]
    assert len(preps) == 3  # 40 fields in batches of 15
    assert all(kw["max_tokens"] <= 2000 for kw in preps)
    sent = [json.loads(m[1]["content"].split("FIELD LIST:\n", 1)[1])
            for kind, m in fake_llm.calls if kind == "prep"]
    assert sorted(k for s in sent for k in s["fields"]) == sorted(fields)


def test_wide_tables_get_fewer_records_per_chunk(tmp_path, fake_llm):
    fake_llm.prep = lambda m: {"fields": {k: {"scope": "row"} for k in SCHEMA["fields"]}}
    seen = []

    def rows(messages):
        seen.append(len(re.findall(r"row\d+: U\d{3}", rows_section(messages))))
        return {"rows": []}

    fake_llm.rows = rows
    pipeline.run(JobContext("j", _csv(tmp_path / "rr.csv", 100), "csv", "rr.csv", SCHEMA,
                            lambda s: None))
    limit = int(pipeline.ROWS_MAX_TOKENS * 0.7) // (3 * pipeline._TOKENS_PER_ROW_FIELD + 10)
    assert seen and max(seen) <= limit
    assert all(kw["max_tokens"] == 8000 for kind, kw in fake_llm.kwargs if kind == "rows")


# --------------------------------------------------------------------------- restart recovery


def _submit(tmp_path):
    path = config.FILES_DIR / "rr.csv"
    _csv(path, units=30)
    job, _ = db.create_job(org_id="", idem_key="k", filename="rr.csv", file_path="files/rr.csv",
                           file_type="csv", schema=SCHEMA, webhook_url=None)
    return job


def test_spend_survives_a_restart_and_the_budget_continues(tmp_path, client, monkeypatch):
    monkeypatch.setattr(llm, "MAX_JOB_COST_USD", 0.05)
    job = _submit(tmp_path)
    first = db.claim_job("w-before-restart", lease_seconds=-1)  # crashes: lease already gone
    assert db.record_spend(job["id"], "w-before-restart", 5, 0.05)  # spent the whole budget

    recovered = db.claim_job("w-after-restart")
    assert recovered["id"] == job["id"] and recovered["attempts"] == 2
    assert recovered["llm_calls"] == 5 and recovered["llm_cost_usd"] == pytest.approx(0.05)
    worker.process_job(recovered, "w-after-restart")

    assert client.requests == []  # the recovered attempt starts with no budget left
    final = db.get_job(job["id"], "")
    assert final["status"] == "PARTIAL" and final["error"]["code"] == "BUDGET_EXCEEDED"
    assert first["attempts"] == 1


def test_worker_persists_spend_after_every_request(tmp_path, client):
    job = _submit(tmp_path)
    claimed = db.claim_job("w1")
    worker.process_job(claimed, "w1")
    row = db.get_job(job["id"], "")
    assert row["llm_calls"] == len(client.requests) > 0
    assert row["llm_cost_usd"] == pytest.approx(0.01 * len(client.requests))


def test_recovered_jobs_are_capped_by_the_attempt_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_ATTEMPTS", 3)
    job = _submit(tmp_path)
    for n in range(3):  # three restarts, each one mid-job
        claimed = db.claim_job(f"w{n}", lease_seconds=-1)
        assert claimed["id"] == job["id"] and claimed["attempts"] == n + 1
    assert db.claim_job("w-final") is None  # not handed out a fourth time
    final = db.get_job(job["id"], "")
    assert final["status"] == "FAILED" and final["error"]["code"] == "WORKER_LOST"


def test_old_database_is_migrated(tmp_path, monkeypatch):
    import sqlite3
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE jobs (id TEXT PRIMARY KEY, org_id TEXT NOT NULL,
        idem_key TEXT NOT NULL UNIQUE, status TEXT NOT NULL, stage TEXT, filename TEXT,
        file_path TEXT NOT NULL, file_type TEXT NOT NULL, schema_json TEXT NOT NULL,
        webhook_url TEXT, result_json TEXT, error_json TEXT,
        attempts INTEGER NOT NULL DEFAULT 0, lease_owner TEXT, lease_expires REAL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    conn.commit()
    conn.close()
    monkeypatch.setattr(config, "DB_PATH", path)
    db.init_db()
    job, _ = db.create_job(org_id="", idem_key="k", filename="a", file_path="f", file_type="csv",
                           schema={"fields": {"a": {}}}, webhook_url=None)
    assert job["llm_calls"] == 0 and job["llm_cost_usd"] == 0
