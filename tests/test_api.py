import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import db, worker
from app.main import app

SCHEMA = json.dumps({"name": "t", "fields": {"a": {"name": "A", "description": "a"},
                                             "b": {"name": "B", "description": "b"}}})


@pytest.fixture
def client(fake_llm):
    with TestClient(app) as c:
        yield c


def _post(client, content=b"col1,col2\n1,2\n", name="x.csv", schema=SCHEMA, org=None):
    headers = {"X-Organization-Id": org} if org else {}
    return client.post("/jobs", files={"file": (name, content)}, data={"schema": schema},
                       headers=headers)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_submit_poll_and_complete(client):
    r = _post(client)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "PENDING" and body["result"] is None
    job = db.claim_job("test-worker")
    worker.process_job(job, "test-worker")
    done = client.get(f"/jobs/{body['job_id']}").json()
    assert done["status"] == "SUCCESS"
    entry = done["result"]["entries"][0]
    assert set(entry) == {"a", "b"}
    assert entry["a"]["value"] is None and entry["a"]["citation"] == ""
    assert {"model_calls", "tokens_in", "tokens_out", "cost_usd", "latency_ms"} <= set(
        done["result"]["metadata"])


def test_duplicate_returns_same_job_even_if_schema_is_reformatted(client):
    a = _post(client).json()
    b = _post(client, schema=json.dumps(json.loads(SCHEMA), indent=4)).json()
    assert a["job_id"] == b["job_id"]


def test_other_org_gets_404(client):
    job = _post(client, org="acme").json()
    assert client.get(f"/jobs/{job['job_id']}", headers={"X-Organization-Id": "acme"}).status_code == 200
    assert client.get(f"/jobs/{job['job_id']}", headers={"X-Organization-Id": "evil"}).status_code == 404
    assert client.get(f"/jobs/{job['job_id']}").status_code == 404


@pytest.mark.parametrize("kwargs,code", [
    ({"schema": "{not json"}, "INVALID_SCHEMA"),
    ({"schema": json.dumps({"fields": {}})}, "INVALID_SCHEMA"),
    ({"content": b"\x00\x01\x02binary"}, "UNSUPPORTED_FILE_TYPE"),
    ({"content": b""}, "EMPTY_FILE"),
])
def test_bad_requests_are_400(client, kwargs, code):
    r = _post(client, **kwargs)
    assert r.status_code == 400 and r.json()["code"] == code


def test_missing_parts_are_400_not_422(client):
    assert client.post("/jobs", data={"schema": SCHEMA}).status_code == 400
    assert client.post("/jobs", files={"file": ("x.csv", b"a,b\n")}).status_code == 400


def test_type_detected_from_content(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<x/>")
        z.writestr("xl/workbook.xml", "<x/>")
    r = _post(client, content=buf.getvalue(), name="no_extension")
    assert r.status_code == 202
    job = db.get_job(r.json()["job_id"], "")
    assert job["file_type"] == "xlsx"
    assert _post(client, content=b"%PDF-1.7\n...", name="a.bin").status_code == 202
