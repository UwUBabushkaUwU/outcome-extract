import pytest

from app import config, db, llm, pipeline


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Every test gets its own DB, file store and LLM cache."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "data" / "jobs.db")
    monkeypatch.setattr(config, "FILES_DIR", tmp_path / "data" / "files")
    monkeypatch.setattr(config, "TMP_DIR", tmp_path / "data" / "tmp")
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    for d in (config.FILES_DIR, config.TMP_DIR):
        d.mkdir(parents=True)
    db.init_db()
    yield tmp_path


class FakeLLM:
    """Stands in for llm.chat_json. Routes on the system prompt; handlers can be replaced
    per test. Records every call."""

    def __init__(self):
        self.calls = []
        self.prep = lambda msgs: {"fields": {}}
        self.doc = lambda msgs: {"fields": {}}
        self.rows = lambda msgs: {"rows": []}
        self.transcribe = lambda msgs: {"text": ""}

    def __call__(self, messages, *, usage, model=llm.DEFAULT_MODEL, **kw):
        system = messages[0]["content"]
        kind = {pipeline.PREP_SYSTEM: "prep", pipeline.DOC_SYSTEM: "doc",
                pipeline.ROW_SYSTEM: "rows", pipeline.TRANSCRIBE_SYSTEM: "transcribe"}[system]
        self.calls.append((kind, messages))
        usage.record(model, 10, 5, 0.0001)
        out = getattr(self, kind)(messages)
        if isinstance(out, Exception):
            raise out
        return out

    def count(self, kind):
        return sum(1 for k, _ in self.calls if k == kind)


@pytest.fixture
def fake_llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(llm, "chat_json", fake)
    pipeline._prep_memo.clear()
    return fake
