import pytest

from app import config, db


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
