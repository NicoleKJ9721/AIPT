import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect, text


os.environ.setdefault("AIPT_DATABASE_URL", "sqlite:///:memory:")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402


def test_default_sqlite_path_prefers_legacy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_backend_dir = tmp_path / "backend"
    legacy_dir = fake_backend_dir / "data"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / "aipt.db"
    legacy_path.write_text("", encoding="utf-8")

    # Point the module's __file__ at a temp tree so the legacy lookup is isolated
    # from the real repository path.
    monkeypatch.setattr(db, "__file__", str(fake_backend_dir / "db.py"))

    assert Path(db._default_sqlite_path()) == legacy_path


def test_default_sqlite_path_uses_app_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_backend_dir = tmp_path / "backend"
    fake_backend_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(db, "__file__", str(fake_backend_dir / "db.py"))

    app_home = tmp_path / "app_home"
    monkeypatch.setattr(db, "app_home_dir", lambda: app_home)

    resolved = Path(db._default_sqlite_path())
    assert resolved == (app_home / "aipt.db").resolve()
    assert resolved.parent.exists()


def test_database_url_defaults_to_sqlite(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AIPT_DATABASE_URL", raising=False)
    monkeypatch.setattr(db, "_default_sqlite_path", lambda: r"C:\tmp\aipt.db")

    assert db._database_url() == "sqlite:///C:/tmp/aipt.db"


def test_create_engine_handles_file_sqlite(tmp_path: Path):
    path = (tmp_path / "test.db").resolve()
    url = f"sqlite:///{str(path).replace(os.sep, '/')}"
    engine = db._create_engine(url)

    with engine.begin() as conn:
        conn.execute(text("SELECT 1"))


def test_create_engine_accepts_alternate_sqlite_scheme():
    # Coverage for the non `sqlite:///` branch while staying on stdlib SQLite.
    engine = db._create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("SELECT 1"))


def test_set_sqlite_pragma_is_best_effort(caplog: pytest.LogCaptureFixture):
    class BadConnection:
        def cursor(self):
            raise RuntimeError("boom")

    with caplog.at_level("DEBUG"):
        db._set_sqlite_pragma(BadConnection(), None)


def test_migrate_sqlite_noop_when_not_sqlite():
    class FakeDialect:
        name = "postgres"

    class FakeEngine:
        dialect = FakeDialect()

    db._migrate_sqlite(FakeEngine())  # should not crash


def test_migrate_sqlite_skips_when_tables_missing():
    engine = db._create_engine("sqlite:///:memory:")
    db._migrate_sqlite(engine)  # no tables -> no ALTERs


def test_migrate_sqlite_adds_missing_columns():
    engine = db._create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE images (id VARCHAR PRIMARY KEY)"))
        conn.execute(text("CREATE TABLE projects (id VARCHAR PRIMARY KEY)"))

    db._migrate_sqlite(engine)

    inspector = inspect(engine)
    image_cols = {c["name"] for c in inspector.get_columns("images")}
    project_cols = {c["name"] for c in inspector.get_columns("projects")}

    assert {"dataset_id", "dataset_file_id"}.issubset(image_cols)
    assert {"storage_root"}.issubset(project_cols)

