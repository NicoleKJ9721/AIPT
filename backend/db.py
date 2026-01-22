from __future__ import annotations

import os
import logging
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app_config import app_home_dir

logger = logging.getLogger(__name__)

def _default_sqlite_path() -> str:
    """
    Default SQLite path (local-first, out-of-repo), with legacy fallback.

    - If a legacy DB exists under `backend/data/aipt.db`, keep using it to avoid
      surprising data loss when upgrading.
    - Otherwise, store the DB under the per-user app home directory.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    legacy_dir = os.path.join(base_dir, "data")
    legacy_path = os.path.join(legacy_dir, "aipt.db")
    if os.path.exists(legacy_path):
        os.makedirs(legacy_dir, exist_ok=True)
        return legacy_path

    home_path = str(app_home_dir() / "aipt.db")
    os.makedirs(os.path.dirname(home_path), exist_ok=True)
    return home_path


def _database_url() -> str:
    env = os.getenv("AIPT_DATABASE_URL")
    if env:
        return env
    default_path = _default_sqlite_path().replace("\\", "/")
    return f"sqlite:///{default_path}"


DATABASE_URL = _database_url()


def _create_engine(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        connect_args = {"check_same_thread": False}
        if url.endswith(":memory:"):
            return create_engine(
                url,
                connect_args=connect_args,
                poolclass=StaticPool,
            )
        return create_engine(url, connect_args=connect_args)
    return create_engine(url)


engine = _create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
    # Enforce FK constraints on SQLite.
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        # Best-effort only: keep the app running even if the PRAGMA fails.
        logger.debug("Failed to set SQLite PRAGMA foreign_keys=ON", exc_info=True)


class Base(DeclarativeBase):
    pass


def _migrate_sqlite(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    def ensure_column(table: str, column: str, ddl: str) -> None:
        if table not in table_names:
            return
        cols = {c["name"] for c in inspector.get_columns(table)}
        if column in cols:
            return
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))

    # Backfill new optional linkage columns (for existing aipt.db files).
    ensure_column("images", "dataset_id", "dataset_id VARCHAR")
    ensure_column("images", "dataset_file_id", "dataset_file_id VARCHAR")
    ensure_column("projects", "storage_root", "storage_root VARCHAR")


def init_db() -> None:
    from db_models import (  # noqa: F401
        Annotation,
        Dataset,
        DatasetFile,
        Image,
        LabelClass,
        Project,
        TrainedModel,
    )

    Base.metadata.create_all(bind=engine)
    _migrate_sqlite(engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
