"""SQLite database connection (SQLAlchemy 2)."""
import sys
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


class Base(DeclarativeBase):
    pass


SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False)
engine = None


def init_db(url: str | None = None):
    """Create the engine and tables once. Calling it again without a URL does nothing,
    so tests can point it at a temporary database first."""
    global engine
    if engine is not None and url is None:
        return engine
    url = url or f"sqlite:///{config.DB_PATH}"
    # check_same_thread=False: grading runs in a background thread
    engine = create_engine(url, connect_args={"check_same_thread": False})
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_settings(dbapi_conn, _record):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")     # pages can read while grading writes
            cur.execute("PRAGMA busy_timeout=10000")   # wait up to 10 s instead of "database is locked"
            cur.close()
    from app import models  # noqa: F401  (registers the tables)

    Base.metadata.create_all(engine)
    SessionLocal.configure(bind=engine)
    return engine


def get_db():
    """FastAPI dependency: one session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()