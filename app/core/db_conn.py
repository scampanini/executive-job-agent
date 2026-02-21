import os
import sqlite3
from typing import Optional


def get_db_path() -> str:
    # Override if you already use a different DB path/env var.
    return os.getenv("APP_DB_PATH", "data/app.db")


def get_conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
