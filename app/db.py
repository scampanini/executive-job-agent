import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from app.utils import safe_text

DB_PATH = os.getenv("APP_DB_PATH", "app_data.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT NOT NULL,
        company TEXT,
        title TEXT,
        location TEXT,
        url TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        raw_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        resume_id INTEGER NOT NULL,
        model TEXT,
        result_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS grounded_gap_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_id INTEGER NOT NULL,
        job_id INTEGER NOT NULL,
        result_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_id INTEGER,
        job_id INTEGER,
        text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()


def save_job(
    conn: sqlite3.Connection,
    description: str,
    company: Optional[str] = None,
    title: Optional[str] = None,
    location: Optional[str] = None,
    url: Optional[str] = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO jobs (description, company, title, location, url)
        VALUES (?, ?, ?, ?, ?)
        """,
        (description, company, title, location, url),
    )
    conn.commit()
    return int(cur.lastrowid)


def save_resume(conn: sqlite3.Connection, source: str, raw_text: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO resumes (source, raw_text)
        VALUES (?, ?)
        """,
        (source, raw_text),
    )
    conn.commit()
    return int(cur.lastrowid)


def save_score(
    conn: sqlite3.Connection,
    job_id: int,
    resume_id: int,
    result: Dict[str, Any],
    model: str,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO scores (job_id, resume_id, model, result_json)
        VALUES (?, ?, ?, ?)
        """,
        (job_id, resume_id, model, json.dumps(result)),
    )
    conn.commit()
    return int(cur.lastrowid)


def save_grounded_gap_result(
    conn: sqlite3.Connection,
    resume_id: int,
    job_id: int,
    result: Dict[str, Any],
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO grounded_gap_results (resume_id, job_id, result_json)
        VALUES (?, ?, ?)
        """,
        (resume_id, job_id, json.dumps(result)),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_latest_grounded_gap_result(conn: sqlite3.Connection, job_id: int) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT result_json
        FROM grounded_gap_results
        WHERE job_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (job_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row["result_json"])
    except Exception:
        return None


def list_scores(conn: sqlite3.Connection, limit: int = 100) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT s.id, s.job_id, s.resume_id, s.model, s.result_json, s.created_at,
               j.company, j.title, j.location
        FROM scores s
        LEFT JOIN jobs j ON j.id = s.job_id
        ORDER BY s.id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            result = json.loads(row["result_json"])
        except Exception:
            result = {}
        out.append(
            {
                "id": row["id"],
                "job_id": row["job_id"],
                "resume_id": row["resume_id"],
                "company": row["company"],
                "title": row["title"],
                "location": row["location"],
                "model": row["model"],
                "result": result,
                "created_at": row["created_at"],
            }
        )
    return out


def save_portfolio_text(
    conn: sqlite3.Connection,
    text: str,
    resume_id: Optional[int] = None,
    job_id: Optional[int] = None,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO portfolio_items (resume_id, job_id, text)
        VALUES (?, ?, ?)
        """,
        (resume_id, job_id, safe_text(text)),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_portfolio_texts(
    conn: sqlite3.Connection,
    resume_id: Optional[int] = None,
    job_id: Optional[int] = None,
    limit: int = 50,
) -> List[str]:
    cur = conn.cursor()

    if resume_id and job_id:
        cur.execute(
            """
            SELECT text
            FROM portfolio_items
            WHERE resume_id = ? OR job_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (resume_id, job_id, limit),
        )
    elif resume_id:
        cur.execute(
            """
            SELECT text
            FROM portfolio_items
            WHERE resume_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (resume_id, limit),
        )
    elif job_id:
        cur.execute(
            """
            SELECT text
            FROM portfolio_items
            WHERE job_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (job_id, limit),
        )
    else:
        cur.execute(
            """
            SELECT text
            FROM portfolio_items
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cur.fetchall()
    return [safe_text(r["text"]) for r in rows if safe_text(r["text"])]
