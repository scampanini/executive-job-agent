import sqlite3
from typing import Optional


def _find_first_existing_column(conn: sqlite3.Connection, table: str, candidates: list[str]) -> Optional[str]:
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]  # type: ignore[index]
    except Exception:
        return None

    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def get_job_description(conn: sqlite3.Connection, job_id: int) -> str:
    for table in ["jobs", "job", "job_posts", "job_post"]:
        col = _find_first_existing_column(conn, table, ["description", "job_desc", "job_description", "raw_text", "text"])
        id_col = _find_first_existing_column(conn, table, ["id", "job_id"])
        if not col or not id_col:
            continue
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT {col} FROM {table} WHERE {id_col} = ?", (job_id,))
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0])
        except Exception:
            continue
    return ""


def get_resume_text(conn: sqlite3.Connection, resume_id: int) -> str:
    for table in ["resumes", "resume"]:
        col = _find_first_existing_column(conn, table, ["raw_text", "text", "content", "resume_text"])
        id_col = _find_first_existing_column(conn, table, ["id", "resume_id"])
        if not col or not id_col:
            continue
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT {col} FROM {table} WHERE {id_col} = ?", (resume_id,))
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0])
        except Exception:
            continue
    return ""
