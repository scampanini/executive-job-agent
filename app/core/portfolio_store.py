import sqlite3
from typing import Any, Dict, List, Optional


def save_portfolio_item(
    conn: sqlite3.Connection,
    resume_id: int,
    raw_text: str,
    source_name: str = "Portfolio (pasted)",
    source_type: str = "paste",
    job_id: Optional[int] = None,
    url: Optional[str] = None,
) -> int:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("portfolio raw_text is empty")

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO portfolio_items (resume_id, job_id, source_name, source_type, url, raw_text)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (resume_id, job_id, source_name, source_type, url, raw_text),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_portfolio_items(
    conn: sqlite3.Connection,
    resume_id: int,
    job_id: Optional[int] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    if job_id is None:
        cur.execute(
            """
            SELECT * FROM portfolio_items
            WHERE resume_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (resume_id, limit),
        )
    else:
        cur.execute(
            """
            SELECT * FROM portfolio_items
            WHERE resume_id = ? AND (job_id = ? OR job_id IS NULL)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (resume_id, job_id, limit),
        )

    rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_portfolio_texts(
    conn: sqlite3.Connection,
    resume_id: int,
    job_id: Optional[int] = None,
    limit: int = 50,
) -> List[str]:
    items = list_portfolio_items(conn=conn, resume_id=resume_id, job_id=job_id, limit=limit)
    out: List[str] = []
    for it in items:
        txt = (it.get("raw_text") or "").strip()
        if txt:
            out.append(txt)
    return out
