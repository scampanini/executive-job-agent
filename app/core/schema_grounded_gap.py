import sqlite3


def ensure_grounded_gap_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            job_id INTEGER,
            source_name TEXT NOT NULL,
            source_type TEXT NOT NULL,  -- e.g. "paste", "url", "doc", "pdf"
            url TEXT,
            raw_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            job_id INTEGER,
            source_type TEXT NOT NULL,     -- "resume" or "portfolio"
            source_name TEXT NOT NULL,     -- file name or label
            section TEXT,
            chunk_text TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            entities_json TEXT NOT NULL,
            signals_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(resume_id, job_id, source_type, source_name, content_hash)
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS grounded_gap_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id INTEGER NOT NULL,
            job_id INTEGER NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(resume_id, job_id)
        );
        """
    )

    conn.commit()
