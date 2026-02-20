from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# storage.py is at: repo/app/core/storage.py -> parents[3] is repo root
DEFAULT_DB = Path(__file__).resolve().parents[3] / "job_agent.sqlite3"


def get_conn(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path = DEFAULT_DB) -> None:
    conn = get_conn(db_path)
    cur = conn.cursor()

    cur.execute(
        "CREATE TABLE IF NOT EXISTS resume ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "created_at INTEGER NOT NULL,"
        "source TEXT NOT NULL,"
        "raw_text TEXT NOT NULL)"
    )

    cur.execute(
        "CREATE TABLE IF NOT EXISTS job ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "created_at INTEGER NOT NULL,"
        "company TEXT,"
        "title TEXT,"
        "location TEXT,"
        "url TEXT,"
        "description TEXT NOT NULL)"
    )

    cur.execute(
        "CREATE TABLE IF NOT EXISTS score ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "created_at INTEGER NOT NULL,"
        "job_id INTEGER NOT NULL,"
        "resume_id INTEGER NOT NULL,"
        "model TEXT,"
        "result_json TEXT NOT NULL,"
        "FOREIGN KEY(job_id) REFERENCES job(id),"
        "FOREIGN KEY(resume_id) REFERENCES resume(id))"
    )

    cur.execute(
        "CREATE TABLE IF NOT EXISTS pipeline ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "created_at INTEGER NOT NULL,"
        "updated_at INTEGER NOT NULL,"
        "job_id INTEGER NOT NULL,"
        "stage TEXT NOT NULL,"
        "next_action_date TEXT,"
        "notes TEXT,"
        "is_active INTEGER NOT NULL DEFAULT 1,"
        "FOREIGN KEY(job_id) REFERENCES job(id))"
    )

    # Migration-safe additions
    try:
        cur.execute("ALTER TABLE pipeline ADD COLUMN fit_score REAL")
    except sqlite3.OperationalError:
        pass

    try:
        cur.execute("ALTER TABLE pipeline ADD COLUMN priority TEXT")
    except sqlite3.OperationalError:
        pass
    # -------------------------
    # Phase 3: settings (email + feature flags)
    # -------------------------
    cur.execute(
        "CREATE TABLE IF NOT EXISTS settings ("
        "key TEXT PRIMARY KEY,"
        "value TEXT NOT NULL,"
        "updated_at INTEGER NOT NULL)"
    )

    # -------------------------
    # Phase 3: documents (resume + portfolio)
    # -------------------------
    cur.execute(
        "CREATE TABLE IF NOT EXISTS documents ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "created_at INTEGER NOT NULL,"
        "doc_type TEXT NOT NULL,"  # 'resume' | 'portfolio'
        "source TEXT,"             # filename or origin label
        "mime TEXT,"               # e.g., application/pdf
        "raw_text TEXT NOT NULL,"
        "text_hash TEXT)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_type_created ON documents(doc_type, created_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(text_hash)"
    )

    # -------------------------
    # Phase 3: Gmail ingest storage (read-only first)
    # -------------------------
    cur.execute(
        "CREATE TABLE IF NOT EXISTS emails_raw ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "ingested_at INTEGER NOT NULL,"
        "gmail_message_id TEXT NOT NULL,"      # Gmail API message.id (unique per mailbox)
        "thread_id TEXT,"
        "rfc_message_id TEXT,"                 # RFC Message-ID header if available
        "internal_date_ms INTEGER,"            # Gmail internalDate
        "from_email TEXT,"
        "subject TEXT,"
        "snippet TEXT,"
        "headers_json TEXT,"
        "body_text_sanitized TEXT,"
        "raw_json TEXT,"
        "UNIQUE(gmail_message_id))"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_emails_raw_rfc_mid ON emails_raw(rfc_message_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_emails_raw_internal_date ON emails_raw(internal_date_ms)"
    )

    cur.execute(
        "CREATE TABLE IF NOT EXISTS email_ingest_runs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "started_at INTEGER NOT NULL,"
        "finished_at INTEGER,"
        "target_email TEXT,"
        "query TEXT,"
        "max_results INTEGER,"
        "fetched_count INTEGER NOT NULL DEFAULT 0,"
        "inserted_count INTEGER NOT NULL DEFAULT 0,"
        "skipped_count INTEGER NOT NULL DEFAULT 0,"
        "status TEXT NOT NULL,"                 # 'started'|'ok'|'error'
        "error_text TEXT)"
    )

    # -------------------------
    # Phase 3: gap questions (after resume+portfolio analysis)
    # -------------------------
    cur.execute(
        "CREATE TABLE IF NOT EXISTS gap_questions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "created_at INTEGER NOT NULL,"
        "job_id INTEGER,"
        "gap_type TEXT,"
        "question TEXT NOT NULL,"
        "answer TEXT,"
        "answered_at INTEGER,"
        "FOREIGN KEY(job_id) REFERENCES job(id))"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_gap_questions_job ON gap_questions(job_id, created_at)"
    )
    
    conn.commit()
    conn.close()


def save_resume(source: str, raw_text: str, db_path: Path = DEFAULT_DB) -> int:
    init_db(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO resume (created_at, source, raw_text) VALUES (?, ?, ?)",
        (int(time.time()), source, raw_text),
    )
    conn.commit()
    rid = int(cur.lastrowid)
    conn.close()
    return rid


def save_job(
    description: str,
    company: Optional[str] = None,
    title: Optional[str] = None,
    location: Optional[str] = None,
    url: Optional[str] = None,
    db_path: Path = DEFAULT_DB,
) -> int:
    init_db(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO job (created_at, company, title, location, url, description) VALUES (?, ?, ?, ?, ?, ?)",
        (int(time.time()), company, title, location, url, description),
    )
    conn.commit()
    jid = int(cur.lastrowid)
    conn.close()
    return jid


def save_score(
    job_id: int,
    resume_id: int,
    result: Dict[str, Any],
    model: Optional[str] = None,
    db_path: Path = DEFAULT_DB,
) -> int:
    init_db(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO score (created_at, job_id, resume_id, model, result_json) VALUES (?, ?, ?, ?, ?)",
        (int(time.time()), job_id, resume_id, model, json.dumps(result, ensure_ascii=False)),
    )
    conn.commit()
    sid = int(cur.lastrowid)
    conn.close()
    return sid


def list_recent_scores(limit: int = 20, db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT score.created_at, job.company, job.title, job.location, score.model, score.result_json "
        "FROM score JOIN job ON job.id = score.job_id "
        "ORDER BY score.created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for created_at, company, title, location, model, result_json in rows:
        out.append(
            {
                "created_at": created_at,
                "company": company,
                "title": title,
                "location": location,
                "model": model,
                "result": json.loads(result_json),
            }
        )
    return out


# -------------------------
# Pipeline helpers (CRM)
# -------------------------

def create_pipeline_item(
    job_id: int,
    stage: str,
    next_action_date: Optional[str] = None,
    notes: Optional[str] = None,
    fit_score: Optional[float] = None,
    priority: Optional[str] = None,
    db_path: Path = DEFAULT_DB,
) -> int:
    init_db(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()
    now = int(time.time())
    cur.execute(
        "INSERT INTO pipeline (created_at, updated_at, job_id, stage, next_action_date, notes, fit_score, priority, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
        (now, now, job_id, stage, next_action_date, notes, fit_score, priority),
    )
    conn.commit()
    pid = int(cur.lastrowid)
    conn.close()
    return pid


def update_pipeline_item(
    pipeline_id: int,
    stage: str,
    next_action_date: Optional[str] = None,
    notes: Optional[str] = None,
    is_active: bool = True,
    fit_score: Optional[float] = None,
    priority: Optional[str] = None,
    db_path: Path = DEFAULT_DB,
) -> None:
    init_db(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()
    now = int(time.time())
    cur.execute(
        "UPDATE pipeline SET updated_at=?, stage=?, next_action_date=?, notes=?, is_active=?, fit_score=?, priority=? WHERE id=?",
        (now, stage, next_action_date, notes, 1 if is_active else 0, fit_score, priority, pipeline_id),
    )
    conn.commit()
    conn.close()


def list_pipeline_items(
    active_only: bool = True,
    limit: int = 200,
    db_path: Path = DEFAULT_DB,
) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()

    where_clause = "WHERE pipeline.is_active=1" if active_only else ""
    cur.execute(
        "SELECT pipeline.id, pipeline.created_at, pipeline.updated_at, pipeline.stage, pipeline.next_action_date, pipeline.notes, "
        "pipeline.fit_score, pipeline.priority, "
        "job.id, job.company, job.title, job.location, job.url "
        "FROM pipeline JOIN job ON job.id = pipeline.job_id "
        f"{where_clause} "
        "ORDER BY pipeline.updated_at DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for (
        pipeline_id, created_at, updated_at, stage, next_action_date, notes,
        fit_score, priority,
        job_id, company, title, location, url
    ) in rows:
        out.append(
            {
                "pipeline_id": pipeline_id,
                "created_at": created_at,
                "updated_at": updated_at,
                "stage": stage,
                "next_action_date": next_action_date,
                "notes": notes,
                "fit_score": fit_score,
                "priority": priority,
                "job_id": job_id,
                "company": company,
                "title": title,
                "location": location,
                "url": url,
            }
        )
    return out
