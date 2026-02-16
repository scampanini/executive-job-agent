from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import time

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

    conn.commit()
    conn.close()


def save_resume(source: str, raw_text: str, db_path: Path = DEFAULT_DB) -> int:
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


def list_recent_scores(limit: int = 20, db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
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

def create_pipeline_item(
    job_id: int,
    stage: str,
    next_action_date: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Path = DEFAULT_DB,
) -> int:
    conn = get_conn(db_path)
    cur = conn.cursor()
    now = int(time.time())
    cur.execute(
        "INSERT INTO pipeline (created_at, updated_at, job_id, stage, next_action_date, notes, is_active) "
        "VALUES (?, ?, ?, ?, ?, ?, 1)",
        (now, now, job_id, stage, next_action_date, notes),
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
    db_path: Path = DEFAULT_DB,
) -> None:
    conn = get_conn(db_path)
    cur = conn.cursor()
    now = int(time.time())
    cur.execute(
        "UPDATE pipeline SET updated_at=?, stage=?, next_action_date=?, notes=?, is_active=? WHERE id=?",
        (now, stage, next_action_date, notes, 1 if is_active else 0, pipeline_id),
    )
    conn.commit()
    conn.close()


def list_pipeline_items(active_only: bool = True, limit: int = 50, db_path: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    conn = get_conn(db_path)
    cur = conn.cursor()

    where = "WHERE pipeline.is_active=1" if active_only else ""
    cur.execute(
        "SELECT pipeline.id, pipeline.created_at, pipeline.updated_at, pipeline.stage, pipeline.next_action_date, pipeline.notes, "
        "job.id, job.company, job.title, job.location, job.url "
        "FROM pipeline JOIN job ON job.id = pipeline.job_id "
        f"{where} "
        "ORDER BY pipeline.updated_at DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for (
        pipeline_id, created_at, updated_at, stage, next_action_date, notes,
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
                "job_id": job_id,
                "company": company,
                "title": title,
                "location": location,
                "url": url,
            }
        )
    return out

