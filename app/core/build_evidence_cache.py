import sqlite3
from typing import Optional

from app.core.grounded_extract import chunk_text, upsert_evidence_chunks


def build_evidence_cache_for_job(
    conn: sqlite3.Connection,
    resume_id: int,
    job_id: Optional[int],
    resume_text: str,
    portfolio_texts: Optional[list[str]] = None,
) -> None:
    # Resume chunks
    resume_chunks = chunk_text(resume_text)
    upsert_evidence_chunks(
        conn=conn,
        resume_id=resume_id,
        job_id=job_id,
        source_type="resume",
        source_name="resume",
        section="resume",
        chunks=[c for _, c in resume_chunks],
    )

    # Portfolio chunks
    for idx, pt in enumerate(portfolio_texts or []):
        label = f"portfolio_{idx+1}"
        p_chunks = chunk_text(pt)
        upsert_evidence_chunks(
            conn=conn,
            resume_id=resume_id,
            job_id=job_id,
            source_type="portfolio",
            source_name=label,
            section="portfolio",
            chunks=[c for _, c in p_chunks],
        )
