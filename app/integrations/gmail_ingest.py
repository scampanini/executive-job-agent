from __future__ import annotations

from typing import Dict, Optional

from app.core.storage import start_email_ingest_run, finish_email_ingest_run


def ingest_gmail_readonly_stub(
    *,
    target_email: Optional[str],
    query: str,
    max_results: int = 50,
) -> Dict[str, int]:
    """
    Phase 3 stub: does NOT call Gmail yet.
    It only records an ingest run and returns counts = 0.

    This keeps the UI + feature flag deploy-safe before OAuth is added.
    """
    run_id = start_email_ingest_run(
        target_email=target_email,
        query=query,
        max_results=int(max_results),
    )

    try:
        finish_email_ingest_run(
            run_id,
            status="ok",
            fetched_count=0,
            inserted_count=0,
            skipped_count=0,
            error_text=None,
        )
        return {"fetched": 0, "inserted": 0, "skipped": 0}
    except Exception as e:
        finish_email_ingest_run(
            run_id,
            status="error",
            fetched_count=0,
            inserted_count=0,
            skipped_count=0,
            error_text=str(e),
        )
        raise
