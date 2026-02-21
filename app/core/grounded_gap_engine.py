import json
import math
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from app.core.semantic_match import semantic_enabled, semantic_similarity
from app.core.grounded_extract import EvidenceItem, extract_requirements_deterministic, load_evidence_index, tag_and_extract_signals


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    out = []
    buf = []
    for ch in s:
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    # drop tiny tokens
    return [t for t in out if len(t) >= 3]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa.intersection(sb))
    union = len(sa.union(sb))
    return inter / union if union else 0.0


def _best_evidence_for_requirement(
    req_text: str,
    req_competency: str,
    evidence: List[EvidenceItem],
    top_k: int = 3,
) -> List[Tuple[EvidenceItem, float, str]]:
    """
    Hybrid matcher:
      - base deterministic: token Jaccard + tag overlap + evidence confidence
      - optional semantic re-rank on top candidates using embeddings (if enabled)
    """
    req_tokens = _tokenize(req_text)
    req_tags, _, _, _ = tag_and_extract_signals(req_text)
    req_tag_set = set(req_tags + ([req_competency] if req_competency else []))

    scored: List[Tuple[EvidenceItem, float, str]] = []
    for e in evidence:
        ev_tokens = _tokenize(e.chunk_text)
        tok_sim = _jaccard(req_tokens, ev_tokens)

        ev_tag_set = set(e.tags)
        tag_overlap = req_tag_set.intersection(ev_tag_set)
        tag_bonus = 0.0
        if tag_overlap:
            tag_bonus = 0.20 + min(0.20, 0.05 * len(tag_overlap))

        conf_bonus = 0.10 * float(e.confidence)

        score = tok_sim + tag_bonus + conf_bonus
        score = max(0.0, min(1.25, score))

        if score <= 0.05:
            continue

        rationale_parts = []
        if tok_sim >= 0.10:
            rationale_parts.append(f"token_overlap={tok_sim:.2f}")
        if tag_overlap:
            rationale_parts.append(f"tags={sorted(tag_overlap)}")
        if e.confidence >= 0.6:
            rationale_parts.append("strong_signal")

        scored.append((e, score, "; ".join(rationale_parts) if rationale_parts else "weak_match"))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Optional semantic re-rank
    if semantic_enabled() and scored:
        N = min(30, len(scored))
        candidates = [scored[i][0].chunk_text for i in range(N)]
        sims = semantic_similarity(req_text, candidates)

        if sims is not None:
            sim_by_idx = {i: s for (i, s) in sims}
            blended: List[Tuple[EvidenceItem, float, str]] = []

            for i in range(N):
                e, base, rat = scored[i]
                sem = max(0.0, min(1.0, float(sim_by_idx.get(i, 0.0))))
                new_score = float(min(1.25, base + 0.35 * sem))
                new_rat = rat + f"; semantic={sem:.2f}"
                blended.append((e, new_score, new_rat))

            tail = scored[N:]
            blended.sort(key=lambda x: x[1], reverse=True)
            scored = blended + tail
            scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def _classify(match_strength: float, must_have: bool) -> str:
    if match_strength >= 0.65:
        return "match"
    if match_strength >= 0.35:
        return "partial"
    # Below this is gap; for must_have treat as hard gap, else signal gap
    return "gap" if must_have else "signal_gap"


def _to_pct(x: float) -> int:
    return int(round(max(0.0, min(1.0, x)) * 100))


def run_grounded_gap_analysis(
    conn: sqlite3.Connection,
    resume_id: int,
    job_id: int,
    job_description: str,
    evidence_limit: int = 5000,
) -> Dict[str, Any]:
    """
    Deterministic grounded gap analysis:
      - requirements from JD (deterministic)
      - evidence from cached evidence_chunks
      - match via token overlap + tag overlap + confidence
    """
    requirements = extract_requirements_deterministic(job_description)
    evidence = load_evidence_index(conn=conn, resume_id=resume_id, job_id=job_id, limit=evidence_limit)

    results: List[Dict[str, Any]] = []
    total_weight = 0
    score_accum = 0.0

    for req in requirements:
        req_text = str(req.get("text") or "").strip()
        if not req_text:
            continue

        competency = str(req.get("competency") or "general")
        must_have = bool(req.get("must_have"))
        weight = int(req.get("weight") or 1)

        best = _best_evidence_for_requirement(req_text=req_text, req_competency=competency, evidence=evidence, top_k=3)

        if best:
            best_score = float(best[0][1])
        else:
            best_score = 0.0

        classification = _classify(best_score, must_have=must_have)

        # score contribution (grounded + explainable)
        # matches get full weight, partial gets half, gaps get none (must-have gaps add a penalty)
        contrib = 0.0
        if classification == "match":
            contrib = 1.0
        elif classification == "partial":
            contrib = 0.5
        elif classification == "gap":
            contrib = -0.25  # penalize must-have gaps lightly
        else:
            contrib = 0.0

        total_weight += weight
        score_accum += contrib * weight

        ev_list = []
        for e, sc, rationale in best:
            ev_list.append(
                {
                    "evidence_id": e.evidence_id,
                    "source_type": e.source_type,
                    "source_name": e.source_name,
                    "section": e.section,
                    "quote": e.chunk_text[:600],
                    "match_strength": float(sc),
                    "rationale": rationale,
                }
            )

        # missing signals: simple, based on competency tags
        missing_signals = []
        if classification in ("gap", "signal_gap"):
            missing_signals.append(competency)

        followup_question = ""
        if classification in ("partial", "gap", "signal_gap"):
            followup_question = (
                f"Provide a specific example demonstrating '{req_text}'. "
                f"Include scope (team/budget), stakeholders, and measurable outcomes."
            )

        results.append(
            {
                "requirement_id": req.get("requirement_id"),
                "category": req.get("category"),
                "competency": competency,
                "text": req_text,
                "weight": weight,
                "must_have": must_have,
                "classification": classification,
                "match_strength": float(best_score),
                "match_strength_pct": _to_pct(best_score),
                "evidence": ev_list,
                "missing_signals": missing_signals,
                "followup_question": followup_question,
                "confidence": float(min(1.0, 0.35 + 0.5 * best_score)),
            }
        )

    # Normalize score into 0-100
    if total_weight <= 0:
        overall = 0
    else:
        # score_accum can be negative; map roughly to 0..100
        # baseline assumes partial across all => 50
        raw = score_accum / float(total_weight)  # roughly -0.25..1.0
        # map [-0.25, 1.0] to [0, 100]
        overall = int(round((raw + 0.25) / 1.25 * 100))
        overall = max(0, min(100, overall))

    buckets = {"match": [], "partial": [], "gap": [], "signal_gap": []}
    for r in results:
        buckets.get(r["classification"], buckets["signal_gap"]).append(r)

    summary = (
        f"Overall grounded alignment: {overall}/100. "
        f"Matches: {len(buckets['match'])}, Partial: {len(buckets['partial'])}, "
        f"Gaps: {len(buckets['gap'])}, Signal gaps: {len(buckets['signal_gap'])}."
    )

    return {
        "overall_alignment_score": overall,
        "summary": summary,
        "requirements_total": len(results),
        "matched_requirements": buckets["match"],
        "partial_gaps": buckets["partial"],
        "hard_gaps": buckets["gap"],
        "signal_gaps": buckets["signal_gap"],
        "all_results": results,
    }


def save_grounded_gap_result(
    conn: sqlite3.Connection,
    resume_id: int,
    job_id: int,
    result: Dict[str, Any],
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO grounded_gap_results (resume_id, job_id, result_json)
        VALUES (?, ?, ?)
        ON CONFLICT(resume_id, job_id) DO UPDATE SET
            result_json=excluded.result_json,
            created_at=datetime('now')
        """,
        (resume_id, job_id, json.dumps(result, ensure_ascii=False)),
    )
    conn.commit()


def load_grounded_gap_result(
    conn: sqlite3.Connection,
    resume_id: int,
    job_id: int,
) -> Optional[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT result_json FROM grounded_gap_results
        WHERE resume_id = ? AND job_id = ?
        """,
        (resume_id, job_id),
    )
    row = cur.fetchone()
    if not row:
        return None
    try:
        return json.loads(row["result_json"])
    except Exception:
        return None
