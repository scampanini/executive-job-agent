# app/core/objective_requirements.py
from __future__ import annotations

import re
from typing import Any, Dict, Tuple, Optional

DEGREE_REGEX = re.compile(r"\b(bachelor|b\.?s\.?|b\.?a\.?)\b", re.I)
YEAR_REGEX = re.compile(r"\b(19[7-9]\d|20[0-2]\d)\b")


def infer_has_bachelors(resume_text: str) -> bool:
    txt = resume_text or ""
    return bool(DEGREE_REGEX.search(txt)) and ("university" in txt.lower() or "college" in txt.lower())


def infer_years_experience(resume_text: str) -> Optional[int]:
    txt = resume_text or ""
    years = [int(y) for y in YEAR_REGEX.findall(txt)]
    if len(years) < 2:
        return None
    return max(years) - min(years)


def apply_objective_overrides(
    gap_result: Dict[str, Any],
    resume_text: str,
    *,
    degree_req_ids: Tuple[str, ...] = ("REQ-009",),
    years_req_ids: Tuple[str, ...] = ("REQ-010",),
    years_threshold: int = 15,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    audit: Dict[str, Any] = {"overrides": []}
    if not gap_result:
        return gap_result, audit

    all_results = gap_result.get("all_results") or []
    if not isinstance(all_results, list):
        return gap_result, audit

    has_deg = infer_has_bachelors(resume_text)
    yrs = infer_years_experience(resume_text)

    for r in all_results:
        rid = r.get("requirement_id")
        if rid in degree_req_ids and has_deg:
            if r.get("classification") != "match":
                r["classification"] = "match"
                r["match_strength"] = max(float(r.get("match_strength") or 0.0), 0.85)
                r["match_strength_pct"] = max(int(r.get("match_strength_pct") or 0), 85)
                r["confidence"] = max(float(r.get("confidence") or 0.0), 0.75)
                audit["overrides"].append(
                    {"requirement_id": rid, "reason": "degree_detected", "new_classification": "match"}
                )

        if rid in years_req_ids and yrs is not None and yrs >= years_threshold:
            if r.get("classification") != "match":
                r["classification"] = "match"
                r["match_strength"] = max(float(r.get("match_strength") or 0.0), 0.85)
                r["match_strength_pct"] = max(int(r.get("match_strength_pct") or 0), 85)
                r["confidence"] = max(float(r.get("confidence") or 0.0), 0.70)
                audit["overrides"].append(
                    {"requirement_id": rid, "reason": f"years_detected({yrs})", "new_classification": "match"}
                )

    return gap_result, audit


def rebucket_gap_result(gap_result: dict) -> dict:
    all_results = gap_result.get("all_results") or []

    buckets = {"match": [], "partial": [], "gap": [], "signal_gap": []}
    for r in all_results:
        cls = r.get("classification", "signal_gap")
        buckets.get(cls, buckets["signal_gap"]).append(r)

    gap_result["matched_requirements"] = buckets["match"]
    gap_result["partial_gaps"] = buckets["partial"]
    gap_result["hard_gaps"] = buckets["gap"]
    gap_result["signal_gaps"] = buckets["signal_gap"]

    overall = gap_result.get("overall_alignment_score", 0)
    gap_result["summary"] = (
        f"Overall grounded alignment: {overall}/100. "
        f"Matches: {len(buckets['match'])}, Partial: {len(buckets['partial'])}, "
        f"Gaps: {len(buckets['gap'])}, Signal gaps: {len(buckets['signal_gap'])}."
    )
    return gap_result
