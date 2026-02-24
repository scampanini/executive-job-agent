import re
from typing import Dict, Any, Tuple, Optional

DEGREE_REGEX = re.compile(
    r"\b(bachelor|b\.?s\.?|b\.?a\.?)\b", re.I
)

# Very simple year extraction; you can improve later
YEAR_REGEX = re.compile(r"\b(19[7-9]\d|20[0-2]\d)\b")

def infer_has_bachelors(resume_text: str) -> bool:
    txt = resume_text or ""
    # Accept BS/BA or the word Bachelor
    return bool(DEGREE_REGEX.search(txt)) and ("university" in txt.lower() or "college" in txt.lower())

def infer_years_experience(resume_text: str) -> Optional[int]:
    """
    Deterministic estimate: uses earliest and latest year found.
    Conservative: if only one year found -> None.
    """
    txt = resume_text or ""
    years = [int(y) for y in YEAR_REGEX.findall(txt)]
    if len(years) < 2:
        return None
    return max(years) - min(years)

def apply_objective_overrides(
    gap_result: Dict[str, Any],
    resume_text: str,
    *,
    degree_req_ids=("REQ-009",),
    years_req_ids=("REQ-010",),
    years_threshold=15
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Mutates classification + score contribution for objective requirements.
    Returns (patched_gap_result, audit_log)
    """
    audit = {"overrides": []}
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
                audit["overrides"].append({"requirement_id": rid, "reason": "degree_detected", "new_classification": "match"})

        if rid in years_req_ids and yrs is not None and yrs >= years_threshold:
            if r.get("classification") != "match":
                r["classification"] = "match"
                r["match_strength"] = max(float(r.get("match_strength") or 0.0), 0.85)
                r["match_strength_pct"] = max(int(r.get("match_strength_pct") or 0), 85)
                r["confidence"] = max(float(r.get("confidence") or 0.0), 0.70)
                audit["overrides"].append({"requirement_id": rid, "reason": f"years_detected({yrs})", "new_classification": "match"})

    # IMPORTANT: you must recompute summary buckets after patch
    return gap_result, audit
