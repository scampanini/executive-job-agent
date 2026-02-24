from typing import Dict, Any, List, Tuple

EXEC_COMPETENCIES = {
    "executive_comms": 2.0,
    "financial_comms": 2.0,
    "corporate_narrative": 1.6,
    "product_comms": 1.4,
    "regulated_healthcare": 1.2,
}

# General is noisy; keep low unless objective override already fixed it
GENERAL_MULT = 0.6

CLASS_SCORE = {
    "match": 1.00,
    "partial": 0.60,
    "gap": 0.00,
    "signal_gap": 0.25,  # signal gaps still show some adjacency
}

def executive_weighted_score_from_gap_result(
    gap_result: Dict[str, Any],
    *,
    max_abs_adjustment: int = 8,
    enable: bool = False
) -> Dict[str, Any]:
    """
    Deterministic. Uses grounded classifications + weights.
    When enable=False => adjustment=0 (no regression).
    """
    if not enable or not gap_result:
        return {"enabled": False, "adjustment": 0, "exec_weighted_score": None, "notes": ["exec weighting disabled"]}

    all_results = gap_result.get("all_results") or []
    if not isinstance(all_results, list) or not all_results:
        return {"enabled": True, "adjustment": 0, "exec_weighted_score": None, "notes": ["no all_results"]}

    total_w = 0.0
    earned = 0.0
    must_have_exec_gaps = 0
    signals: List[Dict[str, Any]] = []

    for r in all_results:
        comp = (r.get("competency") or "general").strip()
        cls = (r.get("classification") or "gap").strip()
        w = float(r.get("weight") or 1.0)
        must = bool(r.get("must_have"))

        comp_mult = EXEC_COMPETENCIES.get(comp, GENERAL_MULT)
        eff_w = w * comp_mult

        total_w += eff_w
        earned += eff_w * CLASS_SCORE.get(cls, 0.0)

        if must and cls == "gap" and comp in EXEC_COMPETENCIES:
            must_have_exec_gaps += 1

        if comp in EXEC_COMPETENCIES:
            signals.append({
                "requirement_id": r.get("requirement_id"),
                "competency": comp,
                "classification": cls,
                "weight": w,
                "eff_weight": round(eff_w, 2),
            })

    if total_w <= 0:
        return {"enabled": True, "adjustment": 0, "exec_weighted_score": None, "notes": ["total_w=0"]}

    exec_weighted_score = int(round((earned / total_w) * 100))

    # Adjustment: compare to grounded overall score, bounded
    base = int(gap_result.get("overall_alignment_score") or 0)
    raw_adj = exec_weighted_score - base

    # VP/SVP realism: if multiple must-have exec gaps remain, cap upside
    notes = []
    if must_have_exec_gaps >= 2 and raw_adj > 0:
        raw_adj = min(raw_adj, 2)
        notes.append("must-have exec gaps present; capped positive adjustment")

    # Bound
    adj = max(-max_abs_adjustment, min(max_abs_adjustment, raw_adj))
    if adj != raw_adj:
        notes.append(f"bounded adjustment to {adj}")

    return {
        "enabled": True,
        "exec_weighted_score": exec_weighted_score,
        "adjustment": int(adj),
        "base_grounded_score": base,
        "notes": notes,
        "exec_signals": signals[:30],  # keep payload small
    }
