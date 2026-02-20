from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import re
from dotenv import load_dotenv

load_dotenv()

# Your personal weighting (from our convo)
WEIGHTS = {
    "mission_impact": 0.30,
    "brand_prestige": 0.20,
    "scope_authority": 0.20,
    "comp_alignment": 0.15,
    "cco_pathway": 0.10,
    "stability": 0.05,
}

KEYWORDS = {
    "healthcare": ["healthcare", "hospital", "payer", "provider", "340b", "clinical", "patient", "medicare", "medicaid"],
    "pharma_ls": ["pharma", "pharmaceutical", "biotech", "life sciences", "clinical trial", "fda", "drug", "therapeutic"],
    "ai_health": ["ai", "machine learning", "data platform", "precision health", "real-world data", "rwe", "analytics"],
    "comms_exec": ["head of communications", "svp", "vice president", "corporate communications", "corporate affairs",
                   "reputation", "media relations", "spokesperson", "executive communications", "issues management", "crisis"],
    "public_company": ["earnings", "investor relations", "10-k", "10q", "sec", "public company", "ipo"],
    "govt_policy": ["government affairs", "public policy", "regulatory", "legislation", "hrsa", "hhs", "washington"],
}

def _count_hits(text: str, words: List[str]) -> int:
    t = text.lower()
    return sum(1 for w in words if w.lower() in t)

def heuristic_score(resume_text: str, job_text: str, min_base_salary: int = 275000) -> Dict[str, Any]:
    jt = job_text.lower()
    rt = resume_text.lower()

    # Mission impact
    mission = 0.0
    mission += 0.35 if _count_hits(jt, KEYWORDS["healthcare"]) > 0 else 0.0
    mission += 0.35 if _count_hits(jt, KEYWORDS["pharma_ls"]) > 0 else 0.0
    mission += 0.30 if _count_hits(jt, KEYWORDS["ai_health"]) > 0 else 0.0
    mission = min(1.0, mission)

    # Brand prestige
    prestige = 0.0
    prestige += 0.5 if "alphabet" in jt or "fortune" in jt else 0.0
    prestige += 0.3 if "public" in jt or _count_hits(jt, KEYWORDS["public_company"]) > 0 else 0.0
    prestige += 0.2 if "subsidiary" in jt else 0.0
    prestige = min(1.0, prestige)

    # Scope / authority
    scope = 0.0
    scope += 0.5 if _count_hits(jt, KEYWORDS["comms_exec"]) > 1 else 0.0
    scope += 0.3 if ("team" in jt or "lead" in jt or "oversee" in jt) else 0.0
    scope += 0.2 if ("executive" in jt or "ceo" in jt) else 0.0
    scope = min(1.0, scope)

    # Compensation alignment: parse $275,000 - $375,000
    comp_alignment = 0.5
    m = re.search(r"\$?\s*([0-9]{2,3}(?:,\d{3})?)\s*[\-\–]\s*\$?\s*([0-9]{2,3}(?:,\d{3})?)", job_text)
    if m:
        lo = int(m.group(1).replace(",", ""))
        hi = int(m.group(2).replace(",", ""))
        if hi >= min_base_salary:
            comp_alignment = 1.0 if lo >= min_base_salary else 0.75
        else:
            comp_alignment = 0.25

    # CCO pathway
    cco_path = 0.0
    cco_path += 0.5 if ("head of" in jt or "svp" in jt) else 0.0
    cco_path += 0.3 if ("corporate affairs" in jt or "reputation" in jt) else 0.0
    cco_path += 0.2 if ("ceo" in jt or "board" in jt) else 0.0
    cco_path = min(1.0, cco_path)

    # Stability (light weight per your preference)
    stability = 0.5
    stability += 0.25 if ("public company" in jt or _count_hits(jt, KEYWORDS["public_company"]) > 0) else 0.0
    stability += 0.25 if ("subsidiary" in jt or "established" in jt) else 0.0
    stability = min(1.0, stability)

    # Resume match multiplier (basic)
    req_match = 0.0
    req_match += 0.35 if _count_hits(rt, KEYWORDS["healthcare"]) > 0 else 0.0
    req_match += 0.20 if _count_hits(rt, KEYWORDS["pharma_ls"]) > 0 else 0.0
    req_match += 0.20 if _count_hits(rt, KEYWORDS["public_company"]) > 0 else 0.0
    req_match += 0.25 if _count_hits(rt, KEYWORDS["comms_exec"]) > 1 else 0.0
    req_match = min(1.0, req_match)

    dims = {
        "mission_impact": mission,
        "brand_prestige": prestige,
        "scope_authority": scope,
        "comp_alignment": comp_alignment,
        "cco_pathway": cco_path,
        "stability": stability,
        "resume_match": req_match,
    }

    composite = sum(dims[k] * WEIGHTS[k] for k in WEIGHTS.keys())
    composite = composite * (0.85 + 0.15 * req_match)

    overall_0_100 = int(round(composite * 100))
    priority = "HIGH" if overall_0_100 >= 85 else "MEDIUM" if overall_0_100 >= 70 else "LOW"

    gaps = []
    if _count_hits(job_text, KEYWORDS["ai_health"]) and not _count_hits(resume_text, KEYWORDS["ai_health"]):
        gaps.append("Add 1–2 bullets translating data/technology narratives (AI/precision health).")
    if _count_hits(job_text, KEYWORDS["govt_policy"]) and not _count_hits(resume_text, KEYWORDS["govt_policy"]):
        gaps.append("Emphasize public policy / government affairs partnership experience (HRSA/HHS/340B).")

    strengths = []
    if _count_hits(resume_text, KEYWORDS["healthcare"]) > 0:
        strengths.append("Deep regulated healthcare leadership experience.")
    if "crisis" in rt:
        strengths.append("Crisis-tested executive communications leader.")
    if _count_hits(resume_text, KEYWORDS["public_company"]) > 0 or "earnings" in rt:
        strengths.append("Public-company narrative discipline (earnings/IR readiness).")

    return {
        "overall_score": overall_0_100,
        "priority": priority,
        "dimensions": dims,
        "strengths": strengths[:5],
        "gaps": gaps[:5],
        "recommended_angle": "Crisis-tested, mission-driven healthcare strategist who scales trust and growth in regulated environments.",
    }

def ai_score(resume_text: str, job_text: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    system = (
        "You are an executive recruiter and communications leader. "
        "Given a resume and a job description, produce a structured JSON evaluation. "
        "Use only provided text; no hallucinations."
    )

    user = f"""
RESUME:
{resume_text}

JOB DESCRIPTION:
{job_text}

Return JSON ONLY with this schema:
{{
  "overall_score": 0-100,
  "priority": "HIGH"|"MEDIUM"|"LOW",
  "why_this_fits": [3-6 bullets],
  "risks_or_gaps": [2-5 bullets],
  "top_resume_edits": [5-10 concrete edits, written as instructions],
  "interview_leverage_points": [5-8 bullets],
  "two_line_pitch": "string",
  "likely_reporting_relationships": ["CEO","CMO","GC","Corporate Affairs"],
  "notes": "string"
}}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()

    import json as _json
    try:
        return _json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return _json.loads(m.group(0))
        raise

# ==============================
# PHASE 3B – BLENDED SCORING
# ==============================

def _priority_from_score(s: int) -> str:
    return "HIGH" if s >= 85 else "MEDIUM" if s >= 70 else "LOW"


def _normalize_ai_to_common(ai: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ai_score() schema -> the same top-level keys heuristic_score returns
    so your UI can stay consistent.
    """
    overall = int(ai.get("overall_score", 0) or 0)
    overall = max(0, min(100, overall))

    strengths = ai.get("why_this_fits") or []
    gaps = ai.get("risks_or_gaps") or []

    if not isinstance(strengths, list):
        strengths = []
    if not isinstance(gaps, list):
        gaps = []

    recommended_angle = ai.get("two_line_pitch") or ""
    notes = ai.get("notes") or ""

    return {
        "overall_score": overall,
        "priority": ai.get("priority") or _priority_from_score(overall),
        "strengths": strengths[:5],
        "gaps": gaps[:5],
        "recommended_angle": recommended_angle,
        "notes": notes,
        # keep extra structured fields if you want them later in the UI
        "top_resume_edits": ai.get("top_resume_edits") or [],
        "interview_leverage_points": ai.get("interview_leverage_points") or [],
        "likely_reporting_relationships": ai.get("likely_reporting_relationships") or [],
    }


def blended_score(
    resume_text: str,
    job_text: str,
    min_base_salary: int = 275000,
    ai_gate: int = 55,
    blend_weight_ai: float = 0.50,
) -> Dict[str, Any]:
    """
    Stable blended scoring:
    - Always compute heuristic_score (cheap).
    - Only call ai_score if heuristic >= ai_gate (cost control).
    - Blend the two for a final overall_score.
    - Return the same schema your dashboard expects.
    """
    det = heuristic_score(resume_text, job_text, min_base_salary=min_base_salary)
    det_score = int(det.get("overall_score", 0) or 0)

    # Gate AI (batch cost control)
    if det_score < ai_gate:
        det["scoring_method"] = "heuristic_only"
        det["ai_gated_out"] = True
        return det

    ai = ai_score(resume_text, job_text)
    if not ai:
        det["scoring_method"] = "heuristic_fallback"
        det["ai_failed"] = True
        return det

    ai_norm = _normalize_ai_to_common(ai)
    ai_score_val = int(ai_norm.get("overall_score", 0) or 0)

    blended = int(round((1 - blend_weight_ai) * det_score + blend_weight_ai * ai_score_val))
    blended = max(0, min(100, blended))

    # Merge: keep deterministic dimensions, but promote blended top-level
    out = dict(det)
    out["overall_score"] = blended
    out["priority"] = _priority_from_score(blended)
    out["scoring_method"] = "blended"
    out["deterministic_score"] = det_score
    out["ai_score"] = ai_score_val

    # Prefer AI narrative if present; otherwise keep heuristic
    if ai_norm.get("strengths"):
        out["strengths"] = ai_norm["strengths"]
    if ai_norm.get("gaps"):
        out["gaps"] = ai_norm["gaps"]
    if ai_norm.get("recommended_angle"):
        out["recommended_angle"] = ai_norm["recommended_angle"]

    # Carry extras (optional)
    out["top_resume_edits"] = ai_norm.get("top_resume_edits", [])
    out["interview_leverage_points"] = ai_norm.get("interview_leverage_points", [])
    out["likely_reporting_relationships"] = ai_norm.get("likely_reporting_relationships", [])
    out["notes"] = ai_norm.get("notes", "")

    return out
