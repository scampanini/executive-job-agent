from __future__ import annotations
from typing import Dict, Any, Optional, List
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
