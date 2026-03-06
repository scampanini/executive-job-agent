import json
import os
from typing import Any, Dict, List, Tuple

from app.utils import clamp, count_keyword_hits, safe_text
from openai import OpenAI


EXEC_SIGNAL_KEYWORDS = {
    "board": ["board", "board materials", "committee", "governance", "qbr"],
    "c_suite": ["ceo", "president", "c-suite", "executive leadership", "chief executive", "evp", "svp"],
    "transformation": ["transformation", "turnaround", "reorg", "integration", "enterprise-wide", "change management"],
    "crisis": ["crisis", "issues management", "litigation", "regulatory", "reputation risk"],
    "global": ["global", "worldwide", "international", "multinational"],
    "media": ["media", "earned media", "press", "journalist", "spokesperson"],
    "brand": ["brand", "reputation", "thought leadership", "positioning", "visibility"],
    "regulated": ["regulatory", "compliance", "government", "public affairs", "healthcare", "pharma"],
}


def build_candidate_text(
    resume_text: str,
    portfolio_text: str = "",
    gap_answers_text: str = "",
) -> str:
    combined = safe_text(resume_text)

    if safe_text(portfolio_text):
        combined += "\n\n=== PORTFOLIO / EXPERIENCE EXAMPLES ===\n"
        combined += safe_text(portfolio_text)

    if safe_text(gap_answers_text):
        combined += "\n\n=== GAP ANSWERS ===\n"
        combined += safe_text(gap_answers_text)

    return combined


def executive_signal_scores(candidate_text: str, job_text: str) -> Dict[str, Any]:
    text = safe_text(candidate_text).lower()
    job_l = safe_text(job_text).lower()

    board = 0
    c_suite = 0
    transformation = 0
    crisis = 0

    board_hits = count_keyword_hits(text, EXEC_SIGNAL_KEYWORDS["board"])
    c_suite_hits = count_keyword_hits(text, EXEC_SIGNAL_KEYWORDS["c_suite"])
    transformation_hits = count_keyword_hits(text, EXEC_SIGNAL_KEYWORDS["transformation"])
    crisis_hits = count_keyword_hits(text, EXEC_SIGNAL_KEYWORDS["crisis"])

    if board_hits >= 2:
        board = 5
    elif board_hits == 1:
        board = 3

    if c_suite_hits >= 3:
        c_suite = 4
    elif c_suite_hits >= 1:
        c_suite = 2

    if transformation_hits >= 3:
        transformation = 4
    elif transformation_hits >= 1:
        transformation = 2

    if any(x in job_l for x in EXEC_SIGNAL_KEYWORDS["crisis"]):
        if crisis_hits >= 2:
            crisis = 2
        elif crisis_hits >= 1:
            crisis = 1

    total = board + c_suite + transformation + crisis
    return {
        "board": board,
        "c_suite": c_suite,
        "transformation": transformation,
        "crisis": crisis,
        "total": total,
    }


def requirement_fit_score(candidate_text: str, job_text: str) -> Dict[str, Any]:
    text = safe_text(candidate_text).lower()
    job_l = safe_text(job_text).lower()

    role_requirements = [
        "corporate communications",
        "media relations",
        "brand",
        "reputation",
        "executive communications",
        "issues management",
        "internal communications",
        "thought leadership",
        "product communications",
        "measurement",
        "stakeholders",
        "global",
    ]
    domain_requirements = [
        "healthcare",
        "pharma",
        "biotech",
        "regulated",
        "government",
        "policy",
        "regulatory",
        "public affairs",
    ]
    depth_requirements = [
        "strategy",
        "leadership",
        "cross-functional",
        "transformation",
        "crisis",
        "analytics",
        "earned media",
    ]

    role_hits = count_keyword_hits(text, [x for x in role_requirements if x in job_l or True])
    domain_hits = count_keyword_hits(text, domain_requirements)
    depth_hits = count_keyword_hits(text, depth_requirements)

    role_score = min(45, role_hits * 4)
    domain_score = min(20, domain_hits * 3)
    depth_score = min(20, depth_hits * 3)

    return {
        "role_requirements": role_score,
        "domain_fit": domain_score,
        "functional_depth": depth_score,
        "total": role_score + domain_score + depth_score,
    }


def risk_penalty(candidate_text: str, job_text: str) -> Dict[str, Any]:
    text = safe_text(candidate_text).lower()
    job_l = safe_text(job_text).lower()

    penalty = 0
    reasons: List[str] = []

    if "board" in job_l and "board" not in text:
        penalty -= 4
        reasons.append("Limited explicit board exposure found.")
    if ("crisis" in job_l or "issues management" in job_l) and "crisis" not in text and "issues management" not in text:
        penalty -= 4
        reasons.append("Limited explicit crisis / issues management evidence found.")
    if ("global" in job_l or "worldwide" in job_l) and "global" not in text and "international" not in text:
        penalty -= 3
        reasons.append("Limited explicit global scope evidence found.")
    if ("media relations" in job_l or "earned media" in job_l) and "media" not in text and "press" not in text:
        penalty -= 3
        reasons.append("Limited explicit media relations evidence found.")

    return {"total": max(-15, penalty), "reasons": reasons}


def heuristic_score_role(
    resume_text: str,
    job_text: str,
    min_base: int = 0,
    portfolio_text: str = "",
    gap_answers_text: str = "",
) -> Dict[str, Any]:
    candidate_text = build_candidate_text(resume_text, portfolio_text, gap_answers_text)

    base = requirement_fit_score(candidate_text, job_text)
    exec_signals = executive_signal_scores(candidate_text, job_text)
    penalties = risk_penalty(candidate_text, job_text)

    raw_score = base["total"] + exec_signals["total"] + penalties["total"]
    final_score = int(clamp(raw_score, 0, 100))

    if min_base and final_score < min_base:
        final_score = min_base

    if final_score >= 90:
        priority = "High"
    elif final_score >= 80:
        priority = "High"
    elif final_score >= 70:
        priority = "Medium"
    elif final_score >= 60:
        priority = "Medium"
    else:
        priority = "Low"

    why_this_fits = []
    if exec_signals["board"] > 0:
        why_this_fits.append("Evidence of board or governance-facing work.")
    if exec_signals["c_suite"] > 0:
        why_this_fits.append("Evidence of CEO / C-suite proximity.")
    if exec_signals["transformation"] > 0:
        why_this_fits.append("Evidence of transformation or enterprise change leadership.")
    if base["domain_fit"] >= 8:
        why_this_fits.append("Relevant domain / regulated-environment signals present.")
    if base["functional_depth"] >= 10:
        why_this_fits.append("Strong functional depth across communications disciplines.")

    risks_or_gaps = penalties["reasons"][:]
    if not risks_or_gaps and final_score < 85:
        risks_or_gaps.append("Some requirements appear implied rather than explicitly proven in the résumé.")

    top_resume_edits = [
        "Make board, CEO, and executive leadership support more explicit in top bullets.",
        "Quantify transformation scale, business impact, and stakeholder complexity.",
        "Surface crisis / reputation management examples closer to the top.",
    ]

    interview_leverage_points = [
        "Prepare a concise CEO/Board support story with cadence, stakes, and outcome.",
        "Prepare one transformation narrative showing scale, complexity, and measurable result.",
        "Prepare one crisis/issues-management example with stakeholder alignment and outcome.",
    ]

    two_line_pitch = (
        "Executive communications and reputation leader with cross-functional range, "
        "enterprise visibility, and executive stakeholder fluency. "
        "Best positioned when board proximity, transformation, and reputation risk experience are made explicit."
    )

    return {
        "score": final_score,
        "priority": priority,
        "subscores": {
            "base_fit": base,
            "executive_signals": exec_signals,
            "risk_penalty": penalties,
        },
        "why_this_fits": why_this_fits,
        "risks_or_gaps": risks_or_gaps,
        "top_resume_edits": top_resume_edits,
        "interview_leverage_points": interview_leverage_points,
        "two_line_pitch": two_line_pitch,
    }


def ai_score_role(
    resume_text: str,
    job_text: str,
    portfolio_text: str = "",
    gap_answers_text: str = "",
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)

    candidate_text = build_candidate_text(resume_text, portfolio_text, gap_answers_text)

    prompt = f"""
You are scoring an executive candidate against a job description.

Return strict JSON with keys:
score, priority, why_this_fits, risks_or_gaps, top_resume_edits, interview_leverage_points, two_line_pitch

Scoring guidance:
- 90-100 exceptional
- 80-89 very strong
- 70-79 viable with gaps
- 60-69 stretch
- below 60 weak fit

Focus on:
- board exposure
- CEO/C-suite support
- transformation scale
- crisis/issues management
- media/brand/reputation leadership
- regulated/global complexity

JOB DESCRIPTION:
{job_text}

CANDIDATE:
{candidate_text}
"""

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a careful executive recruiting analyst."},
            {"role": "user", "content": prompt},
        ],
    )

    content = resp.choices[0].message.content
    data = json.loads(content)

    return {
        "score": int(clamp(int(data.get("score", 0)), 0, 100)),
        "priority": safe_text(data.get("priority")) or "Medium",
        "why_this_fits": data.get("why_this_fits", []) or [],
        "risks_or_gaps": data.get("risks_or_gaps", []) or [],
        "top_resume_edits": data.get("top_resume_edits", []) or [],
        "interview_leverage_points": data.get("interview_leverage_points", []) or [],
        "two_line_pitch": safe_text(data.get("two_line_pitch")),
    }


def score_role(
    resume_text: str,
    job_text: str,
    use_ai: bool,
    min_base: int,
    portfolio_text: str = "",
    gap_answers_text: str = "",
) -> Tuple[Dict[str, Any], str]:
    try:
        if use_ai:
            try:
                result = ai_score_role(
                    resume_text=resume_text,
                    job_text=job_text,
                    portfolio_text=portfolio_text,
                    gap_answers_text=gap_answers_text,
                )

                heuristic = heuristic_score_role(
                    resume_text=resume_text,
                    job_text=job_text,
                    min_base=min_base,
                    portfolio_text=portfolio_text,
                    gap_answers_text=gap_answers_text,
                )

                # Blend AI with deterministic score for stability
                blended_score = int(round((result["score"] * 0.6) + (heuristic["score"] * 0.4)))
                result["score"] = int(clamp(blended_score, 0, 100))
                result["subscores"] = heuristic.get("subscores", {})
                return result, "openai+heuristic"
            except Exception:
                pass

        heuristic = heuristic_score_role(
            resume_text=resume_text,
            job_text=job_text,
            min_base=min_base,
            portfolio_text=portfolio_text,
            gap_answers_text=gap_answers_text,
        )
        return heuristic, "heuristic"

    except Exception as e:
        fallback = {
            "score": 0,
            "priority": "Low",
            "subscores": {},
            "why_this_fits": [],
            "risks_or_gaps": [f"Scoring failed: {e}"],
            "top_resume_edits": [],
            "interview_leverage_points": [],
            "two_line_pitch": "",
        }
        return fallback, "error"
