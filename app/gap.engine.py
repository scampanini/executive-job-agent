from typing import Any, Dict, List

from app.utils import safe_text, token_overlap_score, top_matching_lines


REQ_BUCKETS = {
    "board": ["board", "board materials", "committee", "governance", "qbr"],
    "c_suite": ["ceo", "president", "c-suite", "executive leadership", "evp", "svp"],
    "transformation": ["transformation", "turnaround", "reorg", "integration", "change management", "enterprise-wide"],
    "brand": ["brand", "reputation", "thought leadership", "positioning", "corporate visibility"],
    "media": ["media", "earned media", "press", "journalist", "spokesperson", "public relations"],
    "crisis": ["crisis", "issues management", "reputation risk", "litigation", "regulatory"],
    "internal": ["internal communications", "employee communications", "culture", "change communications"],
    "measurement": ["measure", "kpi", "metrics", "dashboard", "sentiment", "analytics"],
    "global": ["global", "worldwide", "international", "multinational"],
    "regulated": ["regulatory", "compliance", "policy", "government", "public affairs", "healthcare", "pharma"],
}


def extract_requirements(job_description: str) -> List[Dict[str, Any]]:
    jd = safe_text(job_description)
    lines = [x.strip("•- ").strip() for x in jd.splitlines() if x.strip()]
    reqs: List[Dict[str, Any]] = []

    seen = set()
    req_id = 1
    for line in lines:
        line_l = line.lower()
        matched_tags = [k for k, kws in REQ_BUCKETS.items() if any(kw in line_l for kw in kws)]
        if matched_tags or len(line.split()) >= 7:
            norm = line_l[:180]
            if norm in seen:
                continue
            seen.add(norm)
            reqs.append(
                {
                    "requirement_id": f"REQ-{req_id:03d}",
                    "text": line,
                    "tags": matched_tags,
                    "must_have": any(x in line_l for x in ["must", "required", "responsible", "accountable"]),
                }
            )
            req_id += 1

    return reqs[:25]


def match_requirement(requirement_text: str, candidate_text: str) -> Dict[str, Any]:
    evidence = top_matching_lines(candidate_text, requirement_text, limit=3)
    match_strength = 0.0
    if evidence:
        match_strength = max(token_overlap_score(x, requirement_text) for x in evidence)

    if match_strength >= 0.33:
        classification = "match"
    elif match_strength >= 0.16:
        classification = "partial"
    else:
        classification = "gap"

    return {
        "classification": classification,
        "match_strength": round(match_strength, 3),
        "match_strength_pct": int(round(match_strength * 100)),
        "evidence": evidence,
    }


def summarize_gap_results(matches: List[Dict[str, Any]]) -> str:
    hard_gaps = len([m for m in matches if m["classification"] == "gap" and m.get("must_have")])
    partials = len([m for m in matches if m["classification"] == "partial"])
    strong = len([m for m in matches if m["classification"] == "match"])

    return (
        f"Grounded analysis found {strong} strong matches, "
        f"{partials} partial matches, and {hard_gaps} must-have gaps."
    )


def run_grounded_gap_analysis(
    resume_text: str,
    job_description: str,
    portfolio_texts: List[str] | None = None,
) -> Dict[str, Any]:
    combined = safe_text(resume_text)
    portfolio_joined = "\n\n".join([safe_text(x) for x in (portfolio_texts or []) if safe_text(x)])
    if portfolio_joined:
        combined += "\n\n" + portfolio_joined

    requirements = extract_requirements(job_description)

    all_matches: List[Dict[str, Any]] = []
    hard_gaps: List[Dict[str, Any]] = []
    partial_gaps: List[Dict[str, Any]] = []
    strong_matches: List[Dict[str, Any]] = []

    for req in requirements:
        m = match_requirement(req["text"], combined)
        item = {
            "requirement_id": req["requirement_id"],
            "text": req["text"],
            "tags": req["tags"],
            "must_have": req["must_have"],
            **m,
        }
        all_matches.append(item)

        if item["classification"] == "gap":
            hard_gaps.append(item)
        elif item["classification"] == "partial":
            partial_gaps.append(item)
        else:
            strong_matches.append(item)

    total = max(len(all_matches), 1)
    score = round(((len(strong_matches) * 1.0) + (len(partial_gaps) * 0.5)) / total * 100)

    return {
        "overall_alignment_score": int(score),
        "summary": summarize_gap_results(all_matches),
        "requirements": all_matches,
        "hard_gaps": hard_gaps[:10],
        "partial_gaps": partial_gaps[:10],
        "strong_matches": strong_matches[:10],
    }
