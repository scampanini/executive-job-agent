from typing import Any, Dict, List


def _take_quotes(items: List[Dict[str, Any]], max_items: int = 6) -> List[str]:
    quotes: List[str] = []
    for it in items:
        ev = it.get("evidence") or []
        if not ev:
            continue
        q = (ev[0].get("quote") or "").strip()
        if q:
            quotes.append(q.replace("\n", " ").strip())
        if len(quotes) >= max_items:
            break
    return quotes


def build_grounded_positioning_brief(header: str, gap_result: Dict[str, Any]) -> str:
    score = gap_result.get("overall_alignment_score", 0)
    matches = gap_result.get("matched_requirements", []) or []
    partial = gap_result.get("partial_gaps", []) or []
    hard = gap_result.get("hard_gaps", []) or []
    signal = gap_result.get("signal_gaps", []) or []

    proof_points = _take_quotes(matches, max_items=6)
    biggest_gaps = [g.get("text", "") for g in (hard[:4] + partial[:2]) if g.get("text")]

    themes: List[str] = []
    for it in matches[:10]:
        comp = it.get("competency")
        if comp and comp not in themes:
            themes.append(comp)
    themes = themes[:6]

    lines: List[str] = []
    lines.append("# Grounded Positioning Brief")
    lines.append("")
    lines.append(f"**Role:** {header}")
    lines.append(f"**Grounded alignment score:** {score}/100")
    lines.append("")

    lines.append("## Positioning themes (grounded)")
    if themes:
        for t in themes:
            lines.append(f"- {t}")
    else:
        lines.append("- (No strong themes detected yet — add portfolio artifacts and re-run.)")
    lines.append("")

    lines.append("## Proof points (quoted)")
    if proof_points:
        for q in proof_points:
            lines.append(f"- “{q}”")
    else:
        lines.append("- (No quoted proof points found yet.)")
    lines.append("")

    lines.append("## Biggest gaps to close (grounded)")
    if biggest_gaps:
        for g in biggest_gaps:
            lines.append(f"- {g}")
    else:
        lines.append("- (No major gaps detected.)")
    lines.append("")

    lines.append("## 7-day action plan")
    lines.append("- Add 2–3 portfolio artifacts: press release, executive speech, crisis summary, policy narrative, measurement framework example.")
    lines.append("- Rewrite 3–5 résumé bullets to explicitly cover the top 2 gaps (stakeholders + scope + outcomes).")
    lines.append("- Prep 2 executive stories: (1) crisis/issues, (2) corporate narrative + thought leadership; each with outcomes.")
    lines.append("- Draft a 5-sentence narrative aligned to the JD: vision → credibility → differentiation → proof → call-to-action.")
    lines.append("")

    if signal:
        lines.append("## Missing executive signals (likely)")
        for s in signal[:6]:
            txt = s.get("text")
            if txt:
                lines.append(f"- {txt}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
