from __future__ import annotations
from typing import Dict, Any, Optional
import os
import re


def tailor_resume_ai(resume_text: str, job_text: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    system = (
        "You are an elite executive resume strategist specializing in CCO-track and SVP-level "
        "corporate communications leaders operating in federally regulated healthcare environments.\n\n"
        "This candidate's market positioning:\n"
        "- Crisis-tested enterprise healthcare leader\n"
        "- Operates under federal oversight and regulatory scrutiny (HRSA, HHS, 340B, legislative exposure)\n"
        "- Advises CEO, board, and executive leadership\n"
        "- Protects enterprise reputation under high-stakes conditions\n"
        "- Aligns corporate affairs with commercialization and enterprise strategy\n\n"
        "MANDATORY WRITING RULES:\n"
        "1. Write with authority, not aspiration.\n"
        "2. Do NOT use weak phrases like 'dynamic', 'proven', 'skilled in', 'expert in', "
        "'strong background', or similar generic language.\n"
        "3. Lead with enterprise impact, governance proximity, and regulatory complexity.\n"
        "4. Emphasize crisis leadership and federal exposure unless the JD strongly shifts toward AI commercialization.\n"
        "5. Use concise, executive-level language suitable for $275K+ SVP roles.\n"
        "6. Preserve employers, titles, and dates exactly as written in the resume.\n"
        "7. Do NOT fabricate achievements, awards, metrics, or credentials.\n"
        "8. Prefer outcome-driven bullets over competency statements.\n\n"
        "Tone: board-ready, decisive, enterprise-scale.\n"
    )

    user = f"""
RESUME (SOURCE OF TRUTH):
{resume_text}

JOB DESCRIPTION:
{job_text}

Return JSON ONLY with this schema:
{{
  "tailored_headline": "one-line headline for the top of the resume",
  "tailored_summary": ["3-5 bullets, executive-level, specific to this role"],
  "core_competencies": ["12-16 skills/competencies, keyword-aligned, not fluff"],
  "rewrite_instructions": ["5-10 very concrete edits to apply to the resume"],
  "tailored_bullets": [
    {{
      "section": "e.g., TENET / VIZIENT / MERCK",
      "bullets": ["4-8 rewritten bullets prioritized for this job"]
    }}
  ],
  "ats_keywords": ["20-30 keywords/phrases pulled from JD that match the resume truthfully"],
  "final_resume_text": "A clean, paste-ready resume draft (text), preserving the candidate's roles and timeline."
}}

Rules:
- Preserve employers, titles, and dates exactly as written in the resume.
- Do not add new awards, degrees, or metrics not in the resume.
- Prefer quantified impact already present (revenue, savings, growth).
- Tone: SVP corporate communications / corporate affairs leader; crisp and high-trust.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )

    text = resp.choices[0].message.content.strip()

    import json
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise
