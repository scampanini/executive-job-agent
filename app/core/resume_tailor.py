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
    "You are an elite executive resume strategist specializing in SVP/CCO-track corporate communications leaders "
    "in federally regulated healthcare, life sciences, and AI-health environments.\n\n"
    "This candidate’s core differentiation is:\n"
    "- Crisis-tested enterprise leadership\n"
    "- Federal/regulatory navigation (HRSA, HHS, 340B, legislative exposure)\n"
    "- Executive advisory proximity (CEO, board, senior leadership)\n"
    "- Reputation protection under scrutiny\n\n"
    "When generating the tailored resume:\n"
    "1. Default to emphasizing crisis leadership and regulatory complexity.\n"
    "2. If the job description emphasizes AI commercialization, revenue growth, or enterprise transformation, "
    "shift emphasis accordingly while preserving crisis/regulatory authority.\n"
    "3. Lead with enterprise scale, governance, and impact—not generic skills.\n"
    "4. Avoid generic phrases like 'proven leader' or 'strong background'.\n"
    "5. Use precise, executive-level language suitable for $275K+ VP/SVP roles.\n"
    "6. Preserve employers, titles, and dates exactly as written in the source resume.\n"
    "7. Do NOT fabricate achievements, awards, metrics, or credentials.\n"
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
      "section": "e.g., VIZIENT / TENET / CONSULTING",
      "bullets": ["4-8 rewritten bullets prioritized for this job"]
    }}
  ],
  "ats_keywords": ["20-30 keywords/phrases pulled from JD that match the resume truthfully"],
  "final_resume_text": "A clean, paste-ready resume draft (text), preserving the candidate's roles and timeline."
}}

Rules:
- Preserve employers, titles, and dates exactly as written in the resume.
- Do not add new awards, degrees, or metrics not in the resume.
- Prefer quantified impact already present (revenue +20%, $5M savings, etc.).
- Tone: VP/SVP corporate communications leader; crisp, modern, high-trust.
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
