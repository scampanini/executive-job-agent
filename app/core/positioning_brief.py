from __future__ import annotations
from typing import Dict, Any, Optional
import os
import re


def generate_positioning_brief(resume_text: str, job_text: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    system = (
        "You are an elite executive strategist drafting a 1-page positioning brief "
        "for an SVP/CCO-track corporate communications leader.\n\n"
        "This candidate:\n"
        "- Protects enterprise value under federal and regulatory scrutiny\n"
        "- Advises CEOs and boards in high-stakes healthcare environments\n"
        "- Operates in HRSA/HHS/340B and federally exposed markets\n"
        "- Aligns corporate affairs with commercialization and transformation strategy\n\n"
        "The output must read like a strategic memo, NOT a cover letter.\n"
        "Tone: Board-ready, decisive, enterprise-scale, forward-looking.\n"
        "Avoid generic phrases like 'excited to apply' or 'I am writing to express interest.'\n"
        "Open with a clear strategic thesis about enterprise value protection + growth acceleration.\n"
        "Preserve factual integrity from the resume. Do not fabricate achievements."
    )

    user = f"""
RESUME:
{resume_text}

JOB DESCRIPTION:
{job_text}

Write a 1-page Executive Positioning Brief structured as:

1. Strategic Opening Thesis (2-3 paragraphs)
2. Enterprise Risk & Regulatory Authority
3. Transformation & Growth Contribution
4. Why This Organization / Why Now
5. Forward-Looking Impact Statement

Write in polished executive prose. No bullet points.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()
