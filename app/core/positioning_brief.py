from __future__ import annotations
from typing import Optional
import os


def generate_positioning_brief(resume_text: str, job_text: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    system = (
        "You are drafting a recruiter-facing Executive Positioning Brief for an SVP/CCO-track "
        "corporate communications leader in federally regulated healthcare.\n\n"

        "This candidate:\n"
        "- Protects enterprise value under federal scrutiny\n"
        "- Advises CEOs and executive leadership\n"
        "- Operates in HRSA/HHS/340B and federally exposed markets\n"
        "- Aligns corporate affairs with commercialization and transformation strategy\n\n"

        "MANDATORY RULES:\n"
        "1. Write in FIRST PERSON (I / my), not third person.\n"
        "2. Do NOT use biography tone (no third-person references).\n"
        "3. Do NOT begin with generic industry commentary.\n"
        "4. The first paragraph must reference enterprise value, governance, or federal scrutiny.\n"
        "5. Tone: decisive, enterprise-scale, recruiter-ready.\n"
        "6. No flattery. No 'excited to apply.'\n"
        "7. Preserve factual accuracy from the resume.\n"
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

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
