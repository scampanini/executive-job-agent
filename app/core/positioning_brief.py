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

    "VOICE + TONE:\n"
    "- First person (I / my)\n"
    "- Direct, specific, recruiter-ready\n"
    "- Reads like an executive operator, not a consultant\n\n"

    "HARD BANS (do not use these phrases or patterns):\n"
    "- No 'In an era where', 'In today's world', 'increasingly', 'unprecedented'\n"
    "- No 'strategic acumen', 'value proposition', 'governance landscapes'\n"
    "- No 'seasoned leader', 'dynamic', 'proven track record', 'expert in'\n"
    "- No generic macro commentary\n\n"

    "MUST DO:\n"
    "1) Start with 1–2 sentences that are concrete and credible, referencing federal scrutiny "
    "and enterprise value protection.\n"
    "2) Use 2–3 specific proof points from the resume (e.g., Apexus / 340B / Tenet / Merck), "
    "without inventing anything.\n"
    "3) Keep the opening thesis to 120–160 words max.\n"
    "4) Preserve factual accuracy from the resume.\n"
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
