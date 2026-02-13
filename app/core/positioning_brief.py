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
        "HARD BANS:\n"
        "- No 'In an era where', 'in today's world', 'unprecedented'\n"
        "- No consultant language or biography tone\n\n"
        "MUST DO:\n"
        "1) Start with concrete enterprise-value language under federal scrutiny.\n"
        "2) Reference real proof points from the resume only.\n"
        "3) Write decisively and concisely.\n"
    )

    user = (
        "RESUME:\n"
        + resume_text
        + "\n\nJOB DESCRIPTION:\n"
        + job_text
        + "\n\n"
        "Write a 1-page Executive Positioning Brief with these sections:\n"
        "1. Strategic Opening Thesis\n"
        "2. Enterprise Risk & Regulatory Authority\n"
        "3. Transformation & Growth Contribution\n"
        "4. Why This Organization / Why Now\n"
        "5. Forward-Looking Impact Statement\n\n"
        "Write in polished executive prose. No bullet points."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
