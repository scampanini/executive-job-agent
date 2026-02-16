from typing import Optional, Dict
import os


def generate_recruiter_outreach(
    resume_text: str,
    job_text: str,
    company: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    system = (
        "You are generating recruiter-facing outreach for an SVP/CCO-track corporate communications leader.\n\n"
        "Tone requirements:\n"
        "- First person\n"
        "- Concise, authoritative, non-salesy\n"
        "- Sounds like a senior executive, not a job seeker\n\n"
        "Hard rules:\n"
        "- No 'excited to apply'\n"
        "- No fluff or buzzwords\n"
        "- No resume summary repetition\n\n"
        "Goal:\n"
        "Help the recruiter immediately see placement potential."
    )

    user = (
        "RESUME:\n"
        + resume_text
        + "\n\nJOB DESCRIPTION:\n"
        + job_text
        + "\n\n"
        "Generate three items:\n"
        "1) Recruiter intro email (5–6 sentences)\n"
        "2) LinkedIn outreach message (2–3 sentences)\n"
        "3) First-call positioning talk track (5 bullet points)\n\n"
        "Output as JSON with keys: email, linkedin, call_talking_points.\n"
        "Keep it tight and executive-level."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    return eval(response.choices[0].message.content.strip())

