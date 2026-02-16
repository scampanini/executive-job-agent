from typing import Optional, Dict
import os
import json
import re


def generate_recruiter_outreach(
    resume_text: str,
    job_text: str,
) -> Optional[Dict[str, str]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    system = (
        "You generate recruiter-facing outreach for an SVP/CCO-track corporate communications leader.\n"
        "Tone: first person, concise, authoritative, non-salesy.\n"
        "Hard bans: 'excited to apply', fluff, buzzwords.\n"
        "Return JSON ONLY."
    )

    user = (
        "RESUME:\n" + resume_text
        + "\n\nJOB DESCRIPTION:\n" + job_text
        + "\n\n"
        "Generate three items and return JSON ONLY with keys:\n"
        "email: recruiter intro email (5–6 sentences)\n"
        "linkedin: LinkedIn outreach message (2–3 sentences)\n"
        "call_talking_points: first-call positioning talk track (5 bullet points as a single string)\n"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )

    text = response.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise
