from typing import Optional
import os


def generate_positioning_brief(resume_text: str, job_text: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=api_key)

    LOCKED_OPENING = (
        "I lead corporate communications as an enterprise growth and risk function using reputation, "
        "narrative, and governance alignment to generate revenue, protect brand value, and sustain trust "
        "in highly regulated healthcare environments. For more than 20 years I have operated as a trusted "
        "C-suite advisor, helping organizations translate communications strategy into measurable business "
        "outcomes, including a 20% increase in revenue and $5 million in operational cost reductions. "
        "I sit at the intersection of regulatory complexity, corporate reputation, and business strategy.\n\n"
    )

    system = (
        "You are drafting a recruiter-facing Executive Positioning Brief for an SVP/CCO-track "
        "corporate communications leader in federally regulated healthcare.\n\n"
        "Write ONLY sections 2–5. Do NOT write or modify the opening paragraph.\n"
        "Tone: decisive, enterprise-scale, recruiter-ready.\n"
    )

    user = (
        "RESUME:\n"
        + resume_text
        + "\n\nJOB DESCRIPTION:\n"
        + job_text
        + "\n\n"
        "Write ONLY the following sections:\n"
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

    return LOCKED_OPENING + response.choices[0].message.content.strip()
