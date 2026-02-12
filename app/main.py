import sys
from pathlib import Path

# Ensure repo root is on the Python path so `import app...` works in Streamlit/Render
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from pathlib import Path
from app.core.resume_parse import load_resume
from app.core.storage import init_db, save_resume, save_job, save_score, list_recent_scores
from app.core.scoring import heuristic_score, ai_score

st.set_page_config(page_title="Executive Job Agent", layout="wide")

st.title("Executive Job Agent (Personal)")
st.caption("Upload your résumé + paste a job description → fit score, positioning, and stored history. (Compliant: no scraping, no auto-applying.)")

init_db()

with st.sidebar:
    st.header("Settings")
    min_base = st.number_input("Minimum base salary ($)", min_value=0, value=275000, step=5000)
    use_ai = st.toggle("Use AI (requires OPENAI_API_KEY in environment)", value=False)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Résumé")
    docx = st.file_uploader("Upload résumé (DOCX)", type=["docx"])
    pasted_resume = st.text_area("…or paste résumé text", height=200)

with col2:
    st.subheader("2) Job")
    company = st.text_input("Company (optional)")
    title = st.text_input("Title (optional)")
    location = st.text_input("Location (optional)")
    url = st.text_input("Job URL (optional)")
    job_desc = st.text_area("Paste the full job description", height=320)

st.divider()
run = st.button("Score this role", type="primary")

def _save_uploaded(docx_file) -> str:
    tmp_dir = Path.cwd() / ".tmp_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / docx_file.name
    out_path.write_bytes(docx_file.getvalue())
    return str(out_path)

if run:
    if not job_desc.strip():
        st.error("Please paste a job description.")
        st.stop()

    docx_path = _save_uploaded(docx) if docx is not None else None

    try:
        resume = load_resume(docx_path, pasted_resume)
    except Exception as e:
        st.error(f"Résumé error: {e}")
        st.stop()

    resume_id = save_resume(resume.source, resume.raw_text)
    job_id = save_job(job_desc, company=company or None, title=title or None, location=location or None, url=url or None)

    result = None
    model = None
    if use_ai:
        try:
            result = ai_score(resume.raw_text, job_desc)
            if result is not None:
                model = "openai"
        except Exception as e:
            st.warning(f"AI scoring failed; falling back to heuristic scoring. Details: {e}")

    if result is None:
        result = heuristic_score(resume.raw_text, job_desc, min_base_salary=int(min_base))
        model = "heuristic"

    save_score(job_id, resume_id, result, model=model)

    st.success("Scored and saved.")
    left, right = st.columns([1, 1])

    with left:
        st.subheader(f"Overall score: {result.get('overall_score', '—')}/100")
        st.write(f"Priority: **{result.get('priority', '—')}**")
        dims = result.get("dimensions")
        if isinstance(dims, dict):
            st.markdown("**Dimensions**")
            st.json(dims)
        if result.get("two_line_pitch"):
            st.markdown("**Two-line pitch**")
            st.write(result["two_line_pitch"])
        elif result.get("recommended_angle"):
            st.markdown("**Recommended positioning angle**")
            st.write(result["recommended_angle"])

    with right:
        for label, key in [
            ("Why this fits", "why_this_fits"),
            ("Strengths", "strengths"),
            ("Risks / gaps", "risks_or_gaps"),
            ("Gaps (tactical fixes)", "gaps"),
            ("Top résumé edits", "top_resume_edits"),
            ("Interview leverage points", "interview_leverage_points"),
        ]:
            items = result.get(key)
            if items:
                st.markdown(f"**{label}**")
                for b in items:
                    st.write(f"- {b}")

st.divider()
st.subheader("Recent scored roles")

scores = list_recent_scores(limit=15)
if not scores:
    st.info("No scored roles yet. Score your first job above.")
else:
    import datetime as dt
    for s in scores:
        ts = dt.datetime.fromtimestamp(s["created_at"]).strftime("%Y-%m-%d %H:%M")
        company = s.get("company") or "—"
        title = s.get("title") or "—"
        loc = s.get("location") or ""
        overall = s["result"].get("overall_score", "—")
        st.markdown(f"**{overall}/100** — {title} @ {company} {('(' + loc + ')') if loc else ''}  \n_{ts}_")

