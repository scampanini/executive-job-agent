import sys
from pathlib import Path

# Ensure repo root is on the Python path so `import app...` works in Streamlit/Render
ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import streamlit as st
import os

from app.core.resume_parse import load_resume
from app.core.storage import (
    init_db,
    save_resume,
    save_job,
    save_score,
    list_recent_scores,
    create_pipeline_item,
    update_pipeline_item,
    list_pipeline_items,
)

from app.core.scoring import heuristic_score, ai_score 
from app.core.resume_tailor import tailor_resume_ai
from app.core.positioning_brief import generate_positioning_brief
from app.core.recruiter_outreach import generate_recruiter_outreach

st.set_page_config(page_title="Executive Job Agent", layout="wide")

st.title("Executive Job Agent (Personal)")
st.caption("Upload your résumé + paste a job description → fit score, positioning, and stored history. (Compliant: no scraping, no auto-applying.)")

init_db()

with st.sidebar:
    st.header("Settings")
    min_base = st.number_input("Minimum base salary ($)", min_value=0, value=275000, step=5000)
    use_ai = st.toggle("Use AI (requires OPENAI_API_KEY in environment)", value=False)

    st.divider()
    st.subheader("AI status")
    st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("OPENAI_MODEL:", os.getenv("OPENAI_MODEL", "(not set)"))


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
tailor = st.button("Generate tailored résumé")
brief = st.button("Generate positioning brief (1-page)")

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
    st.session_state["last_job_id"] = job_id

    result = None
        # Save latest texts for resume tailoring
    st.session_state["last_resume_text"] = resume.raw_text
    st.session_state["last_job_text"] = job_desc
    st.session_state["last_company"] = company or ""
    st.session_state["last_title"] = title or ""

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

st.divider()
st.subheader("Tailored résumé (AI)")

if tailor:
    resume_text = st.session_state.get("last_resume_text")
    job_text = st.session_state.get("last_job_text")

    if not resume_text or not job_text:
        st.error("First, score a role so the app has your latest résumé + job description.")
    else:
        with st.spinner("Generating tailored résumé..."):
            tailored = tailor_resume_ai(resume_text, job_text)

        if not tailored:
            st.error("AI tailoring is not available. Confirm OPENAI_API_KEY is set in Render Environment.")
        else:
            st.success("Tailored résumé generated.")

            st.markdown("**Tailored headline**")
            st.write(tailored.get("tailored_headline", ""))

            st.markdown("**Tailored executive summary**")
            for b in tailored.get("tailored_summary", []) or []:
                st.write(f"- {b}")

            st.markdown("**Core competencies**")
            comps = tailored.get("core_competencies", []) or []
            if comps:
                st.write(", ".join(comps))

            st.markdown("**Top rewrite instructions**")
            for b in tailored.get("rewrite_instructions", []) or []:
                st.write(f"- {b}")

            st.markdown("**Tailored bullets by section**")
            for section in tailored.get("tailored_bullets", []) or []:
                st.write(f"**{section.get('section','')}**")
                for b in section.get("bullets", []) or []:
                    st.write(f"- {b}")

            st.markdown("**ATS keywords**")
            kws = tailored.get("ats_keywords", []) or []
            if kws:
                st.write(", ".join(kws))

            st.markdown("**Paste-ready tailored résumé draft**")
            final_text = tailored.get("final_resume_text", "")
            st.text_area("Tailored résumé text", value=final_text, height=400)

            filename = "tailored_resume.txt"
            if st.session_state.get("last_company") or st.session_state.get("last_title"):
                safe = f"{st.session_state.get('last_title','').strip()}_{st.session_state.get('last_company','').strip()}"
                safe = "_".join(safe.split())
                filename = f"tailored_resume_{safe[:60]}.txt"

st.divider()
st.subheader("Executive Positioning Brief (1-page)")

if brief:
    resume_text = st.session_state.get("last_resume_text")
    job_text = st.session_state.get("last_job_text")

    if not resume_text or not job_text:
        st.error("First, score a role so the app has your latest résumé + job description.")
    else:
        with st.spinner("Generating positioning brief..."):
            memo = generate_positioning_brief(resume_text, job_text)

        if memo:
            st.text_area("Positioning brief", value=memo, height=450)

            st.download_button(
                "Download positioning brief (TXT)",
                data=memo.encode("utf-8"),
                file_name="positioning_brief.txt",
                mime="text/plain",
            )

st.divider()
st.subheader("Pipeline Tracker")

PIPELINE_STAGES = [
    "Interested",
    "Applied",
    "Recruiter outreach",
    "Recruiter screen",
    "Hiring manager screen",
    "Interview loop",
    "Finalist",
    "Offer",
    "Rejected",
    "Withdrawn",
]

with st.expander("Add current role to pipeline", expanded=False):
    st.caption("Tip: Score a role first so company/title are captured, then add it to your pipeline.")
    default_stage = "Interested"
    stage = st.selectbox("Stage", PIPELINE_STAGES, index=PIPELINE_STAGES.index(default_stage))
    next_action = st.text_input("Next action date (YYYY-MM-DD)", value="")
    notes = st.text_area("Notes", height=120)

    add_to_pipeline = st.button("Add to pipeline")

if add_to_pipeline:
    job_id = st.session_state.get("last_job_id")

    if not job_id:
        st.error("Missing last job reference. Score a role first, then add it to the pipeline.")
        return

    score_data = st.session_state.get("last_score_result", {})
    fit_score = score_data.get("total_score")
    priority = score_data.get("priority")

    create_pipeline_item(
        job_id=job_id,
        stage=stage,
        next_action_date=next_action or None,
        notes=notes or None,
        fit_score=fit_score,
        priority=priority,
    )
    st.success("Added to pipeline.")

st.session_state["last_job_id"] = job_id
st.session_state["last_score_result"] = score_result

if not job_id:
    st.error("Missing last job reference. Score a role again, then click 'Add to pipeline'.")
else:
    score_data = st.session_state.get("last_score_result", {})
    fit_score = score_data.get("total_score")
    priority = score_data.get("priority")

    create_pipeline_item(
        job_id=job_id,
        stage=stage,
        next_action_date=next_action or None,
        notes=notes or None,
        fit_score=fit_score,
        priority=priority,
    )
    st.success("Added to pipeline.")

    create_pipeline_item(
        job_id=job_id,
        stage=stage,
        next_action_date=next_action or None,
        notes=notes or None,
        fit_score=fit_score,
        priority=priority,
)
st.success("Added to pipeline.")

st.markdown("### Active roles")
items = list_pipeline_items(active_only=True, limit=50)

if not items:
    st.info("No active pipeline items yet.")
else:
    for it in items:
        title = it.get("title") or "—"
        company = it.get("company") or "—"
        loc = it.get("location") or ""
        url = it.get("url") or ""
        header = f"{title} @ {company}" + (f" ({loc})" if loc else "")
        st.markdown(f"**{header}**")
        if url:
            st.write(url)
        st.write(f"Stage: **{it['stage']}**")
        if it.get("next_action_date"):
            st.write(f"Next action: **{it['next_action_date']}**")
        if it.get("notes"):
            st.write(it["notes"])

        with st.expander("Update this pipeline item", expanded=False):
            new_stage = st.selectbox(
                "New stage",
                PIPELINE_STAGES,
                index=PIPELINE_STAGES.index(it["stage"]) if it["stage"] in PIPELINE_STAGES else 0,
                key=f"stage_{it['pipeline_id']}",
            )
            new_next = st.text_input(
                "Next action date (YYYY-MM-DD)",
                value=it.get("next_action_date") or "",
                key=f"next_{it['pipeline_id']}",
            )
            new_notes = st.text_area(
                "Notes",
                value=it.get("notes") or "",
                height=120,
                key=f"notes_{it['pipeline_id']}",
            )
            deactivate = st.checkbox("Mark inactive (closed)", value=False, key=f"closed_{it['pipeline_id']}")

            if st.button("Save update", key=f"save_{it['pipeline_id']}"):
                update_pipeline_item(
                    pipeline_id=it["pipeline_id"],
                    stage=new_stage,
                    next_action_date=new_next or None,
                    notes=new_notes or None,
                    is_active=not deactivate,
                )
                st.success("Updated. Refreshing list...")
                st.rerun()

st.divider()
st.subheader("Recruiter Outreach Kit")

outreach = st.button("Generate recruiter outreach kit")

if outreach:
    resume_text = st.session_state.get("last_resume_text")
    job_text = st.session_state.get("last_job_text")

    if not resume_text or not job_text:
        st.error("First, score a role so the app has your latest résumé + job description.")
    else:
        with st.spinner("Generating outreach kit..."):
            kit = generate_recruiter_outreach(resume_text, job_text)

        if not kit:
            st.error("Outreach kit is not available. Confirm OPENAI_API_KEY is set in Render Environment.")
        else:
            st.success("Outreach kit generated.")

            st.markdown("### Recruiter email")
            email_text = kit.get("email", "")
            st.text_area("Email", value=email_text, height=180)

            st.markdown("### LinkedIn message")
            li_text = kit.get("linkedin", "")
            st.text_area("LinkedIn", value=li_text, height=120)

            st.markdown("### First-call talking points")
            call_text = kit.get("call_talking_points", "")
            st.text_area("Call talking points", value=call_text, height=160)

            st.download_button(
                "Download outreach kit (TXT)",
                data=(
                    "RECRUITER EMAIL\n\n"
                    + email_text
                    + "\n\nLINKEDIN MESSAGE\n\n"
                    + li_text
                    + "\n\nFIRST-CALL TALKING POINTS\n\n"
                    + call_text
                ).encode("utf-8"),
                file_name="recruiter_outreach_kit.txt",
                mime="text/plain",
            )
