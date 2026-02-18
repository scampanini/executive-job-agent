import sys
from pathlib import Path

# Ensure repo root is on the Python path so `import app...` works in Streamlit/Render
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import tempfile
from datetime import date, datetime

import streamlit as st

from app.core.resume_parse import load_resume
from app.core.scoring import heuristic_score, ai_score
from app.core.resume_tailor import tailor_resume_ai
from app.core.positioning_brief import generate_positioning_brief
from app.core.recruiter_outreach import generate_recruiter_outreach
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


def safe_text(x) -> str:
    return "" if x is None else str(x)


def _call_scorer(fn, resume_text: str, job_text: str, min_base: int):
    # Try a few possible signatures to stay compatible with your scoring.py
    for args, kwargs in [
        ((resume_text, job_text), {"min_base": min_base}),
        ((resume_text, job_text), {"min_salary": min_base}),
        ((resume_text, job_text, min_base), {}),
        ((resume_text, job_text), {}),
    ]:
        try:
            return fn(*args, **kwargs)
        except TypeError:
            continue
    return fn(resume_text, job_text)


def score_role(resume_text: str, job_text: str, use_ai: bool, min_base: int):
    if use_ai:
        try:
            return _call_scorer(ai_score, resume_text, job_text, min_base), "openai"
        except Exception as e:
            return {
                "error": f"AI scoring failed; falling back to heuristic scoring. Details: {e}",
                **_call_scorer(heuristic_score, resume_text, job_text, min_base),
            }, "heuristic"
    return _call_scorer(heuristic_score, resume_text, job_text, min_base), "heuristic"


def parse_yyyy_mm_dd(s: str):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


st.set_page_config(page_title="Executive Job Agent (Personal)", layout="wide")
init_db()

st.title("Executive Job Agent (Personal)")

with st.sidebar:
    st.header("Settings")
    min_base = st.number_input("Minimum base salary ($)", min_value=0, value=275000, step=5000)
    use_ai = st.toggle("Use AI for scoring", value=True)

    st.divider()
    st.subheader("AI status")
    st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("OPENAI_MODEL:", os.getenv("OPENAI_MODEL", "(not set)"))


col_l, col_r = st.columns(2)

with col_l:
    st.subheader("1) Upload your résumé")
    uploaded = st.file_uploader("Upload résumé (PDF/DOCX)", type=["pdf", "docx"])

    resume_text = ""
    resume_source = ""

    if uploaded:
        suffix = ".pdf" if uploaded.type == "application/pdf" else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        try:
            # load_resume expects (file_path, pasted_text)
            resume = load_resume(tmp_path, None)
            resume_text = (
                getattr(resume, "raw_text", None)
                or getattr(resume, "text", None)
                or str(resume)
            )
            resume_source = uploaded.name
            st.success(f"Loaded résumé: {resume_source}")
        except Exception as e:
            st.error(f"Could not parse résumé: {e}")
            resume_text = ""
            resume_source = ""

with col_r:
    st.subheader("2) Paste job description")
    company = st.text_input("Company (optional)", value="")
    title = st.text_input("Title (optional)", value="")
    location = st.text_input("Location (optional)", value="")
    url = st.text_input("Job URL (optional)", value="")
    job_desc = st.text_area("Job description", height=320)

run = st.button("Score this role", type="primary")

if run:
    if not resume_text.strip():
        st.error("Please upload your résumé first.")
        st.stop()
    if not job_desc.strip():
        st.error("Please paste a job description.")
        st.stop()

    resume_id = save_resume(source=resume_source or "upload", raw_text=resume_text)
    job_id = save_job(
        description=job_desc,
        company=company or None,
        title=title or None,
        location=location or None,
        url=url or None,
    )

    result, model_used = score_role(resume_text, job_desc, use_ai=use_ai, min_base=min_base)
    save_score(job_id=job_id, resume_id=resume_id, result=result, model=model_used)

    # Session state for downstream tools
    st.session_state["last_resume_text"] = resume_text
    st.session_state["last_job_text"] = job_desc
    st.session_state["last_company"] = company or ""
    st.session_state["last_title"] = title or ""
    st.session_state["last_job_id"] = job_id
    st.session_state["last_score_result"] = result

    # Readable output
    if isinstance(result, dict) and result.get("error"):
        st.warning(result["error"])

    if not isinstance(result, dict):
        st.write(result)
    else:
        overall = result.get("overall_score") or result.get("total_score") or result.get("score")
        priority = result.get("priority")

        st.subheader("Fit score")
        if overall is not None:
            st.metric("Overall Score", overall)
        if priority:
            st.write(f"Priority: **{priority}**")

        def _render_list(title_txt: str, key: str):
            items = result.get(key) or []
            if items:
                st.subheader(title_txt)
                for x in items:
                    st.write(f"- {x}")

        _render_list("Why this fits", "why_this_fits")
        _render_list("Risks / gaps", "risks_or_gaps")
        _render_list("Top résumé edits", "top_resume_edits")
        _render_list("Interview leverage points", "interview_leverage_points")

        pitch = result.get("two_line_pitch")
        if pitch:
            st.subheader("Two-line pitch")
            st.write(pitch)

        with st.expander("Full scoring output (debug)", expanded=False):
            st.json(result)

    st.info(f"Scoring mode used: {model_used}")

st.divider()

# -------------------------
# Tailored résumé
# -------------------------
st.subheader("Tailored résumé (AI)")
tailor = st.button("Generate tailored résumé")

if tailor:
    resume_text_last = st.session_state.get("last_resume_text")
    job_text_last = st.session_state.get("last_job_text")

    if not resume_text_last or not job_text_last:
        st.error("First, score a role so the app has your latest résumé + job description.")
    else:
        with st.spinner("Generating tailored résumé..."):
            tailored = tailor_resume_ai(resume_text_last, job_text_last)

        if not tailored:
            st.error("AI tailoring not available. Confirm OPENAI_API_KEY is set in Render Environment.")
        else:
            st.success("Tailored résumé generated.")
            st.markdown("**Tailored headline**")
            st.write(safe_text(tailored.get("tailored_headline")))

            st.markdown("**Tailored executive summary**")
            for b in (tailored.get("tailored_summary") or []):
                st.write(f"- {b}")

            final_text = safe_text(tailored.get("final_resume_text"))
            st.text_area("Tailored résumé text", value=final_text, height=420)

            st.download_button(
                "Download tailored résumé (TXT)",
                data=final_text.encode("utf-8"),
                file_name="tailored_resume.txt",
                mime="text/plain",
            )

st.divider()

# -------------------------
# Positioning brief (opening locked in positioning_brief.py)
# -------------------------
st.subheader("Executive Positioning Brief (1-page)")
brief = st.button("Generate positioning brief (1-page)")

if brief:
    resume_text_last = st.session_state.get("last_resume_text")
    job_text_last = st.session_state.get("last_job_text")

    if not resume_text_last or not job_text_last:
        st.error("First, score a role so the app has your latest résumé + job description.")
    else:
        with st.spinner("Generating positioning brief..."):
            memo = generate_positioning_brief(resume_text_last, job_text_last)

        if memo:
            st.text_area("Positioning brief", value=memo, height=460)
            st.download_button(
                "Download positioning brief (TXT)",
                data=memo.encode("utf-8"),
                file_name="positioning_brief.txt",
                mime="text/plain",
            )

st.divider()

# -------------------------
# Recruiter Outreach Kit
# -------------------------
st.subheader("Recruiter Outreach Kit")
outreach = st.button("Generate recruiter outreach kit")

if outreach:
    resume_text_last = st.session_state.get("last_resume_text")
    job_text_last = st.session_state.get("last_job_text")

    if not resume_text_last or not job_text_last:
        st.error("First, score a role so the app has your latest résumé + job description.")
    else:
        with st.spinner("Generating outreach kit..."):
            kit = generate_recruiter_outreach(resume_text_last, job_text_last)

        if not kit:
            st.error("Outreach kit is not available. Confirm OPENAI_API_KEY is set in Render Environment.")
        else:
            st.success("Outreach kit generated.")
            email_text = safe_text(kit.get("email"))
            li_text = safe_text(kit.get("linkedin"))
            call_text = safe_text(kit.get("call_talking_points"))

            st.markdown("### Recruiter email")
            st.text_area("Email", value=email_text, height=200)

            st.markdown("### LinkedIn message")
            st.text_area("LinkedIn", value=li_text, height=120)

            st.markdown("### First-call talking points")
            st.text_area("Call talking points", value=call_text, height=180)

            bundle = (
                "RECRUITER EMAIL\n\n" + email_text
                + "\n\nLINKEDIN MESSAGE\n\n" + li_text
                + "\n\nFIRST-CALL TALKING POINTS\n\n" + call_text
            )

            st.download_button(
                "Download outreach kit (TXT)",
                data=bundle.encode("utf-8"),
                file_name="recruiter_outreach_kit.txt",
                mime="text/plain",
            )

st.divider()

# -------------------------
# DASHBOARD (Active roles only)
# -------------------------
st.subheader("Dashboard (Active roles)")

items_all = list_pipeline_items(active_only=True, limit=500)

if not items_all:
    st.info("No active pipeline roles yet. Add a scored role to the pipeline to populate the dashboard.")
else:
    # Metrics
    total_active = len(items_all)
    scores = [it.get("fit_score") for it in items_all if it.get("fit_score") is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None

    due_today = 0
    overdue = 0
    today = date.today()
    for it in items_all:
        d = parse_yyyy_mm_dd(it.get("next_action_date"))
        if not d:
            continue
        if d == today:
            due_today += 1
        elif d < today:
            overdue += 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active roles", total_active)
    m2.metric("Avg fit score", avg_score if avg_score is not None else "—")
    m3.metric("Next actions due today", due_today)
    m4.metric("Overdue next actions", overdue)

    # Stage counts
    st.markdown("### Roles by stage")
    stage_counts: dict[str, int] = {}
    for it in items_all:
        stage_counts[it.get("stage", "—")] = stage_counts.get(it.get("stage", "—"), 0) + 1

    # Render stage counts as a simple table
    for stage_name, count in sorted(stage_counts.items(), key=lambda x: (-x[1], x[0])):
        st.write(f"- **{stage_name}**: {count}")

    # Avg score by stage
    st.markdown("### Average score by stage")
    stage_scores: dict[str, list[float]] = {}
    for it in items_all:
        s = it.get("fit_score")
        if s is None:
            continue
        stage = it.get("stage", "—")
        stage_scores.setdefault(stage, []).append(float(s))

    if not stage_scores:
        st.caption("No fit scores stored yet. Score a role, then add it to pipeline to snapshot the score.")
    else:
        for stage_name, arr in sorted(stage_scores.items(), key=lambda x: x[0]):
            st.write(f"- **{stage_name}**: {round(sum(arr)/len(arr), 1)}")

    # Top roles
    st.markdown("### Top roles (by fit score)")
    top = sorted(
        items_all,
        key=lambda it: (it.get("fit_score") is not None, it.get("fit_score") or -1),
        reverse=True,
    )[:10]

    for it in top:
        title_txt = safe_text(it.get("title")) or "—"
        company_txt = safe_text(it.get("company")) or "—"
        loc = safe_text(it.get("location"))
        url_txt = safe_text(it.get("url"))
        score_txt = it.get("fit_score")
        pr = safe_text(it.get("priority"))

        line = f"**{title_txt} @ {company_txt}**"
        if loc:
            line += f" ({loc})"
        st.write(line)
        if url_txt:
            st.write(url_txt)
        st.write(f"Score: **{score_txt if score_txt is not None else '—'}**   |   Priority: **{pr or '—'}**   |   Stage: **{safe_text(it.get('stage'))}**")
        if it.get("next_action_date"):
            st.write(f"Next action: **{safe_text(it.get('next_action_date'))}**")
        st.divider()

st.divider()

# -------------------------
# Pipeline Tracker
# -------------------------
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
    stage = st.selectbox("Stage", PIPELINE_STAGES, index=0)
    next_action = st.text_input("Next action date (YYYY-MM-DD)", value="")
    notes = st.text_area("Notes", height=120)
    add_to_pipeline = st.button("Add to pipeline")

    if add_to_pipeline:
        job_id = st.session_state.get("last_job_id")
        if not job_id:
            st.error("Missing last job reference. Score a role first, then add it to the pipeline.")
            st.stop()

        score_data = st.session_state.get("last_score_result", {}) or {}
        fit_score = (
            score_data.get("overall_score")
            or score_data.get("total_score")
            or score_data.get("score")
        )
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

st.markdown("### Active roles")
items = list_pipeline_items(active_only=True, limit=200)

if not items:
    st.info("No active pipeline items yet.")
else:
    for it in items:
        title_txt = safe_text(it.get("title")) or "—"
        company_txt = safe_text(it.get("company")) or "—"
        loc = safe_text(it.get("location"))
        url_txt = safe_text(it.get("url"))

        header = f"{title_txt} @ {company_txt}" + (f" ({loc})" if loc else "")
        st.markdown(f"**{header}**")
        if url_txt:
            st.write(url_txt)

        st.write(f"Stage: **{safe_text(it.get('stage'))}**")
        if it.get("fit_score") is not None:
            st.write(f"Fit score: **{it.get('fit_score')}**   |   Priority: **{safe_text(it.get('priority')) or '—'}**")
        if it.get("next_action_date"):
            st.write(f"Next action: **{safe_text(it.get('next_action_date'))}**")
        if it.get("notes"):
            st.write(safe_text(it.get("notes")))

        with st.expander("Update this pipeline item", expanded=False):
            new_stage = st.selectbox(
                "New stage",
                PIPELINE_STAGES,
                index=PIPELINE_STAGES.index(it["stage"]) if it.get("stage") in PIPELINE_STAGES else 0,
                key=f"stage_{it['pipeline_id']}",
            )
            new_next = st.text_input(
                "Next action date (YYYY-MM-DD)",
                value=safe_text(it.get("next_action_date")),
                key=f"next_{it['pipeline_id']}",
            )
            new_notes = st.text_area(
                "Notes",
                value=safe_text(it.get("notes")),
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
                    fit_score=it.get("fit_score"),
                    priority=it.get("priority"),
                )
                st.success("Updated. Refreshing list...")
                st.rerun()

        st.divider()

st.divider()
st.subheader("Recent scored roles")
recent = list_recent_scores(limit=10)
if not recent:
    st.caption("No scores saved yet.")
else:
    for r in recent:
        st.write(
            f"- {safe_text(r.get('title'))} @ {safe_text(r.get('company'))} "
            f"({safe_text(r.get('location'))}) — model: {safe_text(r.get('model'))}"
        )
