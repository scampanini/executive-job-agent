import streamlit as st

from app.file_parsers import load_uploaded_file
from app.db import (
    get_conn,
    get_latest_grounded_gap_result,
    get_portfolio_texts,
    init_db,
    list_scores,
    save_grounded_gap_result,
    save_job,
    save_portfolio_text,
    save_resume,
    save_score,
)
from app.scoring_engine import score_role
from app.utils import job_desc_mentions_salary, safe_text


st.set_page_config(page_title="Executive Job Agent", layout="wide")

if "show_debug" not in st.session_state:
    st.session_state["show_debug"] = False
if "last_score_result" not in st.session_state:
    st.session_state["last_score_result"] = None
if "last_model_used" not in st.session_state:
    st.session_state["last_model_used"] = None
if "last_gap_result" not in st.session_state:
    st.session_state["last_gap_result"] = None


@st.cache_resource
def init_connection():
    conn = get_conn()
    init_db(conn)
    return conn


conn = init_connection()

st.title("Executive Job Agent")

tab1, tab2 = st.tabs(["Score Role", "Pipeline Dashboard"])

with tab1:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("1) Candidate inputs")
    
        resume_file = st.file_uploader(
            "Upload résumé (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="resume_file",
        )
        resume_text_manual = st.text_area("Or paste résumé text", height=220)
    
        portfolio_file = st.file_uploader(
            "Upload portfolio / case study file (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            key="portfolio_file",
        )
        portfolio_text_manual = st.text_area("Or paste portfolio / case study text", height=180)
    
        use_ai = st.checkbox("Use OpenAI scoring", value=True)
        min_base = st.number_input(
            "Minimum score floor when salary is present",
            min_value=0,
            max_value=100,
            value=0,
            step=1,
        )
    
        resume_file_text = load_uploaded_file(resume_file) if resume_file else ""
        portfolio_file_text = load_uploaded_file(portfolio_file) if portfolio_file else ""
    
        resume_text = resume_file_text or resume_text_manual
        portfolio_text = portfolio_file_text or portfolio_text_manual
    
        if resume_file and resume_text:
            st.caption(f"Loaded résumé file: {resume_file.name}")
    
        if portfolio_file and portfolio_text:
            st.caption(f"Loaded portfolio file: {portfolio_file.name}")

    with col_r:
        st.subheader("2) Job description")
        company = st.text_input("Company (optional)", value="")
        title = st.text_input("Title (optional)", value="")
        location = st.text_input("Location (optional)", value="")
        url = st.text_input("Job URL (optional)", value="")
        job_desc = st.text_area("Job description", height=320)

    with st.form("score_role_form"):
        run = st.form_submit_button("Score role")

    st.checkbox("Show grounded debug JSON", key="show_debug", value=False)

    if run:
        if not safe_text(resume_text):
            st.error("Please upload or paste your résumé first.")
            st.stop()

        if not safe_text(job_desc):
            st.error("Please paste a job description.")
            st.stop()

        min_base_for_scoring = 0
        if job_desc_mentions_salary(job_desc):
            min_base_for_scoring = int(min_base)

        try:
            job_id = save_job(
                conn=conn,
                description=job_desc,
                company=safe_text(company) or None,
                title=safe_text(title) or None,
                location=safe_text(location) or None,
                url=safe_text(url) or None,
            )
            resume_id = save_resume(conn=conn, source="manual", raw_text=resume_text)

            if safe_text(portfolio_text):
                save_portfolio_text(conn=conn, text=portfolio_text, resume_id=resume_id, job_id=job_id)

            portfolio_texts = get_portfolio_texts(conn=conn, resume_id=resume_id, job_id=job_id, limit=50)
            portfolio_for_scoring = "\n\n".join([x for x in portfolio_texts if safe_text(x)])

            result, model_used = score_role(
                resume_text=resume_text,
                job_text=job_desc,
                use_ai=use_ai,
                min_base=min_base_for_scoring,
                portfolio_text=portfolio_for_scoring,
                gap_answers_text="",
            )

            try:
                from app.gap_engine import run_grounded_gap_analysis
            
                gap_result = run_grounded_gap_analysis(
                    resume_text=resume_text,
                    job_description=job_desc,
                    portfolio_texts=portfolio_texts,
                )
            except Exception as e:
                gap_result = {
                    "overall_alignment_score": 0,
                    "summary": f"Grounded gap analysis unavailable: {e}",
                    "requirements": [],
                    "hard_gaps": [],
                    "partial_gaps": [],
                    "strong_matches": [],
                }

            save_score(conn=conn, job_id=job_id, resume_id=resume_id, result=result, model=model_used)
            save_grounded_gap_result(conn=conn, resume_id=resume_id, job_id=job_id, result=gap_result)

            st.session_state["last_score_result"] = result
            st.session_state["last_model_used"] = model_used
            st.session_state["last_gap_result"] = gap_result
            st.session_state["last_job_id"] = job_id

            st.success("Scoring completed.")

        except Exception as e:
            st.error(f"Run failed: {e}")

    result = st.session_state.get("last_score_result")
    gap_result_ui = st.session_state.get("last_gap_result")
    model_used_ui = st.session_state.get("last_model_used")

    if result:
        st.divider()
        st.subheader("Score result")
        st.metric("Fit score", result.get("score", 0))
        st.write(f"Priority: **{safe_text(result.get('priority')) or '—'}**")

        why_this_fits = result.get("why_this_fits") or []
        risks_or_gaps = result.get("risks_or_gaps") or []
        top_resume_edits = result.get("top_resume_edits") or []
        interview_leverage_points = result.get("interview_leverage_points") or []

        if why_this_fits:
            st.subheader("Why this fits")
            for item in why_this_fits:
                st.write(f"- {item}")

        if risks_or_gaps:
            st.subheader("Risks / gaps")
            for item in risks_or_gaps:
                st.write(f"- {item}")

        if top_resume_edits:
            st.subheader("Top résumé edits")
            for item in top_resume_edits:
                st.write(f"- {item}")

        if interview_leverage_points:
            st.subheader("Interview leverage points")
            for item in interview_leverage_points:
                st.write(f"- {item}")

        if result.get("two_line_pitch"):
            st.subheader("Two-line pitch")
            st.write(result["two_line_pitch"])

        if model_used_ui:
            st.info(f"Scoring mode used: {model_used_ui}")

    if gap_result_ui:
        st.divider()
        st.subheader("Gap Insights")
        st.write(gap_result_ui.get("summary", ""))
        st.metric("Alignment Score", gap_result_ui.get("overall_alignment_score", 0))

        c1, c2, c3 = st.columns(3)
        c1.metric("Hard gaps", len(gap_result_ui.get("hard_gaps") or []))
        c2.metric("Partial gaps", len(gap_result_ui.get("partial_gaps") or []))
        c3.metric("Strong matches", len(gap_result_ui.get("strong_matches") or []))

        with st.expander("Grounded requirement details", expanded=False):
            st.json(gap_result_ui.get("requirements") or [])

    if st.session_state.get("show_debug"):
        st.divider()
        st.subheader("Debug")
        st.write("Last model:", model_used_ui)
        st.write("Last job id:", st.session_state.get("last_job_id"))
        st.json(result or {})
        st.json(gap_result_ui or {})

with tab2:
    st.subheader("Pipeline Dashboard")
    rows = list_scores(conn=conn, limit=100)

    if not rows:
        st.caption("No scored roles yet.")
    else:
        total = len(rows)
        high = len([r for r in rows if (r.get("result") or {}).get("priority") == "High"])
        medium = len([r for r in rows if (r.get("result") or {}).get("priority") == "Medium"])
        low = len([r for r in rows if (r.get("result") or {}).get("priority") == "Low"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total scored", total)
        c2.metric("High priority", high)
        c3.metric("Medium priority", medium)
        c4.metric("Low priority", low)

        for row in rows:
            result = row.get("result") or {}
            with st.container():
                st.write(f"**{safe_text(row.get('title')) or 'Untitled role'}**")
                st.write(f"Company: **{safe_text(row.get('company')) or '—'}**")
                st.write(f"Location: **{safe_text(row.get('location')) or '—'}**")
                st.write(
                    f"Fit score: **{result.get('score', 0)}** | "
                    f"Priority: **{safe_text(result.get('priority')) or '—'}** | "
                    f"Model: **{safe_text(row.get('model')) or '—'}**"
                )
                st.caption(f"Created: {row.get('created_at')}")
                st.divider()
