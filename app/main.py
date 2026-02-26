# =========================
# main.py (STABLE BASELINE)
# =========================

# ---- stdlib ----
import sys
from pathlib import Path
import os
import tempfile
import json
import re
import sqlite3
from datetime import datetime, date
from collections import Counter

# ---- third-party ----
import pandas as pd
import streamlit as st

# Headless-safe matplotlib for Render (must be before pyplot)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- Ensure repo root on path BEFORE importing app.* ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---- app imports ----
from app.core.resume_parse import load_resume
from app.core.scoring import heuristic_score, blended_score
from app.core.resume_tailor import tailor_resume_ai
from app.core.positioning_brief import generate_positioning_brief
from app.core.recruiter_outreach import generate_recruiter_outreach

from app.core.storage import (
    init_db,
    get_setting,
    set_setting,
    save_document,
    create_gap_question,
    list_gap_questions,
    answer_gap_question,
    attach_unlinked_gap_questions_to_job,
    save_resume,
    save_job,
    save_score,
    create_pipeline_item,
    update_pipeline_item,
    list_pipeline_items,
)
from app.core.db_conn import get_conn
from app.core.schema_grounded_gap import ensure_grounded_gap_tables
from app.core.portfolio_store import get_portfolio_texts
from app.core.build_evidence_cache import build_evidence_cache_for_job
from app.core.grounded_gap_engine import run_grounded_gap_analysis, save_grounded_gap_result
from app.core.job_resume_fetch import get_job_description


# -------------------------
# Helpers
# -------------------------
def safe_text(x) -> str:
    return "" if x is None else str(x)


def slugify_filename(s: str, max_len: int = 60) -> str:
    s = safe_text(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s[:max_len] or "unknown")


def dl_name(company: str, base: str, ext: str = "txt") -> str:
    c = slugify_filename(company)
    b = slugify_filename(base)
    return f"{c}__{b}.{ext}"


def append_job_description_block(content: str, company: str, job_desc: str) -> str:
    header = f"\n\n{'='*80}\nJOB DESCRIPTION REFERENCE — {safe_text(company)}\n{'='*80}\n\n"
    return f"{content.strip()}{header}{safe_text(job_desc).strip()}\n"


def parse_yyyy_mm_dd(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()
    except Exception:
        return None


def grounded_has_gaps(gap_result: dict | None) -> bool:
    if not gap_result:
        return False
    return bool(
        (gap_result.get("hard_gaps") or [])
        or (gap_result.get("partial_gaps") or [])
        or (gap_result.get("signal_gaps") or [])
    )


def _call_scorer(fn, resume_text: str, job_text: str, min_base: int):
    # Backward-compatible signature attempts
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


def score_role(
    resume_text: str,
    job_text: str,
    use_ai: bool,
    min_base: int,
    portfolio_text: str = "",
    gap_answers_text: str = "",
):
    combined = resume_text

    if portfolio_text.strip():
        combined += "\n\n=== PORTFOLIO / EXPERIENCE EXAMPLES (USER-PROVIDED) ===\n"
        combined += portfolio_text.strip()

    if gap_answers_text.strip():
        combined += "\n\n=== GAP ANSWERS (USER-PROVIDED) ===\n"
        combined += gap_answers_text.strip()

    if use_ai:
        try:
            return (_call_scorer(blended_score, combined, job_text, min_base), "openai")
        except Exception as e:
            # Safe fallback
            base = _call_scorer(heuristic_score, combined, job_text, min_base)
            if isinstance(base, dict):
                base = {"error": f"AI scoring failed; falling back to heuristic scoring. Details: {e}", **base}
            return (base, "heuristic")

    return (_call_scorer(heuristic_score, combined, job_text, min_base), "heuristic")


@st.cache_data(ttl=15)
def make_donut_figure(labels: tuple, values: tuple, title: str):
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, wedgeprops={"width": 0.4})
    ax.set_title(title)
    return fig


def _updated_at_sort_value(val) -> int:
    if val is None:
        return 0

    # epoch seconds?
    try:
        return int(val)
    except Exception:
        pass

    # ISO datetime?
    try:
        return int(datetime.fromisoformat(str(val).replace("Z", "+00:00")).timestamp())
    except Exception:
        return 0


def get_latest_grounded_gap_result(conn, job_id: int):
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT result
            FROM grounded_gap_results
            WHERE job_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (job_id,),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return None

        if isinstance(row[0], str):
            try:
                return json.loads(row[0])
            except Exception:
                return row[0]

        return row[0]
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return None


# -------------------------
# App init
# -------------------------
st.set_page_config(page_title="Executive Job Agent (Personal)", layout="wide")
init_db()
st.title("Executive Job Agent (Personal)")

# -------------------------
# Sidebar settings
# -------------------------
with st.sidebar:
    st.header("Settings")
    min_base = st.number_input("Minimum base salary ($)", min_value=0, value=275000, step=5000)
    use_ai = st.toggle("Use AI for scoring", value=True)

    st.divider()
    st.subheader("AI status")
    st.write("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
    st.write("OPENAI_MODEL:", os.getenv("OPENAI_MODEL", "(not set)"))

    st.divider()
    st.subheader("Email Address for Job Targets")

    enabled = get_setting("gmail_ingest_enabled", "0")
    gmail_ingest_enabled = st.toggle("Enable Gmail ingest (read-only)", value=(enabled == "1"))
    set_setting("gmail_ingest_enabled", "1" if gmail_ingest_enabled else "0")

    env_email = os.getenv("TARGET_INBOX_EMAIL", "")
    current_email = get_setting("target_inbox_email", env_email) or ""
    target_email = st.text_input(
        "Target inbox email address",
        value=current_email,
        help="Stored in SQLite. Falls back to TARGET_INBOX_EMAIL if not set.",
    )
    if st.button("Save inbox email", use_container_width=True):
        set_setting("target_inbox_email", target_email.strip())
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Current inbox: {get_setting('target_inbox_email', env_email) or '(not set)'}")

# -------------------------
# Main columns
# -------------------------
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("1) Upload your résumé")
    uploaded = st.file_uploader("Upload résumé (PDF/DOCX)", type=["pdf", "docx"])

    resume_text = ""
    resume_source = ""

    if uploaded:
        suffix = ".pdf" if uploaded.type == "application/pdf" else ".docx"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            resume = load_resume(tmp_path, None)
            resume_text = getattr(resume, "raw_text", None) or getattr(resume, "text", None) or str(resume)
            resume_source = uploaded.name
            st.success(f"Loaded résumé: {resume_source}")
        except Exception as e:
            st.error(f"Could not parse résumé: {e}")
            resume_text = ""
            resume_source = ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    st.divider()
    st.subheader("2) Upload your portfolio / experience examples (optional)")

    uploaded_portfolio = st.file_uploader(
        "Upload portfolio (PDF/DOCX)",
        type=["pdf", "docx"],
        key="portfolio_uploader",
        help="Optional. Add real examples, metrics, leadership scope, and impact stories not fully reflected in your résumé.",
    )

    portfolio_text = ""
    portfolio_source = ""

    if uploaded_portfolio:
        suffix = ".pdf" if uploaded_portfolio.type == "application/pdf" else ".docx"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_portfolio.getvalue())
                tmp_path = tmp.name

            portfolio = load_resume(tmp_path, None)
            portfolio_text = getattr(portfolio, "raw_text", None) or getattr(portfolio, "text", None) or str(portfolio)
            portfolio_source = uploaded_portfolio.name
            st.success(f"Loaded portfolio: {portfolio_source}")

            save_document(
                doc_type="portfolio",
                raw_text=portfolio_text,
                source=portfolio_source,
                mime=uploaded_portfolio.type,
            )
        except Exception as e:
            st.error(f"Could not parse portfolio: {e}")
            portfolio_text = ""
            portfolio_source = ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

with col_r:
    st.subheader("2) Paste job description")
    company = st.text_input("Company (optional)", value="")
    title = st.text_input("Title (optional)", value="")
    location = st.text_input("Location (optional)", value="")
    url = st.text_input("Job URL (optional)", value="")
    job_desc = st.text_area("Job description", height=320)

    st.divider()
    st.subheader("Fill gaps (so outputs stay factual)")

    if st.button("Generate gap questions", use_container_width=True):
        questions = [
            "What are 1–2 examples of supporting a CEO or C-suite leader (cadence, priorities, decision support)?",
            "Do you have an example involving Board materials, QBRs, or executive presentations? What was your role?",
            "What tools/systems are you strongest in (Office, Google, Slack, Teams, Concur, Workday, ATS, CRM, etc.)?",
            "Have you managed budgets, purchase orders, invoices, or vendor relationships? Any approximate scope?",
            "Any experience with confidential matters (reorgs, M&A, HR issues, legal) you can describe at a high level?",
            "Any example of process improvement (what you changed, why, and the result)?",
        ]
        for q in questions:
            create_gap_question(question=q, gap_type="resume", job_id=None)
        st.success("Gap questions created. Answer them below.")

    gaps = list_gap_questions(job_id=None, unanswered_only=True, limit=20)
    if not gaps:
        st.caption("No open gap questions yet. Click “Generate gap questions” to create a short set.")
    else:
        for item in gaps:
            st.write(f"**Q:** {item['question']}")
            ans = st.text_input("Your answer", value="", key=f"gap_answer_{item['id']}")
            if st.button("Save answer", key=f"gap_save_{item['id']}"):
                answer_gap_question(question_id=int(item["id"]), answer=ans.strip())
                st.cache_data.clear()
                st.rerun()

with st.form("score_role_form"):
    run = st.form_submit_button("Score role")

show_debug = st.checkbox("Show grounded debug JSON", value=False)
st.session_state["show_debug"] = show_debug


# -------------------------
# Run scoring + grounded gap engine
# -------------------------
if run:
    if not resume_text.strip():
        st.error("Please upload your résumé first.")
        st.stop()

    if not job_desc.strip():
        st.error("Please paste a job description.")
        st.stop()

    # Save job + resume
    job_id = save_job(
        description=job_desc,
        company=company or None,
        title=title or None,
        location=location or None,
        url=url or None,
    )
    resume_id = save_resume(source=resume_source or "upload", raw_text=resume_text)

    # Persist identifiers + canonical fields for grounded downloads
    st.session_state["last_job_id"] = job_id
    st.session_state["last_resume_id"] = resume_id
    st.session_state["last_company"] = company or ""
    st.session_state["last_title"] = title or ""
    st.session_state["last_job_text"] = job_desc

    conn = None
    try:
        conn = get_conn()
        ensure_grounded_gap_tables(conn)

        # Portfolio texts for evidence cache + scoring fallback
        portfolio_texts = get_portfolio_texts(conn=conn, resume_id=resume_id, job_id=job_id, limit=50) or []

        build_evidence_cache_for_job(
            conn=conn,
            resume_id=resume_id,
            job_id=job_id,
            resume_text=resume_text,
            portfolio_texts=portfolio_texts,
        )

        gap_result = run_grounded_gap_analysis(
            conn=conn,
            resume_id=resume_id,
            job_id=job_id,
            job_description=job_desc,
            resume_text=resume_text,
        )

        st.session_state["gap_result_this_run"] = gap_result
        st.session_state["last_gap_result"] = gap_result
        st.session_state["job_id"] = job_id

        use_gap_questions = grounded_has_gaps(gap_result)
        st.session_state["last_use_gap_questions"] = use_gap_questions

        # Save grounded result only if valid
        if gap_result:
            save_grounded_gap_result(conn=conn, resume_id=resume_id, job_id=job_id, result=gap_result)
        else:
            st.warning("Grounded gap engine returned no result; nothing was saved.")

        if use_gap_questions:
            attach_unlinked_gap_questions_to_job(job_id=job_id, limit=50)

        # Gap answer text (grounded addendum)
        gap_answers_text = ""
        if use_gap_questions:
            answered = list_gap_questions(job_id=job_id, unanswered_only=False, limit=50) or []
            answered_pairs = []
            for item in answered:
                if item.get("answer"):
                    answered_pairs.append(f"Q: {item['question']}\nA: {item['answer']}")
            gap_answers_text = "\n\n".join(answered_pairs)

        # Merge portfolio into scoring context even if user didn't re-upload this session
        if portfolio_text.strip():
            portfolio_for_scoring = portfolio_text
        else:
            # portfolio_texts could be dicts or strings depending on your store
            joined = []
            for p in portfolio_texts:
                if isinstance(p, dict):
                    joined.append(safe_text(p.get("raw_text") or p.get("text") or ""))
                else:
                    joined.append(safe_text(p))
            portfolio_for_scoring = "\n\n".join([x for x in joined if x.strip()])

        # Blended scoring
        result, model_used = score_role(
            resume_text=resume_text,
            job_text=job_desc,
            use_ai=use_ai,
            min_base=min_base,
            portfolio_text=portfolio_for_scoring,
            gap_answers_text=gap_answers_text,
        )

        st.session_state["last_score_result"] = result
        st.session_state["last_model_used"] = model_used
        st.session_state["last_resume_text"] = resume_text

        save_score(job_id=job_id, resume_id=resume_id, result=result, model=model_used)

    except Exception as e:
        st.error(f"Run failed: {e}")
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# -------------------------
# Persisted UI state
# -------------------------
result_ui = st.session_state.get("last_score_result")
model_used_ui = st.session_state.get("last_model_used")
job_id_ui = st.session_state.get("job_id") or st.session_state.get("last_job_id")
resume_id_ui = st.session_state.get("last_resume_id")

gap_result_ui = st.session_state.get("last_gap_result")
use_gap_questions_ui = st.session_state.get("last_use_gap_questions", False)
show_debug = st.session_state.get("show_debug", False)

if show_debug and gap_result_ui:
    with st.expander("🔬 DEBUG – Full grounded gap_result", expanded=False):
        st.json(gap_result_ui)


def render_gap_block(g):
    if not g:
        st.info("No grounded gap result available.")
        return
    st.write(g.get("summary", ""))
    st.metric("Alignment Score", f"{g.get('overall_alignment_score', 0)}/100")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hard gaps", len(g.get("hard_gaps") or []))
    c2.metric("Partial gaps", len(g.get("partial_gaps") or []))
    c3.metric("Signal gaps", len(g.get("signal_gaps") or []))


st.subheader("🔎 Grounded Gap Analysis (this run)")
if "gap_result_this_run" in st.session_state:
    render_gap_block(st.session_state.get("gap_result_this_run"))
else:
    st.caption("No new run yet in this session.")

st.subheader("🔎 Grounded Gap Analysis (latest)")
conn_ui = None
gap_result_latest = None
try:
    conn_ui = get_conn()
    gap_result_latest = get_latest_grounded_gap_result(conn_ui, int(job_id_ui)) if job_id_ui else None
except Exception as e:
    st.warning(f"Could not load latest grounded gap result: {e}")
finally:
    if conn_ui is not None:
        try:
            conn_ui.close()
        except Exception:
            pass

render_gap_block(gap_result_latest)

if not use_gap_questions_ui:
    st.info("No grounded gaps detected — skipping gap questions.")


# -------------------------
# Fit score
# -------------------------
st.subheader("Fit score")

if result_ui is None:
    st.info("No score generated yet. Click “Score role”.")
elif isinstance(result_ui, dict) and result_ui.get("error"):
    st.warning(result_ui["error"])
    if show_debug:
        st.json(result_ui)
elif not isinstance(result_ui, dict):
    st.write(result_ui)
else:
    overall = result_ui.get("overall_score")
    priority = result_ui.get("priority")

    if overall is not None:
        st.metric("Overall Score", overall)

    if priority:
        st.write(f"Priority: **{priority}**")

    def _render_list(title_txt: str, key: str):
        items = result_ui.get(key) or []
        if items:
            st.subheader(title_txt)
            for x in items:
                st.write(f"- {x}")

    _render_list("Why this fits", "why_this_fits")
    _render_list("Risks / gaps", "risks_or_gaps")
    _render_list("Top résumé edits", "top_resume_edits")
    _render_list("Interview leverage points", "interview_leverage_points")

    pitch = result_ui.get("two_line_pitch")
    if pitch:
        st.subheader("Two-line pitch")
        st.write(pitch)

    with st.expander("Full scoring output (debug)", expanded=False):
        st.json(result_ui)

    if model_used_ui:
        st.info(f"Scoring mode used: {model_used_ui}")

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
            st.error("AI tailoring not available. Confirm OPENAI_API_KEY is set.")
        else:
            st.session_state["last_tailored"] = tailored
            st.success("Tailored résumé generated.")

tailored = st.session_state.get("last_tailored")
if tailored:
    headline = safe_text(tailored.get("tailored_headline", ""))
    st.markdown("**Tailored headline**")
    st.write(headline)

    st.markdown("**Tailored executive summary**")
    for b in (tailored.get("tailored_summary") or []):
        st.write(f"- {safe_text(b)}")

    final_text = safe_text(tailored.get("final_resume_text")) or ""
    st.text_area("Tailored résumé text", value=final_text, height=420)

    company_for_file = st.session_state.get("last_company") or company or ""
    job_desc_for_file = st.session_state.get("last_job_text") or job_desc or ""

    st.download_button(
        "Download tailored résumé (TXT)",
        data=append_job_description_block(final_text, company_for_file, job_desc_for_file).encode("utf-8"),
        file_name=dl_name(company_for_file, "tailored_resume", "txt"),
        mime="text/plain",
    )

with st.expander("📄 Job description (saved)", expanded=False):
    jd_show = ""
    job_id_show = st.session_state.get("last_job_id")
    if job_id_show is not None:
        try:
            conn = get_conn()
            jd_show = get_job_description(conn, int(job_id_show))
        except Exception:
            jd_show = ""
        finally:
            try:
                conn.close()
            except Exception:
                pass
    st.text_area("Job description", value=jd_show, height=260)

st.divider()


# -------------------------
# Positioning brief
# -------------------------
st.subheader("Executive Positioning Brief (1-page)")
brief_btn = st.button("Generate positioning brief (1-page)")

if brief_btn:
    resume_text_last = st.session_state.get("last_resume_text")
    job_text_last = st.session_state.get("last_job_text")
    if not resume_text_last or not job_text_last:
        st.error("First, score a role so the app has your latest résumé + job description.")
    else:
        with st.spinner("Generating positioning brief..."):
            memo = generate_positioning_brief(resume_text_last, job_text_last)
        st.session_state["last_positioning_brief"] = memo

memo = st.session_state.get("last_positioning_brief")
if memo:
    st.text_area("Positioning brief", value=memo, height=460)

    company_for_file = st.session_state.get("last_company") or company or ""
    job_desc_for_file = st.session_state.get("last_job_text") or job_desc or ""

    full_text = append_job_description_block(memo, company_for_file, job_desc_for_file)
    st.download_button(
        "Download positioning brief (TXT)",
        data=full_text.encode("utf-8"),
        file_name=dl_name(company_for_file, "executive_positioning_brief", "txt"),
        mime="text/plain",
    )

st.divider()


# -------------------------
# Recruiter Outreach Kit
# -------------------------
st.subheader("Recruiter Outreach Kit")
outreach_btn = st.button("Generate recruiter outreach kit")

if outreach_btn:
    resume_text_last = st.session_state.get("last_resume_text")
    job_text_last = st.session_state.get("last_job_text")
    if not resume_text_last or not job_text_last:
        st.error("First, score a role so the app has your latest résumé + job description.")
    else:
        with st.spinner("Generating outreach kit..."):
            kit = generate_recruiter_outreach(resume_text_last, job_text_last)
        st.session_state["last_outreach_kit"] = kit

kit = st.session_state.get("last_outreach_kit")
if not kit:
    st.caption("Generate an outreach kit to see email + LinkedIn + talking points here.")
elif isinstance(kit, dict) and kit.get("error"):
    st.error(safe_text(kit.get("error")))
else:
    st.success("Outreach kit generated.")
    email_text = safe_text(kit.get("email"))
    li_text = safe_text(kit.get("linkedin"))
    call_text = safe_text(kit.get("call_talking_points"))

    st.markdown("### Recruiter email")
    st.text_area("Email", value=email_text, height=200, key="outreach_email")

    st.markdown("### LinkedIn message")
    st.text_area("LinkedIn", value=li_text, height=120, key="outreach_li")

    st.markdown("### First-call talking points")
    st.text_area("Call talking points", value=call_text, height=180, key="outreach_call")

    bundle = (
        "RECRUITER EMAIL\n\n" + email_text
        + "\n\nLINKEDIN MESSAGE\n\n" + li_text
        + "\n\nFIRST-CALL TALKING POINTS\n\n" + call_text
    )

    company_for_file = st.session_state.get("last_company") or company or ""
    job_desc_for_file = st.session_state.get("last_job_text") or job_desc or ""
    full_outreach = append_job_description_block(bundle, company_for_file, job_desc_for_file)

    st.download_button(
        "Download outreach kit (TXT)",
        data=full_outreach.encode("utf-8"),
        file_name=dl_name(company_for_file, "recruiter_outreach_kit", "txt"),
        mime="text/plain",
    )

st.divider()


# -------------------------
# Dashboard + Pipeline Tracker
# -------------------------
st.subheader("Dashboard (Active roles)")
items_all = list_pipeline_items(active_only=True, limit=500) or []

if not items_all:
    st.info("No active pipeline roles yet. Add a scored role to the pipeline to populate the dashboard.")
else:
    stage_counts = Counter((it.get("stage") or "—") for it in items_all)
    labels = tuple(stage_counts.keys())
    values = tuple(stage_counts.values())
    fig = make_donut_figure(labels, values, "Pipeline by Stage")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

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

QUICK_STAGE_BUTTONS = [
    "Interested",
    "Applied",
    "Recruiter screen",
    "Hiring manager screen",
    "Interview loop",
    "Offer",
]

with st.expander("Add current role to pipeline", expanded=False):
    st.caption("Tip: Score a role first so company/title are captured, then add it to your pipeline.")
    stage = st.selectbox("Stage", PIPELINE_STAGES, index=0, key="add_stage")
    next_action = st.text_input("Next action date (YYYY-MM-DD)", value="", key="add_next")
    notes = st.text_area("Notes", height=120, key="add_notes")
    add_to_pipeline = st.button("Add to pipeline", key="add_btn")

    if add_to_pipeline:
        job_id = st.session_state.get("last_job_id")
        if not job_id:
            st.error("Missing last job reference. Score a role first, then add it to the pipeline.")
            st.stop()

        score_data = st.session_state.get("last_score_result", {}) or {}
        fit_score = score_data.get("overall_score") or score_data.get("total_score") or score_data.get("score")
        priority = score_data.get("priority")

        create_pipeline_item(
            job_id=job_id,
            stage=stage,
            next_action_date=next_action or None,
            notes=notes or None,
            fit_score=fit_score,
            priority=priority,
        )
        st.cache_data.clear()
        st.toast("Added to pipeline")
        st.rerun()

st.markdown("### Active roles")

items = items_all

st.markdown("#### Pipeline filters")
pipeline_stage_filter = st.multiselect(
    "Filter stages",
    options=PIPELINE_STAGES,
    default=PIPELINE_STAGES,
    key="pipe_stage_filter",
)

pipeline_sort = st.selectbox(
    "Sort pipeline by",
    options=["Last updated (newest)", "Fit score (high→low)", "Next action date (soonest)"],
    index=0,
    key="pipe_sort",
)

pipeline_overdue_only = st.checkbox("Show only overdue next actions", value=False, key="pipe_overdue_only")
today = date.today()

def _pipeline_pass(it) -> bool:
    stage_val = it.get("stage") or "—"
    if pipeline_stage_filter and stage_val not in pipeline_stage_filter:
        return False
    if pipeline_overdue_only:
        d = parse_yyyy_mm_dd(it.get("next_action_date"))
        return bool(d and d < today)
    return True

items = [it for it in items if _pipeline_pass(it)]

def _pipeline_sort_key(it):
    if pipeline_sort == "Fit score (high→low)":
        score = it.get("fit_score")
        try:
            return float(score)
        except Exception:
            return float("-inf")

    if pipeline_sort == "Next action date (soonest)":
        d = parse_yyyy_mm_dd(it.get("next_action_date"))
        return d or date(9999, 12, 31)

    return _updated_at_sort_value(it.get("updated_at"))

reverse = pipeline_sort in ["Last updated (newest)", "Fit score (high→low)"]
items = sorted(items, key=_pipeline_sort_key, reverse=reverse)

if not items:
    st.info("No active pipeline items yet.")
else:
    for it in items:
        pid = it.get("pipeline_id") or it.get("id")
        if pid is None:
            continue

        _fs = it.get("fit_score")
        try:
            _fs = float(_fs) if _fs is not None else None
        except Exception:
            _fs = None

        title_txt = safe_text(it.get("title")) or "—"
        company_txt = safe_text(it.get("company")) or "—"
        loc = safe_text(it.get("location"))
        url_txt = safe_text(it.get("url"))

        header = f"{title_txt} @ {company_txt}" + (f" ({loc})" if loc else "")
        st.markdown(f"**{header}**")
        if url_txt:
            st.write(url_txt)

        # One-click stage buttons
        cols = st.columns(len(QUICK_STAGE_BUTTONS))
        for idx, target_stage in enumerate(QUICK_STAGE_BUTTONS):
            if cols[idx].button(
                target_stage,
                key=f"quick_{pid}_{slugify_filename(target_stage)}",
                use_container_width=True,
            ):
                update_pipeline_item(
                    pipeline_id=pid,
                    stage=target_stage,
                    next_action_date=it.get("next_action_date"),
                    notes=it.get("notes"),
                    is_active=True,
                    fit_score=_fs,
                    priority=it.get("priority"),
                )
                st.cache_data.clear()
                st.toast(f"Stage updated: {target_stage}")
                st.rerun()

        st.write(f"Stage: **{safe_text(it.get('stage'))}**")
        if it.get("fit_score") is not None:
            st.write(
                f"Fit score: **{it.get('fit_score')}**   |   "
                f"Priority: **{safe_text(it.get('priority')) or '—'}**"
            )
        if it.get("next_action_date"):
            st.write(f"Next action: **{safe_text(it.get('next_action_date'))}**")
        if it.get("notes"):
            st.write(safe_text(it.get("notes")))

        # Edit block
        with st.expander("Edit role", expanded=False):
            new_stage = st.selectbox(
                "Stage",
                PIPELINE_STAGES,
                index=max(0, PIPELINE_STAGES.index(it.get("stage")) if it.get("stage") in PIPELINE_STAGES else 0),
                key=f"edit_stage_{pid}",
            )
            new_next = st.text_input(
                "Next action date (YYYY-MM-DD)",
                value=safe_text(it.get("next_action_date")),
                key=f"edit_next_{pid}",
            )
            new_notes = st.text_area(
                "Notes",
                value=safe_text(it.get("notes")),
                height=120,
                key=f"edit_notes_{pid}",
            )
            if st.button("Save changes", key=f"save_{pid}"):
                update_pipeline_item(
                    pipeline_id=pid,
                    stage=new_stage,
                    next_action_date=new_next or None,
                    notes=new_notes or None,
                    is_active=True,
                    fit_score=_fs,
                    priority=it.get("priority"),
                )
                st.cache_data.clear()
                st.toast("Saved")
                st.rerun()

        st.divider()

# -------------------------
# (Optional) Recent scores / history (safe)
# -------------------------
with st.expander("Recent scoring runs (latest 20)", expanded=False):
    try:
        # If your storage layer returns something else, this will just no-op gracefully.
        # You can remove this section if you don't want history.
        from app.core.storage import list_recent_scores  # imported lazily to avoid unused import drift
        rows = list_recent_scores(limit=20)  # expected list[dict]
        if not rows:
            st.caption("No saved scores yet.")
        else:
            # Render minimal, safe table
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.caption(f"Recent scores view unavailable: {e}")


# -------------------------
# Footer / diagnostics
# -------------------------
with st.expander("Diagnostics", expanded=False):
    st.write("App root:", str(ROOT))
    st.write("DB active:", True)
    st.write("Last job id:", st.session_state.get("last_job_id"))
    st.write("Last resume id:", st.session_state.get("last_resume_id"))
    st.write("Last model used:", st.session_state.get("last_model_used"))

st.caption("Executive Job Agent — stabilized main.py baseline")
