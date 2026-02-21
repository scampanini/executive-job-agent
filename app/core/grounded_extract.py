import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# SVP/VP Corporate Comms lexicon (tuned to Verily-type JD)
# ----------------------------
TAG_LEXICON: Dict[str, List[str]] = {
    "corporate_narrative": ["corporate narrative", "company narrative", "positioning", "reputation", "brand trust"],
    "thought_leadership": ["thought leadership", "op-ed", "keynote", "panel", "speaker", "speaking", "byline"],
    "executive_comms": ["executive communications", "ceo", "cfo", "exec", "leadership team", "board", "executive presence"],
    "media_relations": ["media relations", "journalist", "press", "earned media", "pitch", "story pitch", "newsroom"],
    "product_comms": ["product communications", "launch", "milestone", "product", "platform", "solution", "gtm"],
    "internal_comms": ["internal communications", "employee communications", "all-hands", "town hall", "culture", "people & culture"],
    "financial_comms": ["financial communications", "earnings", "investor", "ir", "analyst", "guidance", "s-1", "ipo"],
    "crisis_issues": ["crisis", "issues management", "incident", "reputation risk", "rapid response", "war room"],
    "policy_public_affairs": ["public policy", "government affairs", "public affairs", "dc", "washington", "regulatory", "legislation"],
    "regulated_healthcare": ["fda", "hipaa", "clinical", "healthcare", "biotech", "life sciences", "medtech", "health tech", "pharma"],
    "measurement": ["measurement", "metrics", "kpi", "share of voice", "sentiment", "dashboard", "effectiveness"],
    "agency_management": ["agency", "pr agency", "agency relationship", "retainer", "vendor"],
    "partner_comms": ["partner", "partnership", "alliances", "customer", "stakeholder"],
    "writing_materials": ["press release", "blog", "q&a", "messaging", "talking points", "presentation", "speech", "guidelines"],
}

RE_METRICS = re.compile(r"(\b\d{1,3}%\b)|(\$\s?\d+(?:\.\d+)?\s?(?:k|m|b)\b)|(\b\d+(?:\.\d+)?\s?(?:k|m|b)\b)", re.IGNORECASE)
RE_TEAM = re.compile(r"\b(team of|managed|led)\s+(\d{1,4})\b", re.IGNORECASE)
RE_BUDGET = re.compile(r"\bbudget\s*(?:of)?\s*\$?\s*(\d+(?:\.\d+)?)\s*(k|m|b)?\b", re.IGNORECASE)
RE_YEARS = re.compile(r"\b(\d{1,2})\+?\s+years?\b", re.IGNORECASE)

SECTION_SPLIT = re.compile(r"\n\s*(?:experience|professional experience|leadership|education|skills|summary|highlights|accomplishments|projects|publications|speaking|press|media)\s*\n",
                           re.IGNORECASE)


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source_type: str  # "resume" | "portfolio"
    source_name: str
    section: str
    chunk_text: str
    tags: List[str]
    entities: Dict[str, Any]
    signals: Dict[str, Any]
    confidence: float


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def chunk_text(text: str, max_chars: int = 700, min_chars: int = 200) -> List[Tuple[str, str]]:
    """
    Returns list of (section, chunk).
    Section detection is heuristic; OK for personal use.
    """
    t = (text or "").strip()
    if not t:
        return []

    # Normalize newlines
    t = re.sub(r"\r\n?", "\n", t)

    # Split into rough sections
    parts = SECTION_SPLIT.split(t)
    # If SECTION_SPLIT doesn't match, we still chunk the whole text as one section
    if len(parts) == 1:
        sections = [("document", t)]
    else:
        # SECTION_SPLIT drops headings; treat all parts as generic sections
        sections = []
        for i, p in enumerate(parts):
            p = p.strip()
            if not p:
                continue
            sections.append((f"section_{i+1}", p))

    chunks: List[Tuple[str, str]] = []
    for section_name, sec_text in sections:
        # Prefer bullet-aware splitting
        lines = [ln.strip() for ln in sec_text.split("\n")]
        buf: List[str] = []
        buf_len = 0

        def flush():
            nonlocal buf, buf_len
            if not buf:
                return
            chunk = "\n".join(buf).strip()
            if chunk:
                chunks.append((section_name, chunk))
            buf = []
            buf_len = 0

        for ln in lines:
            if not ln:
                continue
            # If line is huge, flush buffer and slice line
            if len(ln) > max_chars:
                flush()
                for j in range(0, len(ln), max_chars):
                    piece = ln[j:j + max_chars].strip()
                    if piece:
                        chunks.append((section_name, piece))
                continue

            if buf_len + len(ln) + 1 > max_chars:
                flush()

            buf.append(ln)
            buf_len += len(ln) + 1

            # Flush if buffer is "big enough" and we just ended a bullet-ish line
            if buf_len >= min_chars and (ln.startswith(("-", "•", "*")) or ln.endswith(".")):
                flush()

        flush()

    # Deduplicate exact chunks
    seen = set()
    out: List[Tuple[str, str]] = []
    for sec, ch in chunks:
        h = _hash_text(sec + "\n" + ch)
        if h in seen:
            continue
        seen.add(h)
        out.append((sec, ch))
    return out


def tag_and_extract_signals(chunk: str) -> Tuple[List[str], Dict[str, Any], Dict[str, Any], float]:
    text = chunk or ""
    low = text.lower()

    tags: List[str] = []
    for tag, kws in TAG_LEXICON.items():
        for kw in kws:
            if kw.lower() in low:
                tags.append(tag)
                break

    # Simple entities/signals
    metrics = [m.group(0) for m in RE_METRICS.finditer(text)]
    team_size = None
    m_team = RE_TEAM.search(text)
    if m_team:
        try:
            team_size = int(m_team.group(2))
        except Exception:
            team_size = None

    budget = None
    m_budget = RE_BUDGET.search(text)
    if m_budget:
        num = m_budget.group(1)
        suffix = (m_budget.group(2) or "").lower()
        try:
            val = float(num)
            if suffix == "k":
                val *= 1_000
            elif suffix == "m":
                val *= 1_000_000
            elif suffix == "b":
                val *= 1_000_000_000
            budget = val
        except Exception:
            budget = None

    years = None
    m_years = RE_YEARS.search(text)
    if m_years:
        try:
            years = int(m_years.group(1))
        except Exception:
            years = None

    entities: Dict[str, Any] = {
        "metrics": metrics[:10],
    }
    signals: Dict[str, Any] = {
        "scope": {
            "team_size": team_size,
            "budget": budget,
        },
        "seniority": {
            "years": years,
            "mentions_ceo": ("ceo" in low),
            "mentions_cmo": ("cmo" in low),
            "mentions_board": ("board" in low),
        },
    }

    # Confidence is heuristic: more tags + metrics => higher
    confidence = 0.35
    confidence += min(0.35, 0.08 * len(tags))
    confidence += 0.10 if metrics else 0.0
    confidence += 0.10 if team_size is not None else 0.0
    confidence = max(0.0, min(1.0, confidence))

    # Keep tags unique + stable
    tags = sorted(set(tags))
    return tags, entities, signals, confidence


def extract_requirements_deterministic(job_desc: str) -> List[Dict[str, Any]]:
    """
    Deterministic extraction: pulls bullet responsibilities and qualifications into structured items.
    """
    jd = (job_desc or "").strip()
    if not jd:
        return []

    jd = re.sub(r"\r\n?", "\n", jd)

    lines = [ln.strip() for ln in jd.split("\n") if ln.strip()]
    reqs: List[Dict[str, Any]] = []
    req_id = 1

    mode = "general"
    for ln in lines:
        l = ln.lower()
        if "responsibilit" in l:
            mode = "responsibility"
            continue
        if "qualif" in l:
            mode = "qualification"
            continue
        if "preferred" in l:
            mode = "preferred"
            continue

        if ln.startswith(("●", "-", "•", "*")):
            text = ln.lstrip("●-•* ").strip()
            if not text:
                continue

            must_have = mode in ("qualification", "responsibility")
            weight = 3 if mode in ("qualification", "responsibility") else 1

            # competency guess
            tags, _, _, _ = tag_and_extract_signals(text)
            competency = tags[0] if tags else "general"

            reqs.append(
                {
                    "requirement_id": f"REQ-{req_id:03d}",
                    "category": "required" if mode in ("qualification", "responsibility") else "preferred",
                    "competency": competency,
                    "text": text,
                    "weight": weight,
                    "must_have": must_have,
                }
            )
            req_id += 1

    return reqs


def upsert_evidence_chunks(
    conn: sqlite3.Connection,
    resume_id: int,
    job_id: Optional[int],
    source_type: str,
    source_name: str,
    section: str,
    chunks: Iterable[str],
) -> None:
    cur = conn.cursor()

    for ch in chunks:
        ch = (ch or "").strip()
        if not ch:
            continue

        tags, entities, signals, conf = tag_and_extract_signals(ch)
        content_hash = _hash_text(ch)

        cur.execute(
            """
            INSERT OR IGNORE INTO evidence_chunks
              (resume_id, job_id, source_type, source_name, section, chunk_text,
               tags_json, entities_json, signals_json, confidence, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                resume_id,
                job_id,
                source_type,
                source_name,
                section,
                ch,
                _safe_json(tags),
                _safe_json(entities),
                _safe_json(signals),
                float(conf),
                content_hash,
            ),
        )

    conn.commit()


def load_evidence_index(
    conn: sqlite3.Connection,
    resume_id: int,
    job_id: Optional[int],
    limit: int = 5000,
) -> List[EvidenceItem]:
    cur = conn.cursor()
    if job_id is None:
        cur.execute(
            """
            SELECT * FROM evidence_chunks
            WHERE resume_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (resume_id, limit),
        )
    else:
        cur.execute(
            """
            SELECT * FROM evidence_chunks
            WHERE resume_id = ? AND (job_id = ? OR job_id IS NULL)
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (resume_id, job_id, limit),
        )

    rows = cur.fetchall()
    out: List[EvidenceItem] = []
    for r in rows:
        tags = json.loads(r["tags_json"]) if r["tags_json"] else []
        entities = json.loads(r["entities_json"]) if r["entities_json"] else {}
        signals = json.loads(r["signals_json"]) if r["signals_json"] else {}
        out.append(
            EvidenceItem(
                evidence_id=f"E-{int(r['id']):06d}",
                source_type=str(r["source_type"]),
                source_name=str(r["source_name"]),
                section=str(r["section"] or ""),
                chunk_text=str(r["chunk_text"]),
                tags=list(tags) if isinstance(tags, list) else [],
                entities=dict(entities) if isinstance(entities, dict) else {},
                signals=dict(signals) if isinstance(signals, dict) else {},
                confidence=float(r["confidence"] or 0.0),
            )
        )
    return out
