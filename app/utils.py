import json
import re
from typing import Any, Dict, List


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def slugify_filename(value: str) -> str:
    value = safe_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "file"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_whitespace(text: str) -> str:
    text = safe_text(text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> List[str]:
    text = safe_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    text_l = safe_text(text).lower()
    hits = []
    for kw in keywords:
        if kw.lower() in text_l:
            hits.append(kw)
    return hits


def count_keyword_hits(text: str, keywords: List[str]) -> int:
    return len(keyword_hits(text, keywords))


def token_overlap_score(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9&/\-]+", safe_text(a).lower()))
    b_tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9&/\-]+", safe_text(b).lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    denom = len(b_tokens) if len(b_tokens) > 0 else 1
    return overlap / denom


def top_matching_lines(source_text: str, query_text: str, limit: int = 5) -> List[str]:
    lines = [x.strip() for x in safe_text(source_text).splitlines() if x.strip()]
    scored = []
    for line in lines:
        score = token_overlap_score(line, query_text)
        if score > 0:
            scored.append((score, line))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [line for _, line in scored[:limit]]


def pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def job_desc_mentions_salary(job_desc: str) -> bool:
    jd = safe_text(job_desc).lower()
    salary_terms = [
        "$",
        "salary",
        "base pay",
        "compensation",
        "pay range",
        "salary range",
        "target bonus",
        "annual bonus",
        "incentive",
    ]
    return any(term in jd for term in salary_terms)
