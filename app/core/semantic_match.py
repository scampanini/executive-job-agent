import math
import os
from typing import List, Optional, Tuple


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        dot += a[i] * b[i]
        na += a[i] * a[i]
        nb += b[i] * b[i]
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def semantic_enabled() -> bool:
    v = os.getenv("GROUND_SEMANTIC_MATCH", "0").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def try_embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Returns embeddings or None if OpenAI client/model isn't available.
    """
    if not texts:
        return []

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    try:
        client = OpenAI()
        model = os.getenv("GROUND_EMBED_MODEL", "text-embedding-3-small")
        resp = client.embeddings.create(model=model, input=texts)
        out: List[List[float]] = []
        for item in resp.data:
            out.append(list(item.embedding))
        return out
    except Exception:
        return None


def semantic_similarity(query: str, candidates: List[str]) -> Optional[List[Tuple[int, float]]]:
    """
    Returns list of (candidate_index, cosine_similarity) or None if embeddings unavailable.
    """
    embs = try_embed_texts([query] + candidates)
    if embs is None or len(embs) != (1 + len(candidates)):
        return None

    q = embs[0]
    sims: List[Tuple[int, float]] = []
    for i, e in enumerate(embs[1:]):
        sims.append((i, float(_cosine(q, e))))
    return sims
