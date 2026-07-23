import numpy as np
import re
import unicodedata
from typing import List, Dict, Any


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text or "")).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize_text(text).lower())


def compute_cosine_similarity(query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    query_vec = np.asarray(query_vec, dtype=float)
    chunk_vecs = np.asarray(chunk_vecs, dtype=float)

    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    if chunk_vecs.ndim == 1:
        chunk_vecs = chunk_vecs.reshape(1, -1)

    dot_product = np.dot(chunk_vecs, query_vec.T).reshape(-1)

    query_norm = float(np.linalg.norm(query_vec))
    chunk_norms = np.linalg.norm(chunk_vecs, axis=1)

    denom = chunk_norms * query_norm
    scores = np.divide(
        dot_product,
        denom,
        out=np.zeros_like(dot_product, dtype=float),
        where=denom != 0,
    )

    return np.clip(scores, -1.0, 1.0)



def retrieve_top_k(
    query_vec: np.ndarray,
    chunk_vecs: np.ndarray,
    chunks: List[Dict[str, Any]],
    top_k: int = 3,
    threshold: float = 0.25,
    query_text: str = ""
) -> Dict[str, Any]:
    """
    Retrieves the top-k most similar chunks for a query using cosine similarity.
    """
    if len(chunks) == 0 or chunk_vecs.size == 0:
        return {
            "results": [],
            "is_relevant": False,
            "max_score": 0.0
        }

    scores = compute_cosine_similarity(query_vec, chunk_vecs)

    # Convert cosine scores from [-1, 1] to [0, 1] for combination with
    # simple keyword overlap signals.
    cos01 = np.clip((scores + 1.0) / 2.0, 0.0, 1.0)

    # Compute a lightweight keyword-overlap score (query tokens present in
    # the chunk) to boost exact-match relevance. This helps when the
    # semantic embedding places a related paragraph slightly higher than the
    # exact-definition paragraph; small boost to exact matches improves
    # retrieval for definition-like queries.
    kw_weight = 0.20
    keyword_scores = np.zeros_like(cos01)
    qt = _normalize_text(query_text)
    if qt:
        q_tokens = _tokenize(qt)
        q_token_set = set(q_tokens)
        for i, c in enumerate(chunks):
            c_text = c.get("text", "")
            c_tokens = _tokenize(c_text)
            if not q_tokens:
                keyword_scores[i] = 0.0
            else:
                inter = q_token_set.intersection(set(c_tokens))
                # Normalise by number of query tokens so short queries get
                # stronger per-token influence.
                keyword_scores[i] = len(inter) / max(1, len(q_tokens))

    # If no query_text was provided (e.g., tests or programmatic calls), do
    # not apply the keyword weight so we preserve original cosine-derived
    # ranking behavior. When a query string exists, apply a small keyword
    # boost to favor exact matches.
    if not qt:
        kw_weight = 0.0
    final_scores = (1.0 - kw_weight) * cos01 + kw_weight * keyword_scores
    scores_to_rank = final_scores

    # Rank scores in DESCENDING order so highest-similarity chunks come first.
    ranked_indices = np.argsort(-scores_to_rank, kind="mergesort")

    results = []
    seen_texts = set()
    for idx in ranked_indices:
        normalized_text = _normalize_text(chunks[idx].get("text", "")).lower()
        if normalized_text in seen_texts:
            continue
        seen_texts.add(normalized_text)
        score = float(scores_to_rank[idx])
        results.append({
            "chunk": chunks[idx],
            "score": score
        })
        if len(results) >= min(top_k, len(chunks)):
            break

    max_score = float(scores_to_rank[ranked_indices[0]]) if len(scores_to_rank) > 0 else 0.0
    is_relevant = max_score >= threshold

    return {
        "results": results,
        "is_relevant": is_relevant,
        "max_score": max_score
    }
