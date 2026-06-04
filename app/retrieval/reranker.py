import logging

from sentence_transformers import CrossEncoder

from app.config import settings

logger = logging.getLogger(__name__)

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
        logger.info("Reranker loaded: %s", settings.reranker_model)
    return _reranker


def rerank(query: str, chunks: list[dict], top_k: int = settings.top_k_final) -> list[dict]:
    """Score (query, chunk) pairs with a cross-encoder and return the top_k."""
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(query, c["content"]) for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    result = []
    for score, chunk in ranked[:top_k]:
        c = chunk.copy()
        c["rerank_score"] = float(score)
        result.append(c)
    return result
