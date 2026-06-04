import logging

from app.config import settings
from app.embeddings.encoder import encode_query
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import search
from app.storage.database import get_chunks_by_faiss_ids

logger = logging.getLogger(__name__)


async def retrieve(query: str, top_k: int = settings.top_k_final) -> list[dict]:
    """
    Full retrieval pipeline:
      1. Embed query
      2. FAISS similarity search → top_k_initial candidates
      3. Fetch chunk text from SQLite
      4. Cross-encoder reranking → top_k
    """
    query_vec = encode_query(query)

    scores, faiss_ids = search(query_vec, k=settings.top_k_initial)

    valid_ids = [int(i) for i in faiss_ids if i >= 0]
    if not valid_ids:
        logger.warning("No valid FAISS results for query: %s", query[:80])
        return []

    score_map = {int(i): float(s) for s, i in zip(scores, faiss_ids) if i >= 0}

    chunks = await get_chunks_by_faiss_ids(valid_ids)
    if not chunks:
        return []

    for chunk in chunks:
        chunk["vector_score"] = score_map.get(chunk["faiss_id"], 0.0)

    return rerank(query, chunks, top_k=top_k)
