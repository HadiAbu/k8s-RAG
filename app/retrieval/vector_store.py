import logging
from pathlib import Path

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_index: faiss.Index | None = None


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build a FAISS IndexFlatIP index (cosine sim via inner product on L2-normalised vecs)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS index: %d vectors, dim=%d", index.ntotal, dim)
    return index


def save_index(index: faiss.Index, path: Path = settings.faiss_index_path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("FAISS index saved → %s (%d vectors)", path, index.ntotal)


def load_index(path: Path = settings.faiss_index_path) -> faiss.Index:
    global _index
    if _index is None:
        _index = faiss.read_index(str(path))
        logger.info("FAISS index loaded ← %s (%d vectors)", path, _index.ntotal)
    return _index


def reset_index() -> None:
    """Force reload on next access (call after re-indexing)."""
    global _index
    _index = None


def search(
    query_embedding: np.ndarray,
    k: int = settings.top_k_initial,
    path: Path = settings.faiss_index_path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index.

    Returns (scores, faiss_ids) — both flat arrays of length k.
    Invalid results have faiss_id == -1.
    """
    index = load_index(path)
    scores, indices = index.search(query_embedding, k)
    return scores[0], indices[0]


def get_index_size(path: Path = settings.faiss_index_path) -> int:
    try:
        return load_index(path).ntotal
    except Exception:
        return 0
