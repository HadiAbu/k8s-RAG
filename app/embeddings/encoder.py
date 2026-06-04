import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None


def get_encoder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )
    return _model


def encode_texts(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    model = get_encoder()
    embeddings = model.encode(
        texts,
        batch_size=batch_size or settings.embedding_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """Encode a query with the asymmetric BGE prefix when applicable."""
    model = get_encoder()
    if "bge" in settings.embedding_model.lower():
        query = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.astype(np.float32)
