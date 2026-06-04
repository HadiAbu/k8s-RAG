import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_encode_query_returns_float32():
    with patch("app.embeddings.encoder._model") as mock_model:
        mock_model.encode.return_value = np.ones((1, 384), dtype=np.float32)
        from app.embeddings.encoder import encode_query
        result = encode_query("What is a Pod?")
        assert result.dtype == np.float32
        assert result.shape == (1, 384)


def test_rerank_returns_sorted():
    from app.retrieval.reranker import rerank

    chunks = [
        {"content": "A StatefulSet manages stateful apps.", "title": "StatefulSet", "url": "u1"},
        {"content": "A Deployment manages stateless apps.", "title": "Deployment", "url": "u2"},
        {"content": "A DaemonSet runs on all nodes.", "title": "DaemonSet", "url": "u3"},
    ]

    with patch("app.retrieval.reranker._reranker") as mock_reranker:
        mock_reranker.predict.return_value = np.array([0.9, 0.3, 0.1])
        result = rerank("What is a StatefulSet?", chunks, top_k=2)

    assert len(result) == 2
    assert result[0]["url"] == "u1"
    assert result[0]["rerank_score"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_retrieve_returns_empty_on_no_index():
    with (
        patch("app.retrieval.retriever.encode_query", return_value=np.zeros((1, 384), dtype=np.float32)),
        patch("app.retrieval.retriever.search", return_value=(np.array([0.5]), np.array([-1]))),
        patch("app.retrieval.retriever.get_chunks_by_faiss_ids", new_callable=AsyncMock, return_value=[]),
    ):
        from app.retrieval.retriever import retrieve
        result = await retrieve("What is a Pod?")
        assert result == []
