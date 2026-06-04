import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

from app.api.main import app

_MOCK_CHUNKS = [
    {
        "id": "c1",
        "document_id": "d1",
        "title": "StatefulSets",
        "url": "https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/",
        "section": "Concepts",
        "content": "A StatefulSet manages the deployment of stateful applications.",
        "chunk_index": 0,
        "token_count": 15,
        "faiss_id": 0,
        "rerank_score": 0.95,
        "vector_score": 0.88,
    }
]

_MOCK_ANSWER = {
    "answer": "A StatefulSet manages stateful applications with stable identities.",
    "sources": [
        {
            "title": "StatefulSets",
            "url": "https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/",
        }
    ],
}


@pytest.fixture()
def transport():
    return ASGITransport(app=app)


@pytest.mark.asyncio
async def test_health(transport):
    with patch("app.api.routes.get_index_size", return_value=1234):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["index_size"] == 1234


@pytest.mark.asyncio
async def test_query_returns_answer(transport):
    with (
        patch("app.api.routes.retrieve", new_callable=AsyncMock, return_value=_MOCK_CHUNKS),
        patch("app.api.routes.generate_answer", new_callable=AsyncMock, return_value=_MOCK_ANSWER),
    ):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"question": "What is a StatefulSet?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "StatefulSet" in data["answer"]
    assert len(data["sources"]) == 1
    assert len(data["retrieved_chunks"]) == 1


@pytest.mark.asyncio
async def test_query_no_chunks_returns_fallback(transport):
    with patch("app.api.routes.retrieve", new_callable=AsyncMock, return_value=[]):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/query", json={"question": "What is a StatefulSet?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "could not find" in data["answer"].lower()


@pytest.mark.asyncio
async def test_query_validation(transport):
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/query", json={"question": "ab"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_stats(transport):
    mock_stats = {"documents": 500, "chunks": 4000, "indexed_vectors": 4000}
    with patch("app.api.routes.get_stats", new_callable=AsyncMock, return_value=mock_stats):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents"] == 500
