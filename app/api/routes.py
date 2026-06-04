import logging
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.api.schemas import (
    ChunkPreview,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
    Source,
    StatsResponse,
)
from app.generation.generator import generate_answer
from app.retrieval.retriever import retrieve
from app.retrieval.vector_store import get_index_size
from app.storage.database import get_stats

logger = logging.getLogger(__name__)
router = APIRouter()

_NO_ANSWER = (
    "I could not find sufficient information in the indexed Kubernetes documentation "
    "to answer this question."
)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.perf_counter()

    chunks = await retrieve(request.question, top_k=request.top_k)

    if not chunks:
        return QueryResponse(
            question=request.question,
            answer=_NO_ANSWER,
            sources=[],
            retrieved_chunks=[],
        )

    result = await generate_answer(request.question, chunks)
    elapsed = time.perf_counter() - start
    logger.info("query answered in %.2fs (chunks=%d)", elapsed, len(chunks))

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        retrieved_chunks=[
            ChunkPreview(
                title=c["title"],
                url=c["url"],
                section=c.get("section", ""),
                snippet=c["content"][:400] + ("…" if len(c["content"]) > 400 else ""),
                rerank_score=c.get("rerank_score"),
            )
            for c in chunks
        ],
    )


@router.post("/reindex", response_model=ReindexResponse)
async def reindex(background_tasks: BackgroundTasks):
    from scripts.build_index import build_index_pipeline

    background_tasks.add_task(build_index_pipeline)
    return ReindexResponse(
        status="started",
        message="Re-indexing started in background. Poll /stats to monitor progress.",
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    size = get_index_size()
    return HealthResponse(status="ok", indexed=size > 0, index_size=size)


@router.get("/stats", response_model=StatsResponse)
async def stats():
    return StatsResponse(**(await get_stats()))
