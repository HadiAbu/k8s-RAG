from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["What is a StatefulSet?"])
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    title: str
    url: str


class ChunkPreview(BaseModel):
    title: str
    url: str
    section: str
    snippet: str
    rerank_score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    retrieved_chunks: list[ChunkPreview] = Field(default_factory=list)


class ReindexResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    indexed: bool
    index_size: int
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    indexed_vectors: int
