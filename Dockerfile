# ─────────────────────────────────────────────────────────────────────────────
# Kubernetes RAG Assistant
#
# Build args (override at build time with --build-arg):
#   EMBEDDING_MODEL  — HuggingFace model ID for embeddings
#   RERANKER_MODEL   — HuggingFace model ID for cross-encoder reranking
#
# What happens automatically at container start (docker-entrypoint.sh):
#   1. If storage/faiss/index.faiss is missing → clone kubernetes/website
#      and run the full ingestion → chunking → embedding → indexing pipeline.
#   2. Start the FastAPI server.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System packages ───────────────────────────────────────────────────────────
# tini:  PID-1 init — forwards signals and reaps zombie processes
# git:   needed by the index-build pipeline to clone kubernetes/website
# curl:  used by the entrypoint health-wait loop
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini git curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Installed as root so they land on the system Python (no --user path issues).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Directory layout + ownership ──────────────────────────────────────────────
# data/raw      → kubernetes/website clone (populated at runtime via volume)
# storage/faiss → FAISS index file
# storage/sqlite→ SQLite database
# .cache        → HuggingFace / sentence-transformers model cache (baked in)
RUN mkdir -p data/raw storage/faiss storage/sqlite .cache \
    && chown -R appuser:appuser /app

USER appuser

# ── Model cache environment ───────────────────────────────────────────────────
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# ── Pre-download ML models & tokenizer data ───────────────────────────────────
# Baking models into the image means no surprise downloads on first query.
# Change these ARGs at build time to switch models without editing the file.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

RUN DOWNLOAD_EMBEDDING=${EMBEDDING_MODEL} DOWNLOAD_RERANKER=${RERANKER_MODEL} \
    python -c "\
import os; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
import tiktoken; \
print('Downloading embedding model:', os.environ['DOWNLOAD_EMBEDDING']); \
SentenceTransformer(os.environ['DOWNLOAD_EMBEDDING']); \
print('Downloading reranker model:', os.environ['DOWNLOAD_RERANKER']); \
CrossEncoder(os.environ['DOWNLOAD_RERANKER']); \
print('Downloading tiktoken cl100k_base ...'); \
tiktoken.get_encoding('cl100k_base'); \
print('All models cached.')"

# ── Application code ──────────────────────────────────────────────────────────
# Copied last so code changes only invalidate this thin layer,
# not the expensive model-download layer above.
COPY --chown=appuser:appuser . .

RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

# tini as PID 1 → entrypoint.sh → exec uvicorn
ENTRYPOINT ["/usr/bin/tini", "--", "/app/docker-entrypoint.sh"]
