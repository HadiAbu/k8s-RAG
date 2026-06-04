# CLAUDE.md — k8s-RAG

Kubernetes Documentation RAG Assistant. Answers natural-language questions about Kubernetes using only official documentation — no model knowledge used for answers.

---

## Common commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download & index Kubernetes docs (first run: ~20–30 min)
python -m scripts.build_index

# Ask a question (CLI)
python cli.py ask "What is a StatefulSet?"
python cli.py ask "How do I configure liveness probes?" --show-chunks

# Start API server (http://localhost:8000/docs)
python cli.py serve
# or directly:
uvicorn app.api.main:app --reload

# Streamlit UI
streamlit run streamlit_app.py

# Run tests
pytest

# Show index statistics
python cli.py stats
```

## Docker (fully automatic)

```bash
# Start everything — Ollama + model pull + index build + API server
docker compose up

# Override the LLM model without editing any file
LLM_MODEL=phi4-mini docker compose up

# Use a GPU-capable Ollama image
EMBEDDING_DEVICE=cuda docker compose up

# Force a full re-index (e.g. after k8s docs update)
docker compose exec api python -m scripts.build_index

# Tail only the API logs
docker compose logs -f api
```

**First-run sequence (automatic, no user action required):**
1. `ollama` starts; healthcheck waits until `/api/tags` responds (~20 s)
2. `model-pull` pulls the configured model into Ollama (~5–15 min for qwen3:8b)
3. `api` entrypoint (`docker-entrypoint.sh`) detects missing index and runs the full build pipeline (~20–40 min, CPU)
4. uvicorn starts; `GET /health` returns `{"indexed": true}`

Subsequent `docker compose up` calls skip steps 2 and 3 — volumes (`ollama_models`, `k8s_docs`, `rag_index`) persist everything.

**Key Docker files:**
- `Dockerfile` — installs deps, pre-bakes ML models + tiktoken data, copies code
- `docker-entrypoint.sh` — waits for Ollama, builds index if missing, execs uvicorn
- `docker-compose.yml` — three services: `ollama`, `model-pull` (one-shot), `api`
- `.dockerignore` — excludes `.git`, `data/raw/`, `storage/`, `.env`, caches

---

## Architecture

```
User question
  → embed query (BAAI/bge-small-en-v1.5)
  → FAISS IndexFlatIP search → top-20 candidates
  → SQLite fetch chunk text + metadata
  → cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top-5
  → LLM with grounded prompt (Ollama / transformers)
  → answer + citations
```

## Repository layout

```
app/
  config.py              # All settings via K8S_RAG_* env vars
  ingestion/
    downloader.py        # git clone --depth=1 kubernetes/website
    parser.py            # Front matter + Hugo shortcode cleaning
  chunking/
    chunker.py           # Heading-aware, code-block-safe chunker (512 tok, 100 overlap)
  embeddings/
    encoder.py           # SentenceTransformer wrapper; BGE asymmetric query prefix
  retrieval/
    vector_store.py      # FAISS build / save / load / search
    reranker.py          # CrossEncoder reranker (lazy-loaded singleton)
    retriever.py         # Full retrieve() pipeline (async)
  generation/
    prompts.py           # SYSTEM_PROMPT + context formatter
    generator.py         # OpenAI-compatible (Ollama) or transformers pipeline
  storage/
    database.py          # SQLite (aiosqlite): documents + chunks tables
  evaluation/
    benchmark.py         # 100 benchmark questions across 9 categories
    metrics.py           # recall@k, MRR, precision@k, latency P95
  api/
    main.py              # FastAPI app + lifespan (init_db)
    routes.py            # POST /query, POST /reindex, GET /health, GET /stats
    schemas.py           # Pydantic request/response models

scripts/
  build_index.py         # Full pipeline: clone → parse → chunk → embed → FAISS + SQLite
  download_docs.py       # Clone/update docs only
  evaluate.py            # Benchmark harness (Click CLI)

tests/
  conftest.py            # SAMPLE_DOC fixture + tmp_db fixture
  test_chunking.py       # Unit tests for chunker (no ML deps)
  test_ingestion.py      # Unit tests for parser (no ML deps)
  test_retrieval.py      # Mocked retrieval pipeline tests
  test_api.py            # FastAPI endpoint tests (mocked retrieve + generate)

cli.py                   # Click CLI: ask / build-index / serve / stats
streamlit_app.py         # Streamlit UI (Phase 2)
Dockerfile
docker-compose.yml       # API + Ollama services
```

---

## Configuration

All settings live in `app/config.py` (pydantic-settings). Override via environment variables or `.env`:

| Variable | Default | Notes |
|----------|---------|-------|
| `K8S_RAG_LLM_BACKEND` | `openai_compatible` | `openai_compatible` or `transformers` |
| `K8S_RAG_LLM_MODEL` | `qwen3:8b` | Ollama tag or HF model ID |
| `K8S_RAG_LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama / vLLM endpoint |
| `K8S_RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any SentenceTransformer model |
| `K8S_RAG_EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `K8S_RAG_TOP_K_INITIAL` | `20` | FAISS candidates before reranking |
| `K8S_RAG_TOP_K_FINAL` | `5` | Chunks passed to LLM |
| `K8S_RAG_CHUNK_SIZE` | `512` | Target tokens per chunk |
| `K8S_RAG_CHUNK_OVERLAP` | `100` | Overlap tokens between chunks |

---

## Key design decisions

**Chunking** — `app/chunking/chunker.py`
- Splits first by markdown headings, then by paragraphs when a section exceeds `chunk_size`.
- Code blocks (``` fences) are treated as atomic units and never split across chunk boundaries.
- Each chunk is prefixed with `# {title}\n{heading}\n` so every chunk is self-contained for retrieval.

**Embeddings** — `app/embeddings/encoder.py`
- Documents are encoded without a prefix (standard BGE behaviour).
- Queries are encoded with the BGE asymmetric prefix: `"Represent this sentence for searching relevant passages: {query}"`.
- All vectors are L2-normalised (`normalize_embeddings=True`), so FAISS inner product == cosine similarity.

**Retrieval pipeline** — `app/retrieval/retriever.py`
- FAISS returns top-20; invalid indices (`-1`) are filtered before the SQLite lookup.
- The cross-encoder reranker is a lazy singleton — loaded only when first needed.

**LLM generator** — `app/generation/generator.py`
- The `openai_compatible` backend uses `openai.AsyncOpenAI` — works with Ollama, vLLM, LiteLLM.
- The `transformers` backend runs in a thread pool (`asyncio.run_in_executor`) to keep the event loop non-blocking.
- The system prompt explicitly forbids the LLM from using knowledge outside the supplied context.

**Storage**
- FAISS index: `storage/faiss/index.faiss`
- SQLite: `storage/sqlite/k8s_rag.db` — two tables: `documents` and `chunks`
- `chunks.faiss_id` is the row index in the FAISS array (0-based sequential).

---

## Adding a new phase

- **Phase 2 (Qdrant)**: replace `app/retrieval/vector_store.py` with a Qdrant client; the retriever interface (`search()` signature) stays the same.
- **Phase 9 (Hybrid BM25)**: add `app/retrieval/bm25.py`, blend BM25 + FAISS scores in `retriever.py` before the reranker step. `rank-bm25` is already in `requirements.txt`.
- **Phase 10 (Incremental updates)**: track the last-indexed git commit hash in a `index_metadata` SQLite table; in `build_index.py`, use `git diff --name-only` to find changed files and re-index only those.
