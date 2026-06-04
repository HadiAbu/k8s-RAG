# Kubernetes Documentation RAG Assistant

A production-grade Retrieval-Augmented Generation system that answers Kubernetes questions using only official documentation — no hallucinations, always cited.

---

## Architecture

```
User Question
    ↓
FastAPI  /query
    ↓
Embed query (BAAI/bge-small-en-v1.5)
    ↓
FAISS similarity search → top-20 candidates
    ↓
SQLite — fetch chunk text + metadata
    ↓
Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top-5
    ↓
LLM (Ollama / transformers) with grounded prompt
    ↓
Answer + citations
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
# Edit .env to set your LLM model, backend, device, etc.
```

### 3. Start Ollama (recommended LLM backend)

```bash
ollama pull qwen3:8b
ollama serve
```

### 4. Download and index Kubernetes docs

This clones `kubernetes/website` and builds the FAISS + SQLite index (~15–30 min first run):

```bash
python -m scripts.build_index
```

Or via the CLI:

```bash
python cli.py build-index
```

### 5. Ask questions

**CLI:**
```bash
python cli.py ask "What is a StatefulSet?"
python cli.py ask "How do I configure liveness probes?" --show-chunks
```

**API server:**
```bash
python cli.py serve
# → http://localhost:8000/docs  (Swagger UI)
```

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain taints and tolerations"}' | jq .
```

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

---

## Docker Compose

Starts the API + Ollama in one command.  
After starting, run the index build inside the container:

```bash
docker compose up -d
docker compose exec api python -m scripts.build_index
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Ask a question; returns answer + sources |
| `POST` | `/reindex` | Trigger background re-index |
| `GET` | `/health` | Liveness check + index size |
| `GET` | `/stats` | Document / chunk / vector counts |

### `POST /query` example

```json
{
  "question": "How does kube-scheduler work?",
  "top_k": 5
}
```

Response:
```json
{
  "question": "...",
  "answer": "...",
  "sources": [{"title": "...", "url": "..."}],
  "retrieved_chunks": [{"title": "...", "snippet": "...", "rerank_score": 0.92}]
}
```

---

## Configuration

All settings are controlled via environment variables prefixed `K8S_RAG_` or a `.env` file.

| Variable | Default | Description |
|----------|---------|-------------|
| `K8S_RAG_LLM_BACKEND` | `openai_compatible` | `openai_compatible` or `transformers` |
| `K8S_RAG_LLM_MODEL` | `qwen3:8b` | Ollama model tag or HF model ID |
| `K8S_RAG_LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `K8S_RAG_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | SentenceTransformer model |
| `K8S_RAG_EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `K8S_RAG_TOP_K_INITIAL` | `20` | FAISS candidates before reranking |
| `K8S_RAG_TOP_K_FINAL` | `5` | Chunks passed to LLM after reranking |

---

## Evaluation

Run the 100-question benchmark:

```bash
python -m scripts.evaluate --k 5 --output results.json

# Filter by category
python -m scripts.evaluate --category concepts
```

Categories: `concepts`, `networking`, `storage`, `security`, `scheduling`, `controllers`, `kubectl`, `api_objects`, `troubleshooting`.

---

## Repository Structure

```
k8s-RAG/
├── app/
│   ├── config.py              # Pydantic-settings configuration
│   ├── api/                   # FastAPI routes + schemas
│   ├── chunking/              # Token-aware chunker (preserves code blocks)
│   ├── embeddings/            # BAAI/bge encoder
│   ├── evaluation/            # 100-question benchmark + metrics
│   ├── generation/            # Prompt templates + LLM generator
│   ├── ingestion/             # k8s website cloner + markdown parser
│   ├── retrieval/             # FAISS vector store + cross-encoder reranker
│   └── storage/               # SQLite (documents + chunks)
├── scripts/
│   ├── build_index.py         # End-to-end index build pipeline
│   ├── download_docs.py       # Clone / update k8s docs only
│   └── evaluate.py            # Evaluation harness
├── tests/                     # pytest test suite
├── cli.py                     # Click CLI
├── streamlit_app.py           # Streamlit UI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers) |
| Vector DB | FAISS IndexFlatIP |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM (default) | Ollama qwen3:8b (OpenAI-compatible) |
| LLM (alt) | Any HuggingFace transformers model |
| API | FastAPI + uvicorn |
| Storage | SQLite (aiosqlite) |
| CLI | Click + Rich |
| UI | Streamlit |
| Containers | Docker + Docker Compose |
