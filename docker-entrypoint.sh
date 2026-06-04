#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# docker-entrypoint.sh
#
# Runs inside the container as PID 2 (tini is PID 1).
# Steps:
#   1. Wait for the Ollama LLM backend to be reachable.
#   2. Build the RAG index if it does not exist yet.
#   3. Hand off to uvicorn (exec — replaces this shell so signals work).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

FAISS_INDEX="storage/faiss/index.faiss"
SQLITE_DB="storage/sqlite/k8s_rag.db"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

banner() { echo -e "${CYAN}━━━ $* ━━━${NC}"; }
ok()     { echo -e "${GREEN}✔ $*${NC}"; }
warn()   { echo -e "${YELLOW}⚠ $*${NC}"; }
err()    { echo -e "${RED}✘ $*${NC}" >&2; }

# ── 1. Wait for LLM backend ───────────────────────────────────────────────────
# The API itself doesn't need the LLM to start, but we want it ready so the
# very first query doesn't time out while the model is still loading.
LLM_BACKEND="${K8S_RAG_LLM_BACKEND:-openai_compatible}"

if [ "$LLM_BACKEND" = "openai_compatible" ]; then
    # Derive the Ollama health URL from the configured base URL.
    # e.g.  http://ollama:11434/v1  →  http://ollama:11434/api/tags
    BASE_URL="${K8S_RAG_LLM_BASE_URL:-http://localhost:11434/v1}"
    OLLAMA_ROOT="${BASE_URL%/v1*}"
    HEALTH_URL="${OLLAMA_ROOT}/api/tags"

    banner "Waiting for LLM backend at ${OLLAMA_ROOT}"
    MAX_WAIT=120
    ELAPSED=0
    until curl -sf "${HEALTH_URL}" > /dev/null 2>&1; do
        if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
            warn "LLM backend not reachable after ${MAX_WAIT}s — continuing anyway."
            break
        fi
        echo "  not ready yet, retrying in 5 s …"
        sleep 5
        ELAPSED=$((ELAPSED + 5))
    done
    if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
        ok "LLM backend is ready."
    fi
fi

# ── 2. Auto-build the RAG index on first run ──────────────────────────────────
if [ ! -f "$FAISS_INDEX" ] || [ ! -f "$SQLITE_DB" ]; then
    echo ""
    banner "First-run index build"
    echo -e "${YELLOW}"
    echo "  No index found.  Building from official Kubernetes documentation."
    echo "  This clones kubernetes/website (~1 GB) and runs the full"
    echo "  ingestion → chunking → embedding pipeline."
    echo "  Estimated time: 20–40 min (CPU, no GPU)."
    echo -e "${NC}"

    if python -m scripts.build_index; then
        ok "Index build complete."
    else
        err "Index build failed — check logs above."
        err "The API will start but /query will return no results."
        err "Fix the error and run:  docker compose exec api python -m scripts.build_index"
    fi
else
    ok "Index found — skipping build."
fi

echo ""

# ── 3. Start the API server ───────────────────────────────────────────────────
HOST="${K8S_RAG_API_HOST:-0.0.0.0}"
PORT="${K8S_RAG_API_PORT:-8000}"

banner "Starting Kubernetes RAG API on ${HOST}:${PORT}"
exec uvicorn app.api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers 1
