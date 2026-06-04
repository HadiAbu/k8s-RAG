"""
Build the full RAG index from scratch.

Pipeline:
  1. Clone / update kubernetes/website
  2. Parse all markdown documents
  3. Chunk documents (preserving code blocks)
  4. Generate embeddings (BAAI/bge-small-en-v1.5)
  5. Build FAISS IndexFlatIP
  6. Persist: FAISS index + SQLite (documents + chunks)

Usage:
    python -m scripts.build_index
"""
import asyncio
import logging
from pathlib import Path

import aiosqlite
import numpy as np
from tqdm import tqdm

from app.chunking.chunker import chunk_document
from app.config import settings
from app.embeddings.encoder import encode_texts
from app.ingestion.downloader import clone_or_update_repo, get_docs_root, iter_doc_files
from app.ingestion.parser import parse_document
from app.retrieval.vector_store import build_index, reset_index, save_index
from app.storage.database import init_db

logger = logging.getLogger(__name__)


async def build_index_pipeline() -> None:
    logger.info("═══ Kubernetes RAG — Index Build ═══")

    # ── 1. Source ─────────────────────────────────────────────────────────────
    repo_root = clone_or_update_repo()
    docs_root = get_docs_root(repo_root)
    logger.info("Docs root: %s", docs_root)

    # ── 2. Parse ──────────────────────────────────────────────────────────────
    logger.info("Parsing documents …")
    all_files = list(iter_doc_files(docs_root))
    documents: list[dict] = []
    for f in tqdm(all_files, desc="Parsing", unit="file"):
        doc = parse_document(f, docs_root)
        if doc:
            documents.append(doc)
    logger.info("Parsed %d documents (skipped %d)", len(documents), len(all_files) - len(documents))

    if not documents:
        logger.error("No documents parsed. Aborting.")
        return

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    logger.info("Chunking …")
    all_chunks: list[dict] = []
    for doc in tqdm(documents, desc="Chunking", unit="doc"):
        all_chunks.extend(chunk_document(doc))
    logger.info("Created %d chunks", len(all_chunks))

    # ── 4. Embed ──────────────────────────────────────────────────────────────
    logger.info("Generating embeddings with %s …", settings.embedding_model)
    texts = [c["content"] for c in all_chunks]
    embeddings: np.ndarray = encode_texts(texts)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # ── 5. Assign FAISS IDs ───────────────────────────────────────────────────
    for i, chunk in enumerate(all_chunks):
        chunk["faiss_id"] = i

    # ── 6. Build & save FAISS index ───────────────────────────────────────────
    logger.info("Building FAISS index …")
    index = build_index(embeddings)
    save_index(index)
    reset_index()

    # ── 7. Persist to SQLite ──────────────────────────────────────────────────
    logger.info("Writing to SQLite …")
    init_db()
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(str(settings.sqlite_path)) as db:
        await db.execute("DELETE FROM chunks")
        await db.execute("DELETE FROM documents")
        await db.commit()

        doc_rows = [
            (d["id"], d["title"], d["url"], d["path"], d["section"], d.get("version", ""))
            for d in documents
        ]
        await db.executemany(
            "INSERT OR REPLACE INTO documents (id, title, url, path, section, version) "
            "VALUES (?,?,?,?,?,?)",
            doc_rows,
        )

        chunk_rows = [
            (
                c["id"], c["document_id"], c["title"], c["url"], c["section"],
                c["content"], c["chunk_index"], c["token_count"], c["faiss_id"],
            )
            for c in all_chunks
        ]
        # Insert in batches to avoid SQLite variable limits
        batch = 500
        for start in tqdm(range(0, len(chunk_rows), batch), desc="Inserting chunks"):
            await db.executemany(
                "INSERT OR REPLACE INTO chunks "
                "(id, document_id, title, url, section, content, chunk_index, token_count, faiss_id) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                chunk_rows[start : start + batch],
            )
        await db.commit()

    logger.info(
        "Index build complete. documents=%d  chunks=%d  vectors=%d",
        len(documents),
        len(all_chunks),
        index.ntotal,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    asyncio.run(build_index_pipeline())
