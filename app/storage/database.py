import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from app.config import settings

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id      TEXT PRIMARY KEY,
    title   TEXT NOT NULL,
    url     TEXT,
    path    TEXT NOT NULL,
    section TEXT,
    version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    title       TEXT NOT NULL,
    url         TEXT,
    section     TEXT,
    content     TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER,
    faiss_id    INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc      ON chunks(document_id);
"""


def init_db(path: Path = settings.sqlite_path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_CREATE_SQL)


@asynccontextmanager
async def db_connection(path: Path = settings.sqlite_path):
    async with aiosqlite.connect(str(path)) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def get_chunks_by_faiss_ids(faiss_ids: list[int]) -> list[dict]:
    if not faiss_ids:
        return []
    placeholders = ",".join("?" * len(faiss_ids))
    async with db_connection() as db:
        async with db.execute(
            f"SELECT id, document_id, title, url, section, content, chunk_index, token_count, faiss_id "
            f"FROM chunks WHERE faiss_id IN ({placeholders})",
            faiss_ids,
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_stats() -> dict:
    async with db_connection() as db:
        async with db.execute("SELECT COUNT(*) FROM documents") as c:
            docs = (await c.fetchone())[0]
        async with db.execute("SELECT COUNT(*) FROM chunks") as c:
            chunks = (await c.fetchone())[0]
        async with db.execute("SELECT MAX(faiss_id) FROM chunks") as c:
            row = await c.fetchone()
            max_id = row[0] if row and row[0] is not None else -1
    return {
        "documents": docs,
        "chunks": chunks,
        "indexed_vectors": max_id + 1 if max_id >= 0 else 0,
    }
