import hashlib
import re
from typing import Iterator

import tiktoken

from app.config import settings

_enc = tiktoken.get_encoding("cl100k_base")

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
_BLANK_RE = re.compile(r"\n\n+")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ── Heading splitter ──────────────────────────────────────────────────────────

def _split_on_headings(content: str) -> list[tuple[str, str]]:
    """Return list of (heading_line, body_text) pairs."""
    sections: list[tuple[str, str]] = []
    last_pos = 0
    last_heading = ""

    for m in _HEADING_RE.finditer(content):
        body = content[last_pos : m.start()].strip()
        if body:
            sections.append((last_heading, body))
        last_heading = m.group(1)
        last_pos = m.end()

    tail = content[last_pos:].strip()
    if tail:
        sections.append((last_heading, tail))

    return sections or [("", content)]


# ── Segment tokeniser (preserves code blocks) ─────────────────────────────────

def _iter_segments(text: str) -> Iterator[tuple[str, str]]:
    """Yield ("code"|"text", segment_string) preserving code blocks intact."""
    last = 0
    for m in _CODE_BLOCK_RE.finditer(text):
        before = text[last : m.start()]
        for para in _BLANK_RE.split(before):
            para = para.strip()
            if para:
                yield "text", para
        yield "code", m.group(0)
        last = m.end()
    for para in _BLANK_RE.split(text[last:]):
        para = para.strip()
        if para:
            yield "text", para


# ── Large-section splitter ────────────────────────────────────────────────────

def _split_section(body: str, context_prefix: str) -> list[str]:
    """Split *body* into overlapping chunks, never splitting code blocks."""
    segments = list(_iter_segments(body))
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens: int = count_tokens(context_prefix)

    def flush():
        if current_parts:
            text = context_prefix + "\n\n" + "\n\n".join(current_parts)
            chunks.append(text.strip())

    for _, seg in segments:
        seg_tokens = count_tokens(seg)

        if current_tokens + seg_tokens > settings.chunk_size and current_parts:
            flush()
            # Overlap: carry last N tokens worth of parts into next chunk
            overlap_parts: list[str] = []
            overlap_tok = 0
            for part in reversed(current_parts):
                t = count_tokens(part)
                if overlap_tok + t > settings.chunk_overlap:
                    break
                overlap_parts.append(part)
                overlap_tok += t
            current_parts = list(reversed(overlap_parts))
            current_tokens = count_tokens(context_prefix) + overlap_tok

        current_parts.append(seg)
        current_tokens += seg_tokens

    flush()
    return chunks or [context_prefix + "\n\n" + body]


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_document(doc: dict) -> list[dict]:
    title = doc["title"]
    url = doc["url"]
    doc_id = doc["id"]
    section = doc["section"]
    content = doc["content"]

    heading_sections = _split_on_headings(content)
    chunks: list[dict] = []
    idx = 0

    for heading, body in heading_sections:
        prefix = f"# {title}\n{heading}\n" if heading else f"# {title}\n"
        full = prefix + "\n" + body

        if count_tokens(full) <= settings.chunk_size:
            chunks.append(_make(idx, doc_id, title, url, section, full.strip()))
            idx += 1
        else:
            for sub in _split_section(body, prefix):
                chunks.append(_make(idx, doc_id, title, url, section, sub))
                idx += 1

    return chunks


def _make(idx: int, doc_id: str, title: str, url: str, section: str, content: str) -> dict:
    chunk_id = hashlib.sha256(f"{doc_id}:{idx}".encode()).hexdigest()[:16]
    return {
        "id": chunk_id,
        "document_id": doc_id,
        "title": title,
        "url": url,
        "section": section,
        "content": content,
        "chunk_index": idx,
        "token_count": count_tokens(content),
        "faiss_id": None,
    }
