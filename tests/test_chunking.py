from app.chunking.chunker import chunk_document, count_tokens
from tests.conftest import SAMPLE_DOC


def test_chunk_returns_list():
    chunks = chunk_document(SAMPLE_DOC)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_chunk_fields():
    chunks = chunk_document(SAMPLE_DOC)
    required = {"id", "document_id", "title", "url", "section", "content", "chunk_index", "token_count"}
    for chunk in chunks:
        assert required.issubset(chunk.keys())


def test_chunk_token_limit():
    from app.config import settings
    chunks = chunk_document(SAMPLE_DOC)
    for chunk in chunks:
        # Allow code blocks to slightly exceed max_chunk_size (they can't be split)
        assert chunk["token_count"] <= settings.max_chunk_size + 200


def test_code_block_not_split():
    yaml_block = "```yaml\napiVersion: apps/v1\nkind: StatefulSet\nmetadata:\n  name: web\nspec:\n  replicas: 3\n```"
    doc = {
        **SAMPLE_DOC,
        "content": f"Before the block.\n\n{yaml_block}\n\nAfter the block.",
    }
    chunks = chunk_document(doc)
    # The YAML block must appear intact in exactly one chunk
    found = sum(1 for c in chunks if yaml_block in c["content"])
    assert found == 1


def test_document_id_propagated():
    chunks = chunk_document(SAMPLE_DOC)
    for chunk in chunks:
        assert chunk["document_id"] == SAMPLE_DOC["id"]


def test_chunk_index_sequential():
    chunks = chunk_document(SAMPLE_DOC)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_large_document_splits():
    big_content = "## Section\n\n" + ("word " * 200 + "\n\n") * 10
    doc = {**SAMPLE_DOC, "content": big_content}
    chunks = chunk_document(doc)
    assert len(chunks) > 1


def test_count_tokens():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0
