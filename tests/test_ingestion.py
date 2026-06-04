import textwrap
from pathlib import Path

import pytest


def _write_md(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def test_parse_basic_document(tmp_path):
    from app.ingestion.parser import parse_document

    md = _write_md(
        tmp_path,
        "statefulset.md",
        textwrap.dedent("""\
            ---
            title: StatefulSets
            ---
            A StatefulSet manages Pods with stable identities.

            ## Use Cases

            Use StatefulSets for databases and message queues.
        """),
    )
    doc = parse_document(md, tmp_path)
    assert doc is not None
    assert doc["title"] == "StatefulSets"
    assert "StatefulSet" in doc["content"]


def test_parse_removes_shortcodes(tmp_path):
    from app.ingestion.parser import parse_document

    md = _write_md(
        tmp_path,
        "notes.md",
        textwrap.dedent("""\
            ---
            title: Notes
            ---
            Some text.

            {{< note >}}
            This is a note.
            {{< /note >}}

            More text.
        """),
    )
    doc = parse_document(md, tmp_path)
    assert doc is not None
    assert "{{<" not in doc["content"]
    assert "This is a note." in doc["content"]


def test_parse_skips_empty_file(tmp_path):
    from app.ingestion.parser import parse_document

    md = _write_md(tmp_path, "empty.md", "---\ntitle: Empty\n---\n")
    doc = parse_document(md, tmp_path)
    assert doc is None


def test_file_to_url(tmp_path):
    from app.ingestion.parser import _file_to_url

    docs_root = tmp_path
    f = tmp_path / "concepts" / "workloads" / "statefulset.md"
    f.parent.mkdir(parents=True)
    url = _file_to_url(f, docs_root)
    assert url == "https://kubernetes.io/docs/concepts/workloads/statefulset/"


def test_index_file_url(tmp_path):
    from app.ingestion.parser import _file_to_url

    docs_root = tmp_path
    f = tmp_path / "concepts" / "_index.md"
    f.parent.mkdir(parents=True)
    url = _file_to_url(f, docs_root)
    assert url == "https://kubernetes.io/docs/concepts/"
