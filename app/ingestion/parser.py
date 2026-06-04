import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import frontmatter

logger = logging.getLogger(__name__)

K8S_DOCS_BASE = "https://kubernetes.io/docs"

# Hugo shortcode cleaning rules: (pattern, replacement)
_SHORTCODE_RULES = [
    # note / warning / caution / tip blocks — keep inner content
    (re.compile(r"\{\{<\s*(?:note|warning|caution|tip)\s*>\}\}(.*?)\{\{<\s*/(?:note|warning|caution|tip)\s*>\}\}", re.DOTALL), r"\1"),
    # tabs / tab blocks — keep content, drop wrappers
    (re.compile(r"\{\{<\s*(?:tabs|tab\s[^>]*)\s*>\}\}(.*?)\{\{<\s*/(?:tabs?)\s*>\}\}", re.DOTALL), r"\1"),
    # feature-state, codenew, and other self-closing shortcodes
    (re.compile(r"\{\{<[^>]*>\}\}"), ""),
    # closing tags left over
    (re.compile(r"\{\{<\s*/[^>]*>\}\}"), ""),
    # percent-style shortcodes
    (re.compile(r"\{\{%.*?%\}\}", re.DOTALL), ""),
    # HTML comments
    (re.compile(r"<!--.*?-->", re.DOTALL), ""),
    # Raw HTML tags (keep content)
    (re.compile(r"<(?!code|pre|ul|ol|li|table|thead|tbody|tr|th|td|p|br|h[1-6])[a-zA-Z][^>]*>"), ""),
    (re.compile(r"</(?!code|pre|ul|ol|li|table|thead|tbody|tr|th|td|p|h[1-6])[a-zA-Z][^>]*>"), ""),
]

_BLANK_LINES = re.compile(r"\n{3,}")
_HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def _file_to_url(file_path: Path, docs_root: Path) -> str:
    rel = file_path.relative_to(docs_root)
    parts = list(rel.parts)
    last = parts[-1]
    if last in ("_index.md", "_index.mdx", "index.md", "index.mdx"):
        parts = parts[:-1]
    else:
        parts[-1] = re.sub(r"\.(mdx?)$", "", last)
    url_path = "/".join(parts)
    return f"{K8S_DOCS_BASE}/{url_path}/"


def _top_section(file_path: Path, docs_root: Path) -> str:
    rel = file_path.relative_to(docs_root)
    parts = list(rel.parts)
    if parts:
        return parts[0].replace("-", " ").title()
    return "Kubernetes"


def clean_content(raw: str) -> str:
    text = raw
    for pattern, repl in _SHORTCODE_RULES:
        text = pattern.sub(repl, text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def parse_document(file_path: Path, docs_root: Path) -> Optional[dict]:
    try:
        post = frontmatter.load(str(file_path))
    except Exception as exc:
        logger.debug("Skipping %s: %s", file_path, exc)
        return None

    meta = post.metadata
    raw_content = post.content

    content = clean_content(raw_content)
    if len(content) < 80:
        return None

    title: str = meta.get("title") or ""
    if not title:
        m = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = m.group(1).strip() if m else file_path.stem.replace("-", " ").title()

    doc_id = hashlib.sha256(
        str(file_path.relative_to(docs_root)).encode()
    ).hexdigest()[:16]

    return {
        "id": doc_id,
        "title": title,
        "url": _file_to_url(file_path, docs_root),
        "path": str(file_path.relative_to(docs_root)),
        "section": _top_section(file_path, docs_root),
        "version": str(meta.get("version", "")),
        "content": content,
    }
