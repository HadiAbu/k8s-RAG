"""
Download (or update) the kubernetes/website repository.

Usage:
    python -m scripts.download_docs
"""
import logging

from app.ingestion.downloader import clone_or_update_repo, get_docs_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    repo = clone_or_update_repo()
    docs = get_docs_root(repo)
    md_files = list(docs.rglob("*.md")) + list(docs.rglob("*.mdx"))
    logger.info("Repository ready at: %s", repo)
    logger.info("Documentation files found: %d", len(md_files))
