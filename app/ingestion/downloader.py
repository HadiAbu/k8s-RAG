import logging
import subprocess
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

K8S_DOCS_SUBDIR = Path("content") / "en" / "docs"


def clone_or_update_repo() -> Path:
    """Clone kubernetes/website with --depth=1, or pull latest changes if already cloned."""
    target = settings.raw_docs_dir
    target.parent.mkdir(parents=True, exist_ok=True)

    if (target / ".git").exists():
        logger.info("Updating existing repository at %s …", target)
        subprocess.run(
            ["git", "-C", str(target), "pull", "--ff-only"],
            check=True,
        )
    else:
        logger.info("Cloning %s → %s …", settings.k8s_website_repo, target)
        subprocess.run(
            [
                "git", "clone",
                "--depth=1",
                "--branch", settings.k8s_docs_branch,
                "--single-branch",
                settings.k8s_website_repo,
                str(target),
            ],
            check=True,
        )

    return target


def get_docs_root(repo_root: Path) -> Path:
    return repo_root / K8S_DOCS_SUBDIR


def iter_doc_files(docs_root: Path):
    """Yield all .md and .mdx files under the docs root."""
    for pattern in ("*.md", "*.mdx"):
        yield from docs_root.rglob(pattern)
