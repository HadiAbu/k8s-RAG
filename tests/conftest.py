import asyncio
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Provide a fresh SQLite database in a temp dir."""
    db_path = tmp_path / "test.db"
    import app.config as cfg
    monkeypatch.setattr(cfg.settings, "sqlite_path", db_path)
    from app.storage.database import init_db
    init_db(db_path)
    return db_path


SAMPLE_DOC = {
    "id": "abc12345",
    "title": "StatefulSets",
    "url": "https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/",
    "path": "concepts/workloads/controllers/statefulset.md",
    "section": "Concepts",
    "version": "",
    "content": (
        "A StatefulSet manages the deployment and scaling of a set of Pods, "
        "and provides guarantees about the ordering and uniqueness of these Pods.\n\n"
        "## Use Cases\n\nStatefulSets are valuable for applications that require one or more of:\n"
        "- Stable, unique network identifiers\n"
        "- Stable, persistent storage\n"
        "- Ordered, graceful deployment and scaling\n\n"
        "```yaml\napiVersion: apps/v1\nkind: StatefulSet\nmetadata:\n  name: web\n```"
    ),
}
