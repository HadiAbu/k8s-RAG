"""
Kubernetes RAG Assistant — CLI

Usage:
    python cli.py ask "What is a StatefulSet?"
    python cli.py ask "How do I configure liveness probes?" --show-chunks
    python cli.py build-index
    python cli.py serve
    python cli.py stats
"""
import asyncio

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def cli():
    """Kubernetes Documentation RAG Assistant"""


@cli.command()
@click.argument("question")
@click.option("--top-k", default=5, show_default=True, help="Chunks to retrieve")
@click.option("--show-chunks", is_flag=True, help="Show retrieved chunk snippets")
def ask(question: str, top_k: int, show_chunks: bool):
    """Ask a natural-language question about Kubernetes."""
    from app.retrieval.retriever import retrieve
    from app.generation.generator import generate_answer

    async def _run():
        with console.status("[cyan]Searching documentation…[/]"):
            chunks = await retrieve(question, top_k=top_k)

        if not chunks:
            console.print(
                Panel(
                    "I could not find sufficient information in the indexed Kubernetes documentation.",
                    title="[red]No results[/]",
                )
            )
            return

        with console.status("[cyan]Generating answer…[/]"):
            result = await generate_answer(question, chunks)

        console.print(Panel(Markdown(result["answer"]), title="[bold green]Answer[/]", border_style="green"))

        if result["sources"]:
            table = Table(title="Sources", show_header=True)
            table.add_column("Title", style="cyan")
            table.add_column("URL", style="blue")
            for s in result["sources"]:
                table.add_row(s["title"], s["url"])
            console.print(table)

        if show_chunks:
            console.print()
            for i, c in enumerate(chunks, 1):
                score = c.get("rerank_score", 0.0)
                snippet = c["content"][:400] + ("…" if len(c["content"]) > 400 else "")
                console.print(
                    Panel(
                        snippet,
                        title=f"[yellow]Chunk {i}: {c['title']} (rerank={score:.3f})[/]",
                        border_style="yellow",
                    )
                )

    asyncio.run(_run())


@cli.command("build-index")
def build_index():
    """Download and index the Kubernetes documentation."""
    from scripts.build_index import build_index_pipeline

    asyncio.run(build_index_pipeline())


@cli.command()
def serve():
    """Start the FastAPI server."""
    import uvicorn
    from app.config import settings

    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


@cli.command()
def stats():
    """Show current index statistics."""
    from app.storage.database import get_stats

    async def _run():
        s = await get_stats()
        table = Table(title="Index Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        for k, v in s.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)

    asyncio.run(_run())


if __name__ == "__main__":
    cli()
