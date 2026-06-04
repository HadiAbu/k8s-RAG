"""
Run the retrieval evaluation benchmark.

Usage:
    python -m scripts.evaluate [--k 5] [--output results.json] [--category concepts]
"""
import asyncio
import json
import logging
import time
from pathlib import Path

import click
from tqdm import tqdm

from app.evaluation.benchmark import BENCHMARK_QUESTIONS, CATEGORIES
from app.evaluation.metrics import aggregate_metrics
from app.retrieval.retriever import retrieve

logger = logging.getLogger(__name__)


async def _evaluate_question(q: dict, k: int) -> dict:
    start = time.perf_counter()
    chunks = await retrieve(q["question"], top_k=k)
    latency = time.perf_counter() - start

    retrieved_urls = [c["url"] for c in chunks]
    relevant_urls = [q["expected_url"]] if "expected_url" in q else []

    return {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "relevant_urls": relevant_urls,
        "retrieved_urls": retrieved_urls,
        "latency_s": latency,
        "num_chunks": len(chunks),
    }


async def run_evaluation(questions: list[dict], k: int) -> list[dict]:
    results = []
    for q in tqdm(questions, desc="Evaluating"):
        try:
            r = await _evaluate_question(q, k)
        except Exception as exc:
            logger.warning("Question %s failed: %s", q["id"], exc)
            r = {"id": q["id"], "category": q["category"], "error": str(exc)}
        results.append(r)
    return results


@click.command()
@click.option("--k", default=5, show_default=True, help="Top-k for retrieval")
@click.option("--output", default="evaluation_results.json", show_default=True)
@click.option("--category", default=None, help=f"Filter by category: {CATEGORIES}")
def main(k: int, output: str, category: str | None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    questions = BENCHMARK_QUESTIONS
    if category:
        questions = [q for q in questions if q["category"] == category]
        if not questions:
            logger.error("No questions for category '%s'. Valid: %s", category, CATEGORIES)
            raise SystemExit(1)

    logger.info("Evaluating %d questions at k=%d …", len(questions), k)
    results = asyncio.run(run_evaluation(questions, k))

    # Filter results that have relevant_urls for metric computation
    scoreable = [r for r in results if r.get("relevant_urls")]
    if scoreable:
        metrics = aggregate_metrics(scoreable, k=k)
        print("\n── Retrieval Metrics ──────────────────")
        for key, val in metrics.items():
            print(f"  {key:<20} {val:.4f}" if isinstance(val, float) else f"  {key:<20} {val}")
    else:
        logger.warning("No questions with expected_url — cannot compute recall/MRR.")

    by_category: dict[str, list] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r)

    print("\n── Per-category latency ───────────────")
    for cat, cat_results in sorted(by_category.items()):
        lats = [r.get("latency_s", 0) for r in cat_results]
        avg = sum(lats) / len(lats)
        print(f"  {cat:<20} avg={avg:.3f}s  n={len(cat_results)}")

    Path(output).write_text(json.dumps(results, indent=2))
    logger.info("Results written to %s", output)


if __name__ == "__main__":
    main()
