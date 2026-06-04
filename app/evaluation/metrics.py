"""Retrieval and answer quality metrics."""


def recall_at_k(relevant_urls: set[str], retrieved_urls: list[str], k: int) -> float:
    """Fraction of relevant documents found in top-k retrieved results."""
    if not relevant_urls:
        return 0.0
    top_k = set(retrieved_urls[:k])
    return len(relevant_urls & top_k) / len(relevant_urls)


def precision_at_k(relevant_urls: set[str], retrieved_urls: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved_urls[:k]
    hits = sum(1 for u in top_k if u in relevant_urls)
    return hits / k


def reciprocal_rank(relevant_urls: set[str], retrieved_urls: list[str]) -> float:
    """Reciprocal rank of the first relevant document in the ranked list."""
    for rank, url in enumerate(retrieved_urls, 1):
        if url in relevant_urls:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(results: list[dict]) -> float:
    """MRR over a list of {relevant_urls, retrieved_urls} dicts."""
    if not results:
        return 0.0
    rr_sum = sum(
        reciprocal_rank(set(r["relevant_urls"]), r["retrieved_urls"])
        for r in results
    )
    return rr_sum / len(results)


def aggregate_metrics(results: list[dict], k: int = 5) -> dict:
    """Compute aggregate retrieval metrics over a list of evaluation results."""
    if not results:
        return {}

    recall_scores = [
        recall_at_k(set(r["relevant_urls"]), r["retrieved_urls"], k)
        for r in results
    ]
    precision_scores = [
        precision_at_k(set(r["relevant_urls"]), r["retrieved_urls"], k)
        for r in results
    ]
    rr_scores = [
        reciprocal_rank(set(r["relevant_urls"]), r["retrieved_urls"])
        for r in results
    ]
    latencies = [r.get("latency_s", 0.0) for r in results]

    return {
        f"recall@{k}": sum(recall_scores) / len(recall_scores),
        f"precision@{k}": sum(precision_scores) / len(precision_scores),
        "mrr": sum(rr_scores) / len(rr_scores),
        "avg_latency_s": sum(latencies) / len(latencies),
        "p95_latency_s": sorted(latencies)[int(0.95 * len(latencies))],
        "total_questions": len(results),
    }
