from __future__ import annotations

import math

from retrieval_pareto.types import Hit, Query


def compute_quality_metrics(
    queries: list[Query],
    rankings: list[list[Hit]],
    qrels: dict[str, dict[str, float]],
) -> dict[str, float]:
    ndcg_values: list[float] = []
    recall_values: list[float] = []
    mrr_values: list[float] = []
    map_values: list[float] = []

    for query, hits in zip(queries, rankings):
        relevant = qrels.get(query.query_id, {})
        if not relevant:
            continue
        ndcg_values.append(_ndcg_at_k(hits, relevant, 10))
        recall_values.append(_recall_at_k(hits, relevant, 100))
        mrr_values.append(_mrr_at_k(hits, relevant, 10))
        map_values.append(_map_at_k(hits, relevant, 100))

    return {
        "ndcg_at_10": _mean(ndcg_values),
        "recall_at_100": _mean(recall_values),
        "mrr_at_10": _mean(mrr_values),
        "map_at_100": _mean(map_values),
    }


def _ndcg_at_k(hits: list[Hit], relevant: dict[str, float], k: int) -> float:
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        rel = relevant.get(hit.doc_id, 0.0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)

    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))
    return dcg / idcg if idcg else 0.0


def _recall_at_k(hits: list[Hit], relevant: dict[str, float], k: int) -> float:
    found = sum(1 for hit in hits[:k] if hit.doc_id in relevant)
    return found / len(relevant) if relevant else 0.0


def _mrr_at_k(hits: list[Hit], relevant: dict[str, float], k: int) -> float:
    for rank, hit in enumerate(hits[:k], start=1):
        if hit.doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _map_at_k(hits: list[Hit], relevant: dict[str, float], k: int) -> float:
    found = 0
    precision_sum = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        if hit.doc_id in relevant:
            found += 1
            precision_sum += found / rank
    return precision_sum / min(len(relevant), k) if relevant else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
