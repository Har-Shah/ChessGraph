"""Ranking metrics.

All take a ranked list of retrieved ids and a set of relevant ids.
"""
from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant set that appears in the top k.

    Undefined with no relevant documents, so those queries are excluded from
    the query set rather than scored as 0 or 1.
    """
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant hit. Rewards getting one right answer high."""
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Included alongside recall because recall treats a relevant document at
    rank 1 and rank 10 identically. For a report that cites the top few
    results, rank position matters.
    """
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1)
              for i, doc_id in enumerate(retrieved[:k], start=1)
              if doc_id in relevant)
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def hit_rate(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Did we find anything relevant at all in the top k."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def summarise(per_query: list[dict]) -> dict:
    """Mean each metric across queries, ignoring NaN."""
    if not per_query:
        return {}
    keys = [k for k in per_query[0] if isinstance(per_query[0][k], (int, float))]
    out = {}
    for key in keys:
        vals = [q[key] for q in per_query
                if isinstance(q.get(key), (int, float)) and not math.isnan(q[key])]
        out[key] = round(sum(vals) / len(vals), 4) if vals else float("nan")
    return out
