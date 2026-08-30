"""Retrieval evaluation harness.

Runs every retriever over every query, computes ranking metrics, and reports
results broken down by query family. The per-family breakdown is the point.
An aggregate mean would hide the only interesting finding, which is that
different retrievers win different question shapes.
"""
from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from chessgraph.evaluation.metrics import (
    recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, hit_rate, summarise,
)
from chessgraph.evaluation.queries import EvalQuery

console = Console()
K_VALUES = (1, 5, 10, 20)


def evaluate_retriever(retriever, queries: list[EvalQuery], k: int = 20) -> dict:
    """Run one retriever over the query set."""
    per_query = []
    latencies = []
    empty_results = 0

    for q in queries:
        res = retriever.search(q.text, k=k)
        ids = res.doc_ids()
        latencies.append(res.latency_ms)
        if not ids:
            empty_results += 1
        row = {
            "qid": q.qid, "family": q.family,
            "n_relevant": len(q.relevant), "n_retrieved": len(ids),
            "mrr": reciprocal_rank(ids, q.relevant),
        }
        for kk in K_VALUES:
            row[f"recall@{kk}"] = recall_at_k(ids, q.relevant, kk)
            row[f"ndcg@{kk}"] = ndcg_at_k(ids, q.relevant, kk)
        row["precision@10"] = precision_at_k(ids, q.relevant, 10)
        row["hit@10"] = hit_rate(ids, q.relevant, 10)
        per_query.append(row)

    by_family = defaultdict(list)
    for row in per_query:
        by_family[row["family"]].append(row)

    return {
        "retriever": retriever.name,
        "overall": summarise(per_query),
        "by_family": {fam: summarise(rows) for fam, rows in by_family.items()},
        "latency_ms_mean": round(statistics.mean(latencies), 2) if latencies else 0,
        "latency_ms_p95": round(
            sorted(latencies)[int(len(latencies) * 0.95) - 1], 2) if latencies else 0,
        "empty_results": empty_results,
        "per_query": per_query,
    }


def run_comparison(retrievers: list, queries: list[EvalQuery],
                   k: int = 20) -> dict:
    results = {}
    for r in retrievers:
        t0 = time.time()
        console.print(f"  running [bold]{r.name}[/] over {len(queries)} queries...")
        results[r.name] = evaluate_retriever(r, queries, k=k)
        results[r.name]["wall_seconds"] = round(time.time() - t0, 2)
    return results


def print_report(results: dict, queries: list[EvalQuery]) -> None:
    families = sorted({q.family for q in queries})

    t = Table(title="Overall (mean across all queries)")
    t.add_column("retriever", style="bold")
    for col in ("recall@10", "recall@20", "precision@10", "ndcg@10", "mrr", "hit@10"):
        t.add_column(col, justify="right")
    t.add_column("ms/query", justify="right")
    t.add_column("empty", justify="right")
    for name, r in results.items():
        o = r["overall"]
        t.add_row(name,
                  f"{o.get('recall@10', 0):.3f}", f"{o.get('recall@20', 0):.3f}",
                  f"{o.get('precision@10', 0):.3f}", f"{o.get('ndcg@10', 0):.3f}",
                  f"{o.get('mrr', 0):.3f}", f"{o.get('hit@10', 0):.3f}",
                  f"{r['latency_ms_mean']:.1f}", str(r["empty_results"]))
    console.print(t)

    for fam in families:
        n = sum(1 for q in queries if q.family == fam)
        ft = Table(title=f"Family: {fam}  ({n} queries)")
        ft.add_column("retriever", style="bold")
        for col in ("recall@10", "recall@20", "precision@10", "ndcg@10", "mrr"):
            ft.add_column(col, justify="right")
        best = {}
        for col in ("recall@10", "recall@20", "precision@10", "ndcg@10"):
            best[col] = max(
                (r["by_family"].get(fam, {}).get(col, 0) for r in results.values()),
                default=0)
        for name, r in results.items():
            f = r["by_family"].get(fam, {})
            def fmt(col):
                v = f.get(col, 0)
                mark = " *" if col in best and v == best[col] and v > 0 else ""
                return f"{v:.3f}{mark}"
            ft.add_row(name, fmt("recall@10"), fmt("recall@20"),
                       fmt("precision@10"), fmt("ndcg@10"), fmt("mrr"))
        console.print(ft)
    console.print("[dim]* marks the best value in that column.[/]")


def save_results(results: dict, queries: list[EvalQuery], path: Path) -> None:
    payload = {
        "queries": [
            {"qid": q.qid, "family": q.family, "text": q.text,
             "n_relevant": len(q.relevant), "description": q.description,
             "meta": q.meta}
            for q in queries
        ],
        "results": {
            name: {k: v for k, v in r.items() if k != "per_query"}
            for name, r in results.items()
        },
        "per_query": {name: r["per_query"] for name, r in results.items()},
    }
    path.write_text(json.dumps(payload, indent=2))
