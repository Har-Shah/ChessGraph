#!/usr/bin/env python
"""Run every evaluation and write a single JSON report.

    python scripts/evaluate_all.py <subject-username>

Covers:
  retrieval        Recall@K, nDCG, MRR for four retrievers, by query family
  opening_pred     held-out opening prediction vs a player-agnostic baseline
  weakness_persist do training-window weaknesses recur in held-out games
  grounding        do report claims resolve, get supported, and hold numerically
  recommendation   do depth 12 recommendations survive a depth 20 re-search
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from chessgraph.config import DATA
from chessgraph.evaluation.grounding import evaluate_grounding
from chessgraph.evaluation.harness import run_comparison, print_report, save_results
from chessgraph.evaluation.holdout import (
    temporal_split, evaluate_opening_prediction, evaluate_weakness_persistence,
    evaluate_opening_weakness_persistence,
)
from chessgraph.evaluation.queries import build_query_set, query_set_stats
from chessgraph.evaluation.recommendation import evaluate_recommendations
from chessgraph.report.generate import build_report
from chessgraph.retrieval.corpus import build_corpus, corpus_stats
from chessgraph.retrieval.graph_retriever import GraphRetriever
from chessgraph.retrieval.hybrid import HybridRetriever
from chessgraph.retrieval.keyword import BM25Retriever
from chessgraph.retrieval.vector import VectorRetriever
from chessgraph.store.db import Store
from chessgraph.store.graph import ChessKnowledgeGraph

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    subject: str,
    k: int = typer.Option(20),
    verify_depth: int = typer.Option(20, help="Depth for recommendation checking"),
    verify_sample: int = typer.Option(40, help="Positions to re-search"),
    skip_recommendation: bool = typer.Option(False, "--skip-recommendation"),
    out: str = typer.Option(""),
):
    t_start = time.time()
    results: dict = {"subject": subject}

    with Store() as store:
        console.rule("Corpus and graph")
        docs = build_corpus(store)
        results["corpus"] = corpus_stats(docs)
        for key, val in results["corpus"].items():
            console.print(f"  [dim]{key:20s}[/] {val}")

        kg = ChessKnowledgeGraph.build(store, subject)
        gs = kg.stats()
        results["graph"] = {"nodes": gs.nodes, "edges": gs.edges,
                            "by_kind": gs.by_kind, "by_relation": gs.by_relation}
        console.print(f"  [dim]{'graph':20s}[/] {gs.nodes} nodes, {gs.edges} edges")

        # ------------------------------------------------------- retrieval
        console.rule("1. Retrieval comparison")
        queries = build_query_set(docs, subject)
        results["query_set"] = query_set_stats(queries)
        console.print(f"  {results['query_set']}")

        bm25, vec = BM25Retriever(), VectorRetriever()
        graph = GraphRetriever(kg, subject=subject)
        graph_sim = GraphRetriever(kg, subject=subject, expand_similar=True)
        for r in (bm25, vec, graph, graph_sim):
            r.index(docs)
        retrievers = [bm25, vec, graph, graph_sim,
                      HybridRetriever([vec, graph], name="hybrid_vector_graph"),
                      HybridRetriever([bm25, vec, graph], name="hybrid_all")]
        retr_results = run_comparison(retrievers, queries, k=k)
        print_report(retr_results, queries)
        results["retrieval"] = {
            name: {kk: vv for kk, vv in r.items() if kk != "per_query"}
            for name, r in retr_results.items()
        }
        save_results(retr_results, queries, DATA / "eval_retrieval.json")

        # -------------------------------------------------------- held-out
        console.rule("2. Held-out temporal evaluation")
        split = temporal_split(store, subject)
        results["split"] = {"n_train": len(split.train_ids),
                            "n_test": len(split.test_ids),
                            "cutoff_date": split.cutoff_date}
        console.print(f"  train {len(split.train_ids)} games, "
                      f"test {len(split.test_ids)} games, cutoff {split.cutoff_date}")

        results["opening_prediction"] = {
            lvl: evaluate_opening_prediction(store, subject, split, level=lvl)
            for lvl in ("family", "opening", "eco")
        }
        t = Table(title="Opening prediction on held-out games")
        t.add_column("level"); t.add_column("top1", justify="right")
        t.add_column("top3", justify="right")
        t.add_column("baseline", justify="right"); t.add_column("lift", justify="right")
        for lvl, r in results["opening_prediction"].items():
            if "error" in r:
                t.add_row(lvl, "-", "-", "-", r["error"]); continue
            t.add_row(lvl, f"{r['top1_accuracy']:.3f}", f"{r['top3_accuracy']:.3f}",
                      f"{r['baseline_top1_ignores_color']:.3f}",
                      f"{r['lift_over_baseline']:+.3f}")
        console.print(t)

        results["weakness_persistence"] = evaluate_weakness_persistence(
            store, subject, split)
        results["opening_acpl_persistence"] = evaluate_opening_weakness_persistence(
            store, subject, split)
        console.print("  weakness persistence:", results["weakness_persistence"])
        console.print("  opening ACPL persistence:", results["opening_acpl_persistence"])

        # ------------------------------------------------------- grounding
        console.rule("3. Grounding")
        report = build_report(store, subject, kg, training_count=60)
        g = evaluate_grounding(store, report)
        results["grounding"] = g.summary()
        results["grounding_issues"] = [asdict(i) for i in g.issues[:20]]
        console.print(f"  {g.summary()}")
        if g.issues:
            console.print(f"  [yellow]{len(g.issues)} issues, first few:[/]")
            for i in g.issues[:5]:
                console.print(f"    {i.problem}: {i.detail}")

        training_positions = report.training_positions

    # -------------------------------------------------- recommendation
    if not skip_recommendation and training_positions:
        console.rule(f"4. Recommendation quality (re-search at depth {verify_depth})")
        console.print(f"  re-searching up to {verify_sample} positions, this is slow")
        rec = evaluate_recommendations(
            training_positions, verify_depth=verify_depth,
            sample=verify_sample)
        results["recommendation"] = rec.summary()
        console.print(f"  {rec.summary()}")

    results["wall_seconds"] = round(time.time() - t_start, 1)
    out_path = Path(out) if out else (DATA / f"evaluation_{subject.lower()}.json")
    out_path.write_text(json.dumps(results, indent=2, default=str))
    console.print(f"\n[green]wrote {out_path} in {results['wall_seconds']}s[/]")


if __name__ == "__main__":
    app()
