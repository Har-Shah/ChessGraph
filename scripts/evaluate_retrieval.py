#!/usr/bin/env python
"""Reproducible retrieval comparison.

    python scripts/evaluate_retrieval.py <subject-username>

Rebuilds the corpus and graph from SQLite, generates the query set with
programmatic ground truth, runs all four retrievers, and writes a JSON report
to data/eval_retrieval.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console

from chessgraph.config import DATA
from chessgraph.evaluation.harness import run_comparison, print_report, save_results
from chessgraph.evaluation.queries import build_query_set, query_set_stats
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
    k: int = typer.Option(20, help="Retrieval depth"),
    min_cp_loss: int = typer.Option(100, help="Minimum loss to count as a mistake"),
    min_relevant: int = typer.Option(5, help="Skip queries with fewer relevant docs"),
    out: str = typer.Option("", help="Where to write the JSON report"),
):
    with Store() as store:
        console.rule("Corpus")
        docs = build_corpus(store, min_cp_loss=min_cp_loss)
        stats = corpus_stats(docs)
        for key, val in stats.items():
            console.print(f"  [dim]{key:20s}[/] {val}")
        if not docs:
            console.print("[red]Empty corpus. Run scripts/ingest.py first.[/]")
            raise typer.Exit(1)

        console.rule("Graph")
        kg = ChessKnowledgeGraph.build(store, subject)
        gs = kg.stats()
        console.print(f"  nodes {gs.nodes}  edges {gs.edges}")
        console.print(f"  [dim]node kinds[/] {gs.by_kind}")
        console.print(f"  [dim]relations [/] {gs.by_relation}")

    console.rule("Query set")
    queries = build_query_set(docs, subject, min_relevant=min_relevant)
    for key, val in query_set_stats(queries).items():
        console.print(f"  [dim]{key:20s}[/] {val}")
    if not queries:
        console.print("[red]No queries had enough relevant documents.[/]")
        raise typer.Exit(1)

    console.rule("Retrievers")
    bm25 = BM25Retriever()
    vec = VectorRetriever()
    graph = GraphRetriever(kg, subject=subject)
    for r in (bm25, vec, graph):
        r.index(docs)
    hybrid_vg = HybridRetriever([vec, graph], name="hybrid_vector_graph")
    hybrid_all = HybridRetriever([bm25, vec, graph], name="hybrid_all")
    for r in (hybrid_vg, hybrid_all):
        r.retrievers = list(r.retrievers)   # already indexed above

    results = run_comparison([bm25, vec, graph, hybrid_vg, hybrid_all], queries, k=k)

    console.rule("Results")
    print_report(results, queries)

    out_path = Path(out) if out else (DATA / "eval_retrieval.json")
    save_results(results, queries, out_path)
    console.print(f"\n[green]wrote {out_path}[/]")


if __name__ == "__main__":
    app()
