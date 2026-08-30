"""Hybrid retrieval by Reciprocal Rank Fusion over any set of retrievers."""
from __future__ import annotations

import time

from chessgraph.retrieval.base import RetrievalResult, reciprocal_rank_fusion
from chessgraph.retrieval.corpus import Document


class HybridRetriever:
    def __init__(self, retrievers: list, name: str = "hybrid",
                 fetch_k: int = 30, rrf_k: int = 60):
        self.retrievers = retrievers
        self.name = name
        # Each component fetches deeper than k so fusion has material to work
        # with. Fusing two top-10 lists can only ever surface 20 documents.
        self.fetch_k = fetch_k
        self.rrf_k = rrf_k

    def index(self, docs: list[Document]) -> None:
        for r in self.retrievers:
            r.index(docs)

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        results = [r.search(query, k=self.fetch_k) for r in self.retrievers]
        hits = reciprocal_rank_fusion(results, k=k, rrf_k=self.rrf_k)
        return RetrievalResult(query, self.name, hits,
                               (time.perf_counter() - t0) * 1000)
