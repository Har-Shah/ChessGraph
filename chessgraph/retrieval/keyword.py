"""BM25 keyword retrieval: the baseline that has to be beaten.

BM25 rather than raw TF-IDF because BM25 is the honest lexical baseline. It is
what a real search system would use, and picking a weak baseline is the easiest
way to manufacture a result that does not survive contact with reality.

Two refinements over plain term frequency that matter here:
  - Saturation (k1): a document repeating "fork" eight times is not eight times
    more relevant than one mentioning it once. k1 caps that growth.
  - Length normalisation (b): our documents vary in length mainly because
    opening names vary in length, which has nothing to do with relevance.
"""
from __future__ import annotations

import math
import time
from collections import Counter

from chessgraph.retrieval.base import Hit, RetrievalResult, tokenize
from chessgraph.retrieval.corpus import Document


class BM25Retriever:
    name = "keyword_bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[Document] = []
        self.tf: list[Counter] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.lengths: list[int] = []
        self.avgdl: float = 0.0

    def index(self, docs: list[Document]) -> None:
        self.docs = docs
        self.tf, self.lengths, self.df = [], [], Counter()
        for d in docs:
            toks = tokenize(d.text)
            counts = Counter(toks)
            self.tf.append(counts)
            self.lengths.append(len(toks))
            self.df.update(counts.keys())
        n = max(len(docs), 1)
        self.avgdl = sum(self.lengths) / n
        # Probabilistic IDF with the +0.5 smoothing; the max() floor keeps
        # terms appearing in almost every document from going negative, which
        # would otherwise let a document be penalised for containing them.
        self.idf = {
            term: max(1e-6, math.log(1 + (n - df + 0.5) / (df + 0.5)))
            for term, df in self.df.items()
        }

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        q_terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        for i, counts in enumerate(self.tf):
            if not counts:
                continue
            dl = self.lengths[i]
            s = 0.0
            for term in q_terms:
                f = counts.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (f * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda kv: -kv[1])
        hits = [
            Hit(doc_id=self.docs[i].doc_id, score=round(s, 4), rank=r + 1,
                doc=self.docs[i], explanation="lexical overlap")
            for r, (i, s) in enumerate(scores[:k])
        ]
        return RetrievalResult(query, self.name, hits,
                               (time.perf_counter() - t0) * 1000)
