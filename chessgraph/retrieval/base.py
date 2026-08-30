"""Retriever interface and rank fusion."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from chessgraph.retrieval.corpus import Document

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    Deliberately simple and shared by every lexical component, so the keyword
    baseline and any lexical part of the hybrid tokenise identically.
    """
    return TOKEN_RE.findall(text.lower())


@dataclass
class Hit:
    doc_id: str
    score: float
    rank: int
    doc: Document | None = None
    explanation: str = ""       # how this retriever reached the document


@dataclass
class RetrievalResult:
    query: str
    retriever: str
    hits: list[Hit] = field(default_factory=list)
    latency_ms: float = 0.0

    def doc_ids(self) -> list[str]:
        return [h.doc_id for h in self.hits]


class Retriever(Protocol):
    name: str

    def index(self, docs: list[Document]) -> None: ...
    def search(self, query: str, k: int = 10) -> RetrievalResult: ...


def reciprocal_rank_fusion(results: list[RetrievalResult], k: int = 10,
                           rrf_k: int = 60) -> list[Hit]:
    """Combine ranked lists by Reciprocal Rank Fusion.

    score(d) = sum over lists of 1 / (rrf_k + rank(d))

    RRF is used here instead of score averaging because the retrievers produce
    incomparable scores: BM25 is unbounded and corpus-dependent, cosine
    similarity sits in [-1, 1], and graph scores are hand-weighted traversal
    counts. Normalising those onto a common scale requires assumptions that are
    hard to defend. RRF only looks at RANK, so it sidesteps the problem
    entirely, and the constant 60 is the value from the original TREC work.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document | None] = {}
    sources: dict[str, list[str]] = {}
    for res in results:
        for hit in res.hits:
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + hit.rank)
            docs.setdefault(hit.doc_id, hit.doc)
            sources.setdefault(hit.doc_id, []).append(res.retriever)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [
        Hit(doc_id=did, score=sc, rank=i + 1, doc=docs.get(did),
            explanation=f"fused from {'+'.join(sources[did])}")
        for i, (did, sc) in enumerate(ranked)
    ]
