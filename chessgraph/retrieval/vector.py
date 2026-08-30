"""Dense vector retrieval over the same corpus.

Model: BAAI/bge-small-en-v1.5 (384-dim) via fastembed. Chosen over
sentence-transformers because fastembed runs on ONNX, so the dependency is
~100MB instead of pulling in all of PyTorch, and it is deterministic on CPU,
which matters for a reproducible evaluation.

Embeddings are cached to disk keyed on a hash of the corpus text. Re-running
the eval must not re-embed, both for speed and so that a rerun compares
identical vectors rather than silently re-encoding.

WHAT TO EXPECT
--------------
Dense retrieval should beat BM25 on paraphrase ("dropping pieces" vs "hanging
piece") and lose on exact identifiers (ECO codes like B02, specific SAN moves
like Nxd5), where lexical match is exactly right and embeddings blur. That
trade is the reason hybrid exists.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from chessgraph.config import CACHE
from chessgraph.retrieval.base import Hit, RetrievalResult
from chessgraph.retrieval.corpus import Document

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# BGE is an ASYMMETRIC model: it expects queries to carry a retrieval
# instruction while passages are embedded bare. Left as a flag rather than
# hardcoded because on this corpus it did not change ranking in spot checks,
# and the evaluation should measure that rather than assume it.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class VectorRetriever:
    name = "vector"

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 cache_dir: Path | None = None,
                 query_prefix: str | None = BGE_QUERY_PREFIX):
        self.model_name = model_name
        self.query_prefix = query_prefix or ""
        self.cache_dir = cache_dir or CACHE
        self.docs: list[Document] = []
        self.matrix: np.ndarray | None = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _cache_path(self, docs: list[Document]) -> Path:
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        for d in docs:
            h.update(d.doc_id.encode())
            h.update(d.text.encode())
        return self.cache_dir / f"emb_{h.hexdigest()[:16]}.npy"

    def index(self, docs: list[Document]) -> None:
        self.docs = docs
        if not docs:
            self.matrix = np.zeros((0, 384), dtype=np.float32)
            return
        path = self._cache_path(docs)
        if path.exists():
            self.matrix = np.load(path)
            return
        model = self._load_model()
        vecs = np.array(list(model.embed([d.text for d in docs])), dtype=np.float32)
        # L2-normalise once at index time so search is a plain dot product
        # rather than a cosine computed per query.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.matrix = vecs / np.clip(norms, 1e-9, None)
        np.save(path, self.matrix)

    def _embed_query(self, query: str) -> np.ndarray:
        model = self._load_model()
        v = np.array(list(model.embed([self.query_prefix + query]))[0],
                     dtype=np.float32)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        if self.matrix is None or len(self.docs) == 0:
            return RetrievalResult(query, self.name, [], 0.0)
        qv = self._embed_query(query)
        sims = self.matrix @ qv
        top = np.argsort(-sims)[:k]
        hits = [
            Hit(doc_id=self.docs[i].doc_id, score=round(float(sims[i]), 4),
                rank=r + 1, doc=self.docs[i], explanation="semantic similarity")
            for r, i in enumerate(top)
        ]
        return RetrievalResult(query, self.name, hits,
                               (time.perf_counter() - t0) * 1000)
