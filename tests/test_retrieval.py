"""Retriever and metric tests that need no database."""
import pytest

from chessgraph.evaluation.metrics import (
    recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, hit_rate,
)
from chessgraph.retrieval.base import tokenize, reciprocal_rank_fusion, Hit, RetrievalResult
from chessgraph.retrieval.corpus import Document
from chessgraph.retrieval.keyword import BM25Retriever


# ------------------------------------------------------------------- metrics
def test_recall_at_k_hand_computed():
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c", "d"}, 3) == pytest.approx(0.5)


def test_recall_is_nan_with_no_relevant_documents():
    import math
    assert math.isnan(recall_at_k(["a"], set(), 5))


def test_precision_and_reciprocal_rank():
    assert precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_is_one_when_all_relevant_are_ranked_first():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_ndcg_normalises_against_a_capped_ideal():
    """With more relevant documents than slots, a perfect top-k still scores 1."""
    rel = {f"d{i}" for i in range(50)}
    perfect = [f"d{i}" for i in range(10)]
    assert ndcg_at_k(perfect, rel, 10) == pytest.approx(1.0)


def test_hit_rate_is_binary():
    assert hit_rate(["x", "a"], {"a"}, 2) == 1.0
    assert hit_rate(["x", "y"], {"a"}, 2) == 0.0


# --------------------------------------------------------------------- BM25
def _docs():
    return [
        Document("a", "fork knight tactics in the Sicilian Defense"),
        Document("b", "back rank mate in the French Defense endgame"),
        Document("c", "hanging piece in the Sicilian Defense Najdorf"),
    ]


def test_bm25_ranks_documents_matching_more_query_terms_higher():
    r = BM25Retriever()
    r.index(_docs())
    hits = r.search("Sicilian fork", k=3).hits
    assert hits[0].doc_id == "a"
    assert hits[0].score > hits[1].score


def test_bm25_returns_nothing_when_no_term_matches():
    r = BM25Retriever()
    r.index(_docs())
    assert r.search("zzz nonexistent", k=5).hits == []


def test_bm25_idf_is_never_negative():
    """A term in every document must not push scores below zero."""
    docs = [Document(str(i), "chess chess chess") for i in range(10)]
    r = BM25Retriever()
    r.index(docs)
    assert all(v >= 0 for v in r.idf.values())


def test_tokenize_is_lowercase_alphanumeric():
    assert tokenize("Nxd5! (B02) -- Alekhine's") == [
        "nxd5", "b02", "alekhine", "s"]


# ----------------------------------------------------------------- fusion
def test_rrf_promotes_documents_ranked_well_by_several_retrievers():
    a = RetrievalResult("q", "r1", [Hit("x", 9.0, 1), Hit("y", 8.0, 2)])
    b = RetrievalResult("q", "r2", [Hit("y", 9.0, 1), Hit("z", 8.0, 2)])
    fused = reciprocal_rank_fusion([a, b], k=3)
    # y is rank 2 and rank 1; x and z each appear once.
    assert fused[0].doc_id == "y"
    assert {h.doc_id for h in fused} == {"x", "y", "z"}


def test_rrf_ignores_score_magnitude():
    """Fusion must depend on rank only, since scores are incomparable."""
    big = RetrievalResult("q", "r1", [Hit("x", 10_000.0, 2), Hit("y", 9_999.0, 1)])
    small = RetrievalResult("q", "r2", [Hit("x", 0.01, 2), Hit("y", 0.02, 1)])
    fused = reciprocal_rank_fusion([big, small], k=2)
    assert fused[0].doc_id == "y"
