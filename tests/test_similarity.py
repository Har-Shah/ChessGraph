"""Position similarity tests.

The pruning-bound test is the important one. `build_similarity_edges` skips
any pair whose pawn similarity is below the threshold, and that is only valid
because the score can never exceed pawn similarity. If the scoring formula
changes and that stops holding, the search silently starts missing real
matches, so the invariant is asserted directly.
"""
import chess
import pytest

from chessgraph.store.similarity import (
    PAWN_FLOOR, extract_features, similarity, build_similarity_edges,
    blocking_stats, _jaccard,
)

KID = "r1bqk2r/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP3PPP/R2QKB1R w KQkq - 0 1"
KID_ONE_PIECE_OFF = "r2qk2r/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP3PPP/R2QKB1R w KQkq - 0 1"
KID_OPPOSITE_CASTLING = "r1bq1rk1/pp2ppbp/2np1np1/8/2PNP3/2N1B3/PP3PPP/2KRQB1R w - - 0 1"
FRENCH = "rnbqkbnr/pp3ppp/4p3/2pp4/3P1B2/4P3/PPP2PPP/RN1QKBNR w KQkq - 0 1"


def f(fen, key="k", phase="middlegame"):
    return extract_features(fen, key, phase)


def test_identical_positions_score_one():
    a = f(KID, "a")
    assert similarity(a, a) == pytest.approx(1.0)


def test_similarity_is_symmetric():
    a, b = f(KID, "a"), f(KID_ONE_PIECE_OFF, "b")
    assert similarity(a, b) == pytest.approx(similarity(b, a))


def test_same_structure_scores_far_above_different_structure():
    a = f(KID, "a")
    same = similarity(a, f(KID_ONE_PIECE_OFF, "b"))
    diff = similarity(a, f(FRENCH, "c", phase="opening"))
    assert same > 0.9
    assert diff < 0.5
    assert same > diff * 2


def test_pawn_structure_gates_the_score():
    """Identical pieces on an unrelated skeleton must not score highly.

    This is the regression guard for the additive version, which scored two
    positions from different openings at 0.604 because material and king terms
    put a floor under everything.
    """
    a = f(KID, "a")
    c = f(FRENCH, "c", phase="opening")
    assert similarity(a, c) < 0.55


def test_opposite_side_castling_scores_below_same_side():
    a = f(KID, "a")
    same_side = similarity(a, f(KID_ONE_PIECE_OFF, "b"))
    opposite = similarity(a, f(KID_OPPOSITE_CASTLING, "d"))
    assert opposite < same_side


def test_score_never_exceeds_pawn_similarity():
    """The invariant that makes the search prune exact rather than heuristic."""
    fens = [KID, KID_ONE_PIECE_OFF, KID_OPPOSITE_CASTLING, FRENCH,
            chess.STARTING_FEN,
            "8/5k2/6p1/8/8/6P1/5K2/8 w - - 0 40",
            "r3k2r/ppp2ppp/2n5/8/8/2N5/PPP2PPP/R3K2R w KQkq - 0 15"]
    feats = [f(x, str(i)) for i, x in enumerate(fens)]
    for a in feats:
        for b in feats:
            pawn = 0.5 * (_jaccard(a.white_pawns, b.white_pawns)
                          + _jaccard(a.black_pawns, b.black_pawns))
            assert similarity(a, b) <= pawn + 1e-9, (
                "score exceeded pawn similarity, the prune is now unsound")


def test_pawn_floor_is_a_proper_fraction():
    assert 0.0 < PAWN_FLOOR < 1.0


# ------------------------------------------------------------------- edges
def _corpus():
    return [f(KID, "kid"), f(KID_ONE_PIECE_OFF, "kid2"),
            f(KID_OPPOSITE_CASTLING, "kid3"),
            f(FRENCH, "french", phase="middlegame")]


def test_edges_link_the_related_positions_and_not_the_unrelated_one():
    edges = build_similarity_edges(_corpus(), top_k=5, threshold=0.60)
    pairs = {tuple(sorted((a, b))) for a, b, _ in edges}
    assert ("kid", "kid2") in pairs
    assert not any("french" in p for p in pairs)


def test_no_self_edges_and_no_duplicate_pairs():
    edges = build_similarity_edges(_corpus(), top_k=5, threshold=0.50)
    assert all(a != b for a, b, _ in edges)
    pairs = [tuple(sorted((a, b))) for a, b, _ in edges]
    assert len(pairs) == len(set(pairs))


def test_top_k_bounds_edges_per_node():
    """Without a per-node cap, common structures accumulate hub edges."""
    feats = [f(KID, f"n{i}") for i in range(12)]
    edges = build_similarity_edges(feats, top_k=2, threshold=0.5)
    from collections import Counter
    degree = Counter()
    for a, b, _ in edges:
        degree[a] += 1
        degree[b] += 1
    # Each node proposes at most top_k, so degree stays bounded well below n.
    assert max(degree.values()) <= 12


def test_threshold_is_respected():
    edges = build_similarity_edges(_corpus(), top_k=5, threshold=0.95)
    assert all(score >= 0.95 for _, _, score in edges)


def test_blocking_reduces_the_comparison_count():
    feats = ([f(KID, f"m{i}") for i in range(30)]
             + [f("8/5k2/6p1/8/8/6P1/5K2/8 w - - 0 40", f"e{i}", "endgame")
                for i in range(30)])
    st = blocking_stats(feats)
    assert st["blocked_comparisons"] < st["naive_comparisons"]
    assert st["blocks"] >= 2
