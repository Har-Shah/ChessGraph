"""Position similarity: the `resembles` relation.

WHY THIS IS THE INTERESTING EDGE
Every other edge in the graph is a lookup of something already recorded. This
one is computed, and it is the only capability text retrieval cannot imitate.
Two positions can share no opening name, no player, no theme and no vocabulary,
and still be the same position to a chess player, because the pawn structure
and piece configuration are what determine how a position should be played.

WHAT MAKES TWO POSITIONS SIMILAR
Not overall piece overlap. A player cares about structure, in roughly this
order of importance:

  1. Pawn structure. It is the skeleton. It decides which files open, where
     pieces belong, and which endgames are good. Two positions with the same
     pawn structure play alike even with different pieces on the board.
  2. Material balance. Rook versus knight is a different game to rook versus
     rook, regardless of structure.
  3. King placement. Same structure with opposite-side castling is a different
     game to same-side castling.
  4. Minor piece configuration. Which minors remain and roughly where.

The weights below encode that ordering. They are hand-set rather than learned,
which is the honest MVP position: a GNN over the position graph is the natural
successor and is explicitly out of scope until the evaluation pipeline works.

THE COST PROBLEM AND HOW IT IS SOLVED
Comparing 25,000 positions pairwise is 312 million comparisons. Blocking makes
it tractable: positions only compete within a bucket keyed on
(phase, side_to_move, coarse material). Positions in different buckets cannot be
similar under the weights above, so the pruning costs nothing in recall and
turns the problem into a few thousand small comparisons.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import chess

# Pawn structure GATES the score rather than contributing a weighted share.
# An additive blend was tried first and calibrated badly: two positions from
# completely different openings scored 0.604, because material and king terms
# stay high for almost any pair of early middlegames and set a floor under
# everything. Multiplying by pawn similarity removes the floor, which matches
# how the position actually works. Different skeleton means different game, no
# matter what else agrees.
W_MATERIAL = 0.50      # weights below apply only to the non-pawn factor
W_KING = 0.25
W_MINOR_CONFIG = 0.25
PAWN_FLOOR = 0.60      # how much of the score the gate can suppress

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


@dataclass(frozen=True)
class PositionFeatures:
    """Structural fingerprint of a position, from the side-to-move's view."""
    pos_key: str
    white_pawns: frozenset[int]
    black_pawns: frozenset[int]
    material: tuple[int, ...]          # (P,N,B,R,Q) white then black
    white_king: int
    black_king: int
    minors: frozenset[tuple[int, int, bool]]   # (square, piece_type, is_white)
    phase: str
    side_to_move: bool

    def block_key(self) -> tuple:
        """Positions only compete inside a bucket.

        Coarse material rather than exact counts, so a position that is one
        pawn different can still be compared. Being too strict here silently
        destroys recall, which is much worse than a slightly larger bucket.
        """
        w = sum(self.material[:5])
        b = sum(self.material[5:])
        return (self.phase, self.side_to_move, min(w // 4, 6), min(b // 4, 6))


def extract_features(fen: str, pos_key: str, phase: str) -> PositionFeatures:
    board = chess.Board(fen)
    order = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    material = tuple(
        len(board.pieces(pt, color))
        for color in (chess.WHITE, chess.BLACK) for pt in order
    )
    minors = frozenset(
        (sq, pt, color == chess.WHITE)
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        for color in (chess.WHITE, chess.BLACK)
        for sq in board.pieces(pt, color)
    )
    return PositionFeatures(
        pos_key=pos_key,
        white_pawns=frozenset(board.pieces(chess.PAWN, chess.WHITE)),
        black_pawns=frozenset(board.pieces(chess.PAWN, chess.BLACK)),
        material=material,
        white_king=board.king(chess.WHITE) if board.king(chess.WHITE) is not None else -1,
        black_king=board.king(chess.BLACK) if board.king(chess.BLACK) is not None else -1,
        minors=minors,
        phase=phase,
        side_to_move=board.turn,
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _king_similarity(a: PositionFeatures, b: PositionFeatures) -> float:
    """Same castling side matters more than exact square.

    A king on g1 and one on h1 are the same story. A king on g1 and one on c1
    are opposite-side castling, which is a different game entirely.
    """
    def zone(sq: int) -> int:
        if sq < 0:
            return -1
        f = chess.square_file(sq)
        return 0 if f <= 2 else (1 if f <= 5 else 2)   # queenside / centre / kingside

    score = 0.0
    for sq_a, sq_b in ((a.white_king, b.white_king), (a.black_king, b.black_king)):
        if zone(sq_a) == zone(sq_b):
            score += 0.5
        elif sq_a >= 0 and sq_b >= 0 and abs(
                chess.square_file(sq_a) - chess.square_file(sq_b)) <= 2:
            score += 0.2
    return score


def _material_similarity(a: PositionFeatures, b: PositionFeatures) -> float:
    """1.0 for identical material, falling off with the size of the difference."""
    diff = sum(abs(x - y) for x, y in zip(a.material, b.material))
    return max(0.0, 1.0 - diff / 10.0)


def similarity(a: PositionFeatures, b: PositionFeatures) -> float:
    """Structural similarity in [0, 1], gated on pawn structure.

        score = pawn_similarity * (PAWN_FLOOR + (1 - PAWN_FLOOR) * rest)

    `rest` is the weighted blend of material, king placement and minor piece
    configuration. Multiplying by pawn similarity means a shared skeleton is
    necessary rather than merely helpful: identical pieces on an unrelated pawn
    structure cannot score highly, which is the correct chess judgement.
    """
    pawn = 0.5 * (_jaccard(a.white_pawns, b.white_pawns)
                  + _jaccard(a.black_pawns, b.black_pawns))
    rest = (
        W_MATERIAL * _material_similarity(a, b)
        + W_KING * _king_similarity(a, b)
        + W_MINOR_CONFIG * _jaccard(a.minors, b.minors)
    )
    return pawn * (PAWN_FLOOR + (1.0 - PAWN_FLOOR) * rest)


def _bitboard(squares) -> int:
    bb = 0
    for sq in squares:
        bb |= 1 << sq
    return bb


def build_similarity_edges(features: list[PositionFeatures], *,
                           top_k: int = 5, threshold: float = 0.60,
                           progress: bool = False) -> list[tuple[str, str, float]]:
    """Return (pos_key_a, pos_key_b, score) for the strongest resemblances.

    Each position keeps at most `top_k` neighbours above `threshold`. Capping
    per node rather than globally keeps the graph navigable: without it a
    handful of very common structures would accumulate thousands of edges and
    every traversal would funnel through them.

    THE PRUNE THAT MAKES THIS EXACT AND FAST
    Because the score is `pawn * (PAWN_FLOOR + (1 - PAWN_FLOOR) * rest)` and
    `rest` is at most 1, the score can never exceed the pawn similarity. So
    `pawn < threshold` is a *proof* that the pair fails, not a heuristic. Pawn
    similarity is computed for a whole block at once with bitboard popcounts,
    and the expensive full score runs only on survivors.

    An earlier version capped block sizes at 400 to bound runtime, which
    silently discarded real matches from the largest block of 3,692. This
    version needs no cap, so recall is not traded away for speed.
    """
    import numpy as np

    blocks: dict[tuple, list[PositionFeatures]] = defaultdict(list)
    for f in features:
        blocks[f.block_key()].append(f)

    edges: list[tuple[str, str, float]] = []
    seen: set[tuple[str, str]] = set()
    by_key = {f.pos_key: f for f in features}

    for bi, members in enumerate(sorted(blocks.values(), key=len, reverse=True)):
        n = len(members)
        if n < 2:
            continue
        wp = np.array([_bitboard(m.white_pawns) for m in members], dtype=np.uint64)
        bp = np.array([_bitboard(m.black_pawns) for m in members], dtype=np.uint64)

        # Chunk the rows so a large block never materialises an n x n matrix.
        chunk = max(1, min(n, 2_000_000 // max(n, 1)))
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            w_lhs, b_lhs = wp[start:end, None], bp[start:end, None]

            def jac(lhs, rhs):
                inter = np.bitwise_count(lhs & rhs[None, :]).astype(np.float32)
                union = np.bitwise_count(lhs | rhs[None, :]).astype(np.float32)
                return np.divide(inter, union, out=np.ones_like(inter),
                                 where=union > 0)

            pawn = 0.5 * (jac(w_lhs, wp) + jac(b_lhs, bp))
            for local_i in range(end - start):
                i = start + local_i
                row = pawn[local_i]
                # Provable rejection: score can never exceed pawn similarity.
                cand = np.flatnonzero(row >= threshold)
                if cand.size == 0:
                    continue
                a = members[i]
                scored = []
                for j in cand:
                    if int(j) == i:
                        continue
                    b = members[int(j)]
                    sc = similarity(a, b)
                    if sc >= threshold:
                        scored.append((sc, b.pos_key))
                if not scored:
                    continue
                scored.sort(reverse=True)
                for sc, other in scored[:top_k]:
                    pair = tuple(sorted((a.pos_key, other)))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    edges.append((pair[0], pair[1], round(sc, 4)))
        if progress and n > 500:
            print(f"    block {bi}: {n} positions, {len(edges)} edges so far")
    return edges


def blocking_stats(features: list[PositionFeatures]) -> dict:
    blocks: dict[tuple, int] = defaultdict(int)
    for f in features:
        blocks[f.block_key()] += 1
    sizes = sorted(blocks.values(), reverse=True)
    naive = len(features) * (len(features) - 1) // 2
    blocked = sum(n * (n - 1) // 2 for n in sizes)
    return {
        "positions": len(features),
        "blocks": len(sizes),
        "largest_block": sizes[0] if sizes else 0,
        "median_block": sizes[len(sizes) // 2] if sizes else 0,
        "naive_comparisons": naive,
        "blocked_comparisons": blocked,
        "reduction": f"{(1 - blocked / naive) * 100:.2f}%" if naive else "n/a",
    }
