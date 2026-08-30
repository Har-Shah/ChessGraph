"""Recommendation quality: do our suggested moves survive deeper analysis?

The system analyses at depth 12 for throughput. Every "you should have played
X" in the report comes from that depth. The obvious question is whether those
recommendations hold up, and it is answerable directly: re-search the same
positions at a much higher depth and compare.

Two metrics.

1. AGREEMENT AT 1. How often the depth 12 recommendation is still the best move
   at the verification depth. This is the headline number for whether depth 12
   was an adequate working depth.

2. RECOMMENDATION CENTIPAWN LOSS. When they disagree, by how much. Agreement is
   binary and pessimistic: two moves can transpose or be equally good, and a
   disagreement between two moves that are both fine costs nothing. Measured at
   the verification depth, this is what the recommendation actually costs the
   student, and it is the number that matters.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import chess

from chessgraph.config import EngineConfig
from chessgraph.engine.analyzer import Analyzer, CLAMP_CP


@dataclass
class RecommendationEval:
    n_positions: int
    verify_depth: int
    working_depth: int
    agreement_at_1: float
    agreement_at_2: float
    mean_recommendation_cp_loss: float
    median_recommendation_cp_loss: float
    worse_than_100cp: int
    details: list[dict]

    def summary(self) -> dict:
        return {
            "n_positions": self.n_positions,
            "working_depth": self.working_depth,
            "verify_depth": self.verify_depth,
            "agreement@1": round(self.agreement_at_1, 4),
            "agreement@2": round(self.agreement_at_2, 4),
            "mean_recommendation_cp_loss": round(self.mean_recommendation_cp_loss, 2),
            "median_recommendation_cp_loss": round(self.median_recommendation_cp_loss, 2),
            "recommendations_worse_than_100cp": self.worse_than_100cp,
        }


def evaluate_recommendations(positions: list[dict], *, verify_depth: int = 20,
                             working_depth: int = 12, sample: int = 60,
                             seed: int = 11) -> RecommendationEval:
    """positions: dicts with at least `fen` and `best_uci` from the report."""
    rng = random.Random(seed)
    pool = [p for p in positions if p.get("best_uci") and p.get("fen")]
    if len(pool) > sample:
        pool = rng.sample(pool, sample)

    # Verification runs with multipv 2 so agreement@2 is available, and with a
    # separate cache namespace so deep evals never contaminate the depth 12
    # numbers the rest of the system relies on.
    cfg = EngineConfig(depth=verify_depth, multipv=3, threads=4, hash_mb=512)
    details, losses = [], []
    agree1 = agree2 = 0

    with Analyzer(config=cfg) as az:
        for p in pool:
            board = chess.Board(p["fen"])
            deep = az.evaluate(board, depth=verify_depth)
            if deep.terminal or not deep.alternatives:
                continue
            deep_best = deep.best_uci
            top2 = {a["uci"] for a in deep.alternatives[:2]}
            rec = p["best_uci"]

            if rec == deep_best:
                agree1 += 1
            if rec in top2:
                agree2 += 1

            # Cost of the recommendation, measured at the verification depth.
            rec_score = next(
                (a for a in deep.alternatives if a["uci"] == rec), None)
            if rec_score is not None:
                cp = (CLAMP_CP if (rec_score["mate"] or 0) > 0
                      else -CLAMP_CP if (rec_score["mate"] or 0) < 0
                      else rec_score["score_cp"] or 0)
            else:
                # Not in the top 3, so search the resulting position directly.
                mv = chess.Move.from_uci(rec)
                if mv not in board.legal_moves:
                    continue
                board.push(mv)
                after = az.evaluate(board, depth=verify_depth)
                cp = -after.clamped_cp()
            loss = max(0, deep.clamped_cp() - cp)
            losses.append(loss)
            details.append({
                "fen": p["fen"], "recommended": rec,
                "deep_best": deep_best, "cp_loss_of_recommendation": loss,
                "agree": rec == deep_best,
            })

    n = len(details)
    srt = sorted(losses)
    return RecommendationEval(
        n_positions=n,
        verify_depth=verify_depth,
        working_depth=working_depth,
        agreement_at_1=agree1 / max(n, 1),
        agreement_at_2=agree2 / max(n, 1),
        mean_recommendation_cp_loss=sum(losses) / max(len(losses), 1),
        median_recommendation_cp_loss=srt[len(srt) // 2] if srt else 0.0,
        worse_than_100cp=sum(1 for l in losses if l >= 100),
        details=details,
    )
