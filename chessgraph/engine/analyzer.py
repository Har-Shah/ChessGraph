"""Stockfish analysis: evaluate positions, score mistakes, suggest better moves.

THE CENTRAL IDEA, how you measure a mistake
--------------------------------------------
An engine evaluation is always *from the perspective of the side to move*.
"+0.5" after 1.e4 means "good for White"; the same +0.5 after 1...e5 means
"good for Black". Every sign bug in every amateur chess-analysis project comes
from forgetting this.

The naive approach is to evaluate a position, then evaluate it again after the
played move, and subtract. That is two engine searches per ply and it is easy
to get backwards.

The efficient and correct approach evaluates each position exactly once and
reads the loss off *consecutive* evaluations:

    Let e[i] = engine eval of position i, from the perspective of whoever
               moves at position i.

    At position i, the mover's best achievable outcome is e[i].
    After they play, we are at position i+1 where the OPPONENT moves, and the
    opponent's eval is e[i+1]. From the original mover's perspective, what they
    actually achieved is -e[i+1].

    centipawn loss = e[i] - (-e[i+1]) = e[i] + e[i+1]

A game of N moves needs N+1 evaluations, not 2N, and the sign convention
collapses into a single addition, which is much harder to get wrong.

CLAMPING, why we cap evals at +/-1000
--------------------------------------
Raw centipawns are a terrible loss scale at the extremes. Going from "mate in
3" to "mate in 8" is a 0-centipawn practical difference (still winning) but
thousands of raw centipawns. Going from +0.2 to -0.8 is one full pawn on paper
and a completely different game in practice. So we clamp evaluations into
[-1000, +1000] before differencing, which is what Lichess does, and we ALSO
record win-probability loss, which handles the extremes correctly by construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import chess
import chess.engine

from chessgraph.config import ENGINE, ANALYSIS, EngineConfig
from chessgraph.engine.cache import EvalCache
from chessgraph.models import MoveRecord, position_key

# Evals beyond this are practically identical (all winning / all lost).
CLAMP_CP = 1000


@dataclass
class PositionEval:
    pos_key: str
    score_cp: Optional[int]      # side-to-move perspective, None if terminal
    mate: Optional[int]          # signed plies to mate
    best_uci: Optional[str]
    best_san: Optional[str]
    pv: str
    alternatives: list[dict]
    terminal: bool = False       # checkmate or stalemate: nothing to play
    failed: bool = False         # engine could not analyse this position

    def clamped_cp(self) -> int:
        """A single comparable number, mates folded in."""
        if self.mate is not None:
            return CLAMP_CP if self.mate > 0 else -CLAMP_CP
        if self.score_cp is None:
            return 0
        return max(-CLAMP_CP, min(CLAMP_CP, self.score_cp))


def win_probability(cp: int) -> float:
    """Convert a centipawn eval to expected score in [0, 1].

    This is Lichess's fitted logistic. It exists because centipawns are not
    linear in practical terms: the difference between +0 and +100 changes the
    result far more often than the difference between +900 and +1000. When we
    report "how much did this move cost you", win-probability loss is the more
    honest number; centipawn loss is the one everybody quotes.
    """
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


class Analyzer:
    """A managed Stockfish process with a persistent evaluation cache.

    Use as a context manager. Starting the engine costs ~100ms, so we start it
    once and keep it warm; a fresh process per position would dominate runtime.
    """

    def __init__(self, config: EngineConfig = ENGINE, cache: EvalCache | None = None):
        self.config = config
        self.cache = cache if cache is not None else EvalCache()
        self._engine: chess.engine.SimpleEngine | None = None
        self.nodes_searched = 0
        self.engine_crashes = 0
        self.crashed_positions: list[str] = []

    def _start_engine(self) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(self.config.path)
        self._engine.configure({
            "Threads": self.config.threads,
            "Hash": self.config.hash_mb,
        })

    def __enter__(self) -> "Analyzer":
        self._start_engine()
        return self

    def __exit__(self, *exc):
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                # The engine may already be dead, in which case quit() raises
                # EngineTerminatedError. Cleanup must not turn a successful run
                # into a failure on the way out, and the process is gone either
                # way, so there is nothing left to clean up.
                pass
            self._engine = None
        self.cache.commit()

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, board: chess.Board, *, depth: int | None = None
                 ) -> PositionEval:
        """Evaluate one position, hitting the cache when possible."""
        depth = depth or self.config.depth
        key = position_key(board.fen())

        if board.is_game_over(claim_draw=False):
            # The engine refuses to search a finished position, and rightly so.
            # Checkmate is -MATE for the side to move (they are the one mated);
            # stalemate and other draws are dead level.
            return PositionEval(
                pos_key=key,
                score_cp=0 if not board.is_checkmate() else -CLAMP_CP,
                mate=0 if board.is_checkmate() else None,
                best_uci=None, best_san=None, pv="", alternatives=[],
                terminal=True,
            )

        cached = self.cache.get(key, depth, self.config.multipv)
        if cached is not None:
            return PositionEval(
                pos_key=key,
                score_cp=cached["score_cp"],
                mate=cached["mate"],
                best_uci=cached["best_uci"],
                best_san=cached["best_san"],
                pv=cached["pv"] or "",
                alternatives=cached["alternatives"],
            )

        if not board.is_valid():
            # Stockfish segfaults on illegal positions rather than rejecting
            # them, so validate before handing anything over. Real game data is
            # always legal; this catches hand-built FENs and construction bugs.
            return PositionEval(key, None, None, None, None, "", [], failed=True)

        if self._engine is None:
            raise RuntimeError("Analyzer must be used as a context manager")

        # Defence in depth against the engine dying mid-run. The known trigger
        # is an ILLEGAL position: Stockfish segfaults when handed one where the
        # side not to move is in check, which the guard above already rejects.
        # Positions parsed from real games are always legal, so this path
        # should never fire on pipeline data. It exists because an engine can
        # also die for reasons that have nothing to do with the position (the
        # OS killing it under memory pressure, an external kill, a bad build),
        # and losing a 17 minute analysis run to that would be avoidable pain.
        # Restart once and retry; if it dies again, mark the position
        # unanalysable and keep going.
        infos = None
        for attempt in range(2):
            try:
                infos = self._engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=self.config.multipv,
                )
                break
            except chess.engine.EngineTerminatedError:
                self.engine_crashes += 1
                self.crashed_positions.append(board.fen())
                try:
                    self._start_engine()
                except Exception:
                    raise
        if infos is None:
            # Reproducible crash on this position. Return a failed marker so
            # annotate_game leaves the surrounding moves unscored rather than
            # silently treating the position as equal.
            return PositionEval(key, None, None, None, None, "", [], failed=True)

        if isinstance(infos, dict):       # multipv=1 returns a bare dict
            infos = [infos]

        alternatives = []
        for info in infos:
            pv_moves = info.get("pv") or []
            if not pv_moves:
                continue
            # .pov(board.turn) forces the score into the side-to-move frame,
            # regardless of how the engine reported it.
            pov = info["score"].pov(board.turn)
            alternatives.append({
                "uci": pv_moves[0].uci(),
                "san": board.san(pv_moves[0]),
                "score_cp": pov.score(),          # None when it is a mate
                "mate": pov.mate(),
                "pv": " ".join(m.uci() for m in pv_moves[:8]),
            })
            self.nodes_searched += info.get("nodes", 0) or 0

        if not alternatives:
            return PositionEval(key, 0, None, None, None, "", [], terminal=True)

        top = alternatives[0]
        result = PositionEval(
            pos_key=key,
            score_cp=top["score_cp"],
            mate=top["mate"],
            best_uci=top["uci"],
            best_san=top["san"],
            pv=top["pv"],
            alternatives=alternatives,
        )
        self.cache.put(
            key, depth, self.config.multipv,
            score_cp=result.score_cp, mate=result.mate,
            best_uci=result.best_uci, best_san=result.best_san,
            pv=result.pv, alternatives=alternatives,
        )
        return result

    # ------------------------------------------------------------ annotate game
    def annotate_game(self, moves: list[MoveRecord], *,
                      depth: int | None = None,
                      max_ply: int | None = None,
                      skip_opening_plies: int | None = None) -> list[MoveRecord]:
        """Fill in eval / cp_loss / best_move / judgment on each MoveRecord.

        Mutates and returns the list. See the module docstring for why the loss
        is `e[i] + e[i+1]` rather than a difference.
        """
        depth = depth or self.config.depth
        max_ply = max_ply if max_ply is not None else ANALYSIS.max_ply
        skip = (skip_opening_plies if skip_opening_plies is not None
                else ANALYSIS.skip_opening_plies)

        window = [m for m in moves if m.ply <= max_ply]
        if not window:
            return moves

        # N+1 evaluations: every position moved FROM, plus the final position.
        evals: list[PositionEval] = []
        for m in window:
            evals.append(self.evaluate(chess.Board(m.fen_before), depth=depth))
        evals.append(self.evaluate(chess.Board(window[-1].fen_after), depth=depth))

        for i, m in enumerate(window):
            before, after = evals[i], evals[i + 1]
            if before.failed or after.failed:
                # No trustworthy evaluation on one side of this move, so any
                # loss we computed would be fiction. Leave it unscored.
                m.judgment = "unanalysed"
                continue
            m.eval_before_cp = before.score_cp
            m.mate_in = before.mate
            m.best_move_uci = before.best_uci
            m.best_move_san = before.best_san
            # after.clamped_cp() is from the OPPONENT's perspective; negate it
            # to express the outcome the mover actually achieved.
            achieved = -after.clamped_cp()
            m.eval_after_cp = achieved
            loss = before.clamped_cp() - achieved
            # Small negatives happen from search noise at fixed depth (the
            # deeper reply search sees something the shallower one missed).
            # Floor at zero: you cannot gain by moving if the engine was right.
            m.cp_loss = max(0, loss)

            if m.ply < skip:
                # Book moves. The engine will happily call a mainline Sicilian
                # move an "inaccuracy" at depth 12; that is engine noise, not a
                # player weakness, and labelling it would poison the whole
                # weakness analysis downstream.
                m.judgment = "book"
            elif m.best_move_uci == m.uci:
                m.judgment = "best"
            else:
                m.judgment = classify_loss(m.cp_loss)

        return moves


def classify_loss(cp_loss: int | None, cfg: EngineConfig = ENGINE) -> str:
    """Bucket a centipawn loss into a human label.

    Thresholds match Lichess's, so our labels are directly comparable to the
    public annotations on the site, which matters for evaluation, where we
    want to check our numbers against an independent source.
    """
    if cp_loss is None:
        return "unknown"
    if cp_loss >= cfg.blunder_cp:
        return "blunder"
    if cp_loss >= cfg.mistake_cp:
        return "mistake"
    if cp_loss >= cfg.inaccuracy_cp:
        return "inaccuracy"
    return "ok"


def average_cp_loss(moves: Iterable[MoveRecord], *, color: str | None = None,
                    subject_only: bool = False, min_ply: int | None = None) -> float:
    """ACPL, the standard single-number strength proxy.

    Excludes book moves by default so opening theory does not flatter the
    number. Typical values: <20 grandmaster, 20-40 strong club player,
    40-80 intermediate, 80+ beginner.
    """
    min_ply = min_ply if min_ply is not None else ANALYSIS.min_ply_for_mistake
    vals = [
        m.cp_loss for m in moves
        if m.cp_loss is not None
        and m.ply >= min_ply
        and m.judgment != "book"
        and (color is None or m.color == color)
        and (not subject_only or m.is_subject_move)
    ]
    return sum(vals) / len(vals) if vals else 0.0
