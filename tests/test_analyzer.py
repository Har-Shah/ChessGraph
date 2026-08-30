"""Tests for the centipawn loss convention and ACPL.

The sign convention is the single highest-risk piece of logic in the project,
so it gets tested against positions where the answer is not a matter of opinion.
"""
import tempfile
from pathlib import Path

import chess
import pytest

from chessgraph.engine.analyzer import (
    Analyzer, average_cp_loss, classify_loss, win_probability, CLAMP_CP,
)
from chessgraph.engine.cache import EvalCache
from chessgraph.models import MoveRecord, position_key


def _move(ply, color, cp_loss, judgment="mistake", is_subject=True):
    return MoveRecord(
        game_id="g", ply=ply, move_number=(ply + 1) // 2, color=color,
        san="Nf3", uci="g1f3", fen_before="", pos_key="", fen_after="",
        pos_key_after="", cp_loss=cp_loss, judgment=judgment,
        is_subject_move=is_subject,
    )


def test_classify_loss_thresholds():
    assert classify_loss(0) == "ok"
    assert classify_loss(49) == "ok"
    assert classify_loss(50) == "inaccuracy"
    assert classify_loss(99) == "inaccuracy"
    assert classify_loss(100) == "mistake"
    assert classify_loss(299) == "mistake"
    assert classify_loss(300) == "blunder"
    assert classify_loss(None) == "unknown"


def test_win_probability_is_monotonic_and_centred():
    assert win_probability(0) == pytest.approx(0.5)
    assert win_probability(100) > win_probability(0) > win_probability(-100)
    assert 0.0 < win_probability(-CLAMP_CP) < 0.1
    assert 0.9 < win_probability(CLAMP_CP) < 1.0


def test_average_cp_loss_excludes_book_moves():
    moves = [
        _move(2, "white", 500, judgment="book"),   # excluded: book
        _move(10, "white", 100),
        _move(12, "white", 200),
    ]
    assert average_cp_loss(moves) == pytest.approx(150.0)


def test_average_cp_loss_respects_colour_and_subject():
    moves = [
        _move(10, "white", 100, is_subject=True),
        _move(11, "black", 900, is_subject=False),
    ]
    assert average_cp_loss(moves, subject_only=True) == pytest.approx(100.0)
    assert average_cp_loss(moves, color="black") == pytest.approx(900.0)


def test_average_cp_loss_skips_early_plies():
    # Default min_ply is 8, so ply 4 must not count.
    moves = [_move(4, "white", 800, judgment="mistake"), _move(20, "white", 100)]
    assert average_cp_loss(moves) == pytest.approx(100.0)


@pytest.mark.slow
def test_cp_loss_sign_on_a_known_blunder():
    """Hanging the queen must produce a large POSITIVE loss for the mover.

    This is the regression guard for the perspective flip. If the sign
    convention ever inverts, this number goes sharply negative instead.

    The position is constructed rather than taken from an opening so there is
    no argument about compensation: White is winning, plays Qf5, and the g6
    pawn takes the queen for free. Qf5 is not a check, so Black is not forced
    into anything and the capture is a plain material win.
    """
    fen = "7k/8/6p1/8/8/8/5Q2/K7 w - - 0 1"
    board = chess.Board(fen)
    mv = board.parse_san("Qf5")
    assert mv in board.legal_moves

    move = MoveRecord(
        game_id="g", ply=20, move_number=10, color="white",
        san="Qf5", uci=mv.uci(), fen_before=fen,
        pos_key=position_key(fen), fen_after="", pos_key_after="",
        is_subject_move=True,
    )
    board.push(mv)
    move.fen_after = board.fen()
    move.pos_key_after = position_key(board.fen())
    assert not board.is_check(), "the refutation must not be forced by a check"
    assert board.parse_san("gxf5") in board.legal_moves

    with Analyzer() as az:
        az.annotate_game([move], depth=10, skip_opening_plies=0)

    assert move.cp_loss is not None
    assert move.cp_loss > 200, f"expected a large positive loss, got {move.cp_loss}"
    assert move.judgment == "blunder"
    assert move.best_move_san not in (None, "Qf5")


@pytest.mark.slow
def test_illegal_positions_are_rejected_before_reaching_the_engine():
    """Stockfish segfaults on illegal positions instead of rejecting them.

    In this FEN it is White to move while Black's king sits in check from Qe2
    down the open e-file, which cannot arise in a real game. Handing it to
    Stockfish 18 kills the process. The analyzer validates first, so the engine
    never sees it and no crash occurs.
    """
    illegal = "4k3/8/6p1/8/8/8/4Q3/4K3 w - - 0 1"
    assert not chess.Board(illegal).is_valid()

    with Analyzer() as az:
        result = az.evaluate(chess.Board(illegal), depth=10)
        assert result.failed is True
        assert az.engine_crashes == 0, "the guard should prevent a crash entirely"

        recovered = az.evaluate(chess.Board(), depth=10)
        assert recovered.failed is False
        assert recovered.best_san is not None


@pytest.mark.slow
def test_analyzer_recovers_when_the_engine_process_dies():
    """An engine killed mid-run must be restarted, not fatal.

    Simulated by killing the process directly, which covers the causes that
    have nothing to do with the position: the OS reclaiming memory, an
    external kill, a bad engine build.
    """
    # A private cache, so every evaluation below actually reaches the engine.
    # With the shared cache these positions are already stored and the recovery
    # path is never exercised, which is how an earlier version of this test
    # passed while testing nothing.
    with tempfile.TemporaryDirectory() as tmp:
        cache = EvalCache(Path(tmp) / "evals.sqlite")
        with Analyzer(cache=cache) as az:
            board = chess.Board()
            board.push_san("a3")
            board.push_san("h6")
            first = az.evaluate(board, depth=8)
            assert first.best_san is not None

            az._engine.close()      # simulate the process dying underneath us

            board.push_san("b3")
            recovered = az.evaluate(board, depth=8)
            assert recovered.failed is False
            assert recovered.best_san is not None
            assert az.engine_crashes >= 1
        # Leaving the context with a previously dead engine must not raise.


@pytest.mark.slow
def test_unanalysable_moves_are_left_unscored():
    """A failed evaluation must not be silently scored as equal."""
    illegal = "4k3/8/6p1/8/8/8/4Q3/4K3 w - - 0 1"
    board = chess.Board(illegal)
    mv = board.parse_san("Qh5")
    move = MoveRecord(
        game_id="g", ply=20, move_number=10, color="white",
        san="Qh5", uci=mv.uci(), fen_before=illegal,
        pos_key=position_key(illegal), fen_after="", pos_key_after="",
        is_subject_move=True,
    )
    board.push(mv)
    move.fen_after = board.fen()
    move.pos_key_after = position_key(board.fen())

    with Analyzer() as az:
        az.annotate_game([move], depth=10, skip_opening_plies=0)

    assert move.judgment == "unanalysed"
    assert move.cp_loss is None
