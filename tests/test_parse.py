"""Parsing and data model tests."""
import chess
import pytest

from chessgraph.ingest.parse import derive_speed, classify_phase, material_signature
from chessgraph.models import position_key


def test_position_key_drops_move_counters():
    """The same structure via different move orders must give one key."""
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    same_later = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 7 42"
    assert position_key(start) == position_key(same_later)
    assert position_key(start).endswith("KQkq -")


def test_position_key_keeps_castling_and_en_passant():
    """Castling rights change the position, so they must stay in the key."""
    a = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    b = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kkq - 0 1"
    assert position_key(a) != position_key(b)


@pytest.mark.parametrize("tc,expected", [
    ("60+0", "bullet"),
    ("180+0", "blitz"),
    ("300+0", "blitz"),
    ("600+0", "rapid"),
    ("1800+0", "classical"),
    ("15+0", "ultrabullet"),
])
def test_derive_speed_buckets(tc, expected):
    assert derive_speed(tc) == expected


def test_derive_speed_counts_the_increment():
    """The increment moves a game across bucket boundaries.

    Estimated duration is base + 40 * increment. The boundary between bullet
    and blitz is 180 seconds, so with a 60 second base the increment decides:
    1+2 gives 140 and is bullet, 1+3 gives 180 and is blitz.
    """
    assert derive_speed("60+0") == "bullet"     # 60
    assert derive_speed("60+2") == "bullet"     # 60 + 80  = 140, under 180
    assert derive_speed("60+3") == "blitz"      # 60 + 120 = 180, at the line
    assert derive_speed("180+2") == "blitz"     # 180 + 80 = 260
    assert derive_speed("300+5") == "rapid"     # 300 + 200 = 500, over 480


def test_derive_speed_handles_missing_values():
    assert derive_speed(None) is None
    assert derive_speed("-") is None
    assert derive_speed("garbage") is None


def test_classify_phase():
    assert classify_phase(chess.Board()) == "opening"
    # Bare kings and pawns is an endgame.
    assert classify_phase(chess.Board("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 30")) == "endgame"


def test_material_signature_is_symmetric_at_the_start():
    sig = material_signature(chess.Board())
    white, black = sig.split("|")
    assert white == black
    assert white == "QRRBBNNPPPPPPPP"


def test_material_signature_reflects_a_capture():
    board = chess.Board()
    board.remove_piece_at(chess.D1)      # remove White's queen
    white, black = material_signature(board).split("|")
    assert "Q" not in white
    assert "Q" in black
