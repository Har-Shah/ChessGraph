"""PGN text -> GameRecord / MoveRecord / PositionRecord.

We lean on python-chess for the hard parts (SAN disambiguation, legality,
FEN generation). Our job is to decide *what to keep* and to attach the
"subject player" perspective that every later query depends on.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn

from chessgraph.models import (
    GameRecord, MoveRecord, PositionRecord, position_key,
)

RESULT_SCORE = {"1-0": (1.0, 0.0), "0-1": (0.0, 1.0), "1/2-1/2": (0.5, 0.5)}


def derive_speed(time_control: str | None) -> str | None:
    """Map a PGN TimeControl to Lichess's speed buckets.

    Lichess classifies by *estimated game duration* = base + 40 * increment,
    not by base time alone. A 1+2 game is blitz, not bullet, because those two
    seconds per move add up over 40 moves. Getting this right matters: bullet
    blunders are time-pressure artefacts and should usually be excluded from a
    weakness analysis, so misfiling 1+2 as bullet would silently drop real data.
    """
    if not time_control or time_control in ("-", "?"):
        return None
    try:
        base, _, inc = time_control.partition("+")
        estimated = int(base) + 40 * int(inc or 0)
    except ValueError:
        return None
    if estimated < 30:
        return "ultrabullet"
    if estimated < 180:
        return "bullet"
    if estimated < 480:
        return "blitz"
    if estimated < 1500:
        return "rapid"
    return "classical"


def classify_phase(board: chess.Board) -> str:
    """opening / middlegame / endgame from material and development.

    Standard heuristic: count non-pawn, non-king material. Fewer than ~7 such
    pieces on the board means endgame. Before move 12 with most pieces home,
    call it opening. This is coarse on purpose — it is a filter, not a claim.
    """
    pieces = board.piece_map()
    heavy = sum(
        1 for p in pieces.values()
        if p.piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    )
    if heavy <= 6:
        return "endgame"
    if board.fullmove_number <= 12:
        return "opening"
    return "middlegame"


def material_signature(board: chess.Board) -> str:
    """Compact material description, e.g. 'QRRBNPPPPP|RRBBNNPPPPPP'.

    Two positions with the same signature have the same material balance, which
    is a cheap first filter for "resembles" edges before we do anything
    expensive like embedding comparison.
    """
    order = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    letters = {chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B",
               chess.KNIGHT: "N", chess.PAWN: "P"}
    sides = []
    for color in (chess.WHITE, chess.BLACK):
        s = "".join(
            letters[pt] * len(board.pieces(pt, color)) for pt in order
        )
        sides.append(s)
    return "|".join(sides)


def _header_int(game: chess.pgn.Game, key: str) -> int | None:
    val = game.headers.get(key, "")
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def parse_game(game: chess.pgn.Game, subject: str | None = None
               ) -> tuple[GameRecord, list[MoveRecord]]:
    """Turn one parsed PGN game into our records."""
    h = game.headers
    site = h.get("Site", "")
    game_id = h.get("GameId") or site.rstrip("/").rsplit("/", 1)[-1] or "unknown"

    white, black = h.get("White", "?"), h.get("Black", "?")
    result = h.get("Result", "*")
    tc = h.get("TimeControl")

    moves: list[MoveRecord] = []
    board = game.board()
    ply = 0
    subject_lower = subject.lower() if subject else None
    subject_is_white = subject_lower == white.lower() if subject_lower else None

    for node in game.mainline():
        move = node.move
        ply += 1
        fen_before = board.fen()
        san = board.san(move)
        color = "white" if board.turn == chess.WHITE else "black"

        board.push(move)
        fen_after = board.fen()

        # python-chess reads the {[%clk ...]} and {[%eval ...]} annotations
        # for us. Both are optional: clocks exist on most Lichess games, evals
        # only on games where someone requested server-side analysis.
        clock = node.clock()
        pov_eval = node.eval()
        lichess_eval = None
        if pov_eval is not None:
            # Normalise to pawns from White's perspective, clamping mates to a
            # large finite number so downstream arithmetic never sees inf.
            score = pov_eval.white()
            lichess_eval = (score.score(mate_score=10000) or 0) / 100.0

        is_subject = (
            subject_is_white is not None
            and ((color == "white") == subject_is_white)
        )

        moves.append(MoveRecord(
            game_id=game_id,
            ply=ply,
            move_number=(ply + 1) // 2,
            color=color,
            san=san,
            uci=move.uci(),
            fen_before=fen_before,
            pos_key=position_key(fen_before),
            fen_after=fen_after,
            pos_key_after=position_key(fen_after),
            clock_seconds=int(clock) if clock is not None else None,
            lichess_eval=lichess_eval,
            is_subject_move=is_subject,
        ))

    w_elo, b_elo = _header_int(game, "WhiteElo"), _header_int(game, "BlackElo")
    rec = GameRecord(
        game_id=game_id,
        url=site,
        date=h.get("UTCDate", h.get("Date", "")).replace(".", "-"),
        white=white,
        black=black,
        white_elo=w_elo,
        black_elo=b_elo,
        result=result,
        eco=h.get("ECO"),
        opening=h.get("Opening"),
        time_control=tc,
        speed=derive_speed(tc),
        variant=h.get("Variant", "Standard"),
        termination=h.get("Termination"),
        event=h.get("Event"),
        ply_count=ply,
    )

    if subject_is_white is not None:
        w_score, b_score = RESULT_SCORE.get(result, (None, None))
        rec.subject = subject
        rec.subject_color = "white" if subject_is_white else "black"
        rec.subject_elo = w_elo if subject_is_white else b_elo
        rec.opponent = black if subject_is_white else white
        rec.opponent_elo = b_elo if subject_is_white else w_elo
        rec.subject_score = w_score if subject_is_white else b_score

    return rec, moves


def parse_pgn_file(path: Path, subject: str | None = None,
                   *, standard_only: bool = True
                   ) -> Iterator[tuple[GameRecord, list[MoveRecord]]]:
    """Stream games out of a PGN file one at a time.

    Streaming rather than read-all keeps memory flat whether the file holds
    500 games or 5 million, which is what lets the same parser serve both the
    API export and the full database dump.
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            if standard_only and game.headers.get("Variant", "Standard") != "Standard":
                continue
            if not list(game.mainline_moves()):
                continue  # aborted game, no moves
            yield parse_game(game, subject=subject)


def collect_positions(moves_by_game: list[list[MoveRecord]],
                      openings: dict[str, tuple[str | None, str | None]]
                      ) -> dict[str, PositionRecord]:
    """Fold every move's before-position into a deduplicated position table."""
    positions: dict[str, PositionRecord] = {}
    for moves in moves_by_game:
        for m in moves:
            existing = positions.get(m.pos_key)
            if existing is not None:
                existing.seen_count += 1
                continue
            board = chess.Board(m.fen_before)
            eco, opening = openings.get(m.game_id, (None, None))
            positions[m.pos_key] = PositionRecord(
                pos_key=m.pos_key,
                fen=m.fen_before,
                ply=m.ply,
                side_to_move=m.color,
                material_signature=material_signature(board),
                phase=classify_phase(board),
                eco=eco,
                opening=opening,
                seen_count=1,
            )
    return positions
