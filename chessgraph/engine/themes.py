"""Tactical theme detection: turning a blunder into a *named, recurring* pattern.

WHY THIS MODULE IS THE HARD PART
--------------------------------
"Player struggles with X" is only useful if X is a real, checkable category.
Most projects fake this by asking an LLM to name the theme, which produces
plausible labels that cannot be verified and do not aggregate — you get 200
slightly different phrasings of "tactical oversight" and can never count them.

So every detector here is *mechanical*: it inspects the board with python-chess
and returns a label only when a specific geometric or material condition holds.
That means the labels are:
  - reproducible (same position -> same label, always)
  - countable (200 blunders collapse into 8 themes you can rank)
  - falsifiable (you can open the position and check we were right)

The method: when a player blunders, the engine tells us the refutation. We play
out that refutation line and ask two questions —
  1. What did it COST? (material swing over the forced sequence)
  2. HOW did it work? (geometry of the refuting move: fork, pin, discovery...)
Question 2 is the theme. Question 1 is the severity.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

PIECE_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 320,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}
# A "valuable" target for fork purposes — forking two pawns is not a theme.
FORK_TARGETS = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def material(board: chess.Board, color: chess.Color) -> int:
    return sum(
        PIECE_VALUE[pt] * len(board.pieces(pt, color))
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def material_balance(board: chess.Board, color: chess.Color) -> int:
    return material(board, color) - material(board, not color)


@dataclass
class ThemeResult:
    themes: list[str]
    material_swing: int      # cp of material the blunderer loses in the line
    refutation_san: str | None
    detail: dict


# ------------------------------------------------------------------ detectors
def _detect_fork(board_after_refutation: chess.Board, to_square: int,
                 victim_color: chess.Color) -> bool:
    """The refuting piece now attacks two or more valuable enemy units.

    Note we check attacks FROM the landing square only. A piece that happened
    to already attack two things is not a fork — the fork is created by this
    move arriving on this square.
    """
    attacked = []
    for sq in board_after_refutation.attacks(to_square):
        piece = board_after_refutation.piece_at(sq)
        if piece and piece.color == victim_color and piece.piece_type in FORK_TARGETS:
            attacked.append(piece.piece_type)
    return len(attacked) >= 2


def _detect_pin_or_skewer(board: chess.Board, to_square: int,
                          victim_color: chess.Color) -> str | None:
    """A sliding piece lands on a line with two enemy pieces behind each other.

    Pin vs skewer is decided by which is worth more: if the FRONT piece is
    worth less than the one behind it, the front one is pinned; if the front
    one is worth more, it is a skewer. This distinction matters to a student —
    they are different things to practise.
    """
    piece = board.piece_at(to_square)
    if not piece or piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return None

    directions = {
        chess.BISHOP: [9, 7, -7, -9],
        chess.ROOK: [8, 1, -1, -8],
        chess.QUEEN: [9, 8, 7, 1, -1, -7, -8, -9],
    }[piece.piece_type]

    for delta in directions:
        found: list[tuple[int, chess.Piece]] = []
        sq = to_square
        prev_file = chess.square_file(sq)
        while True:
            sq += delta
            if not (0 <= sq < 64):
                break
            # Guard against wrapping around the board edge.
            f = chess.square_file(sq)
            if abs(delta) in (1, 7, 9) and abs(f - prev_file) > 1:
                break
            prev_file = f
            occupant = board.piece_at(sq)
            if occupant:
                if occupant.color != victim_color:
                    break                     # own piece blocks the line
                found.append((sq, occupant))
                if len(found) == 2:
                    front_val = PIECE_VALUE[found[0][1].piece_type]
                    back_val = PIECE_VALUE[found[1][1].piece_type]
                    if found[1][1].piece_type == chess.KING:
                        return "absolute_pin"
                    if back_val > front_val:
                        return "pin"
                    if front_val > back_val:
                        return "skewer"
                    break
    return None


def _detect_discovered_attack(board_before: chess.Board, move: chess.Move,
                              victim_color: chess.Color) -> bool:
    """Moving a piece uncovers an attack from a piece that was behind it."""
    after = board_before.copy()
    after.push(move)
    mover_color = board_before.turn

    # Enemy pieces newly attacked by units OTHER than the one that just moved.
    def attacked_set(b: chess.Board, exclude: int | None) -> set[int]:
        out = set()
        for sq in chess.SQUARES:
            p = b.piece_at(sq)
            if not p or p.color != mover_color or sq == exclude:
                continue
            for target in b.attacks(sq):
                tp = b.piece_at(target)
                if tp and tp.color == victim_color and tp.piece_type in FORK_TARGETS:
                    out.add(target)
        return out

    before_set = attacked_set(board_before, move.from_square)
    after_set = attacked_set(after, move.to_square)
    return bool(after_set - before_set)


def _detect_back_rank(board: chess.Board, victim_color: chess.Color) -> bool:
    """Victim's king is on its back rank, boxed in by its own pawns."""
    king_sq = board.king(victim_color)
    if king_sq is None:
        return False
    back_rank = 0 if victim_color == chess.WHITE else 7
    if chess.square_rank(king_sq) != back_rank:
        return False
    # Are the three squares in front of the king occupied by own pawns?
    step = 8 if victim_color == chess.WHITE else -8
    blocked = 0
    for df in (-1, 0, 1):
        f = chess.square_file(king_sq) + df
        if not 0 <= f <= 7:
            continue
        sq = king_sq + step + df
        if 0 <= sq < 64:
            p = board.piece_at(sq)
            if p and p.color == victim_color and p.piece_type == chess.PAWN:
                blocked += 1
    return blocked >= 2


def _detect_hanging(board_after_blunder: chess.Board,
                    victim_color: chess.Color) -> list[str]:
    """Victim pieces that are attacked and insufficiently defended.

    Cheap approximation of a static exchange evaluation: a piece is hanging if
    it is attacked by something cheaper than itself, or attacked at all while
    undefended. This misses some deep exchanges but catches the overwhelming
    majority of real blunders at club level, which is what we care about.
    """
    hanging = []
    for sq in chess.SQUARES:
        p = board_after_blunder.piece_at(sq)
        if not p or p.color != victim_color or p.piece_type == chess.KING:
            continue
        attackers = board_after_blunder.attackers(not victim_color, sq)
        if not attackers:
            continue
        defenders = board_after_blunder.attackers(victim_color, sq)
        cheapest_attacker = min(
            (PIECE_VALUE[board_after_blunder.piece_at(a).piece_type] for a in attackers),
            default=10_000,
        )
        if not defenders or cheapest_attacker < PIECE_VALUE[p.piece_type]:
            hanging.append(chess.piece_name(p.piece_type))
    return hanging


# ------------------------------------------------------------------- main API
def detect_themes(fen_before: str, played_uci: str, refutation_pv: str,
                  *, phase: str = "middlegame",
                  cp_loss: int | None = None) -> ThemeResult:
    """Classify one blunder.

    fen_before     — position the blunderer moved from
    played_uci     — the move they actually played
    refutation_pv  — engine PV from the resulting position (space-separated UCI)
    cp_loss        — engine-measured severity, if known. Needed to separate
                     "lost material" from "won material and lost the game",
                     which look identical on a material count alone.

    material_swing convention: POSITIVE means the blunderer LOST material.
    A negative swing on a move the engine hates means they grabbed material
    and walked into something — a sacrifice accepted, not a piece dropped.
    """
    board = chess.Board(fen_before)
    blunderer = board.turn
    themes: list[str] = []
    detail: dict = {}

    try:
        played = chess.Move.from_uci(played_uci)
        if played not in board.legal_moves:
            return ThemeResult([], 0, None, {"error": "illegal move"})
        board.push(played)
    except ValueError:
        return ThemeResult([], 0, None, {"error": "bad uci"})

    pv_moves = [m for m in (refutation_pv or "").split() if m]
    if not pv_moves:
        # No engine line available. We can still report what is hanging.
        hanging = _detect_hanging(board, blunderer)
        if hanging:
            themes.append("hanging_piece")
            detail["hanging"] = hanging
        return ThemeResult(themes, 0, None, detail)

    # --- 1. What did it cost? Play out the forced line and measure material.
    line_board = board.copy()
    balance_start = material_balance(line_board, blunderer)
    refutation_san = None
    try:
        first = chess.Move.from_uci(pv_moves[0])
        if first in line_board.legal_moves:
            refutation_san = line_board.san(first)
    except ValueError:
        pass

    for uci in pv_moves[:8]:
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            break
        if mv not in line_board.legal_moves:
            break
        line_board.push(mv)
    material_swing = balance_start - material_balance(line_board, blunderer)

    # --- 2. How did the refutation work? Geometry of the refuting move.
    refute_board = board.copy()
    try:
        refute = chess.Move.from_uci(pv_moves[0])
    except ValueError:
        refute = None

    if refute and refute in refute_board.legal_moves:
        is_capture = refute_board.is_capture(refute)
        if _detect_discovered_attack(refute_board, refute, blunderer):
            themes.append("discovered_attack")
        refute_board.push(refute)

        if _detect_fork(refute_board, refute.to_square, blunderer):
            themes.append("fork")
        if pin := _detect_pin_or_skewer(refute_board, refute.to_square, blunderer):
            themes.append(pin)
        if refute_board.is_checkmate() and _detect_back_rank(board, blunderer):
            themes.append("back_rank_mate")
        elif _detect_back_rank(refute_board, blunderer) and refute_board.is_check():
            themes.append("back_rank_weakness")
        if is_capture and material_swing >= 100:
            themes.append("hangs_material")
        detail["refutation_is_capture"] = is_capture

    hanging = _detect_hanging(board, blunderer)
    if hanging and "hangs_material" not in themes:
        themes.append("hanging_piece")
    if hanging:
        detail["hanging"] = hanging

    # --- 3. Contextual themes: not geometry, but still real categories.
    if phase == "endgame":
        themes.append("endgame_technique")
    if board.is_check():
        themes.append("king_safety")

    # Won material but the evaluation collapsed anyway. This is a distinct and
    # very common club-level weakness: taking whatever is offered without
    # checking what it opens up. Without cp_loss these look exactly like a
    # sound capture on the material count, which is why it is passed in.
    if cp_loss is not None and cp_loss >= 100 and material_swing <= -100:
        themes.append("accepts_unsound_sacrifice")
        detail["material_gained"] = -material_swing

    if not themes:
        # Positional error: no material lost, no motif — the move just made the
        # position worse. Honest label rather than a made-up tactical one.
        themes.append("positional_error")

    # Deduplicate, preserve order.
    seen, ordered = set(), []
    for t in themes:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    return ThemeResult(ordered, material_swing, refutation_san, detail)
