"""Core record types.

The shape of these three objects determines what questions the system can
answer, so they are worth thinking about carefully.

GAME  , one played game. Metadata only; the moves live in MoveRecords.
MOVE  , one ply. This is the unit of *behaviour*: a player made a choice here.
POSITION, a board state, deduplicated across every game in the corpus.

The Move/Position split is the crux. A naive design stores moves inside games
and stops there, which can only answer "what happened in game X". By promoting
the position to its own entity keyed by FEN, the same board state reached in
40 different games collapses to one node, and questions like "every time this
player reaches this structure, what do they do?" become a lookup instead of a
scan. That single normalisation is what makes the knowledge graph meaningful
rather than decorative.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


def position_key(fen: str) -> str:
    """Identity of a board state, ignoring move counters.

    A full FEN ends with halfmove-clock and fullmove-number:
        rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
                                                            ^^^ drop these
    Two games reaching the same structure by different move orders must map to
    the same node, so we keep only placement / side-to-move / castling / en
    passant. This is the same key FIDE uses for threefold repetition.
    """
    return " ".join(fen.split(" ")[:4])


@dataclass
class GameRecord:
    game_id: str
    url: str
    date: str                  # ISO YYYY-MM-DD
    white: str
    black: str
    white_elo: Optional[int]
    black_elo: Optional[int]
    result: str                # "1-0" | "0-1" | "1/2-1/2"
    eco: Optional[str]         # e.g. "B02"
    opening: Optional[str]     # e.g. "Alekhine Defense: Sämisch Attack"
    time_control: Optional[str]
    speed: Optional[str]       # bullet/blitz/rapid/classical, derived
    variant: str
    termination: Optional[str]
    event: Optional[str]
    ply_count: int

    # --- Perspective fields -------------------------------------------------
    # Filled in relative to the player we are studying. Every downstream query
    # is "from this player's point of view", so computing it once here saves a
    # colour-flip bug in every consumer.
    subject: Optional[str] = None        # the username we ingested for
    subject_color: Optional[str] = None  # "white" | "black"
    subject_elo: Optional[int] = None
    opponent: Optional[str] = None
    opponent_elo: Optional[int] = None
    subject_score: Optional[float] = None  # 1.0 win, 0.5 draw, 0.0 loss

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MoveRecord:
    """One ply: the position before the move, and the move played from it."""
    game_id: str
    ply: int                   # 1-based; ply 1 is White's first move
    move_number: int           # chess move number (1. e4 e5 -> both are 1)
    color: str                 # "white" | "black", who is moving
    san: str                   # "Nf3"
    uci: str                   # "g1f3"
    fen_before: str            # full FEN of the position being moved from
    pos_key: str               # deduplicated position identity
    fen_after: str
    pos_key_after: str
    clock_seconds: Optional[int] = None   # time left after the move
    lichess_eval: Optional[float] = None  # pawns, White's perspective, if present
    is_subject_move: bool = False         # did the player we study make this?

    # Filled in by the engine pass (Phase 2), left None until then.
    eval_before_cp: Optional[int] = None   # cp, side-to-move perspective
    eval_after_cp: Optional[int] = None
    cp_loss: Optional[int] = None          # how much the move threw away
    best_move_san: Optional[str] = None
    best_move_uci: Optional[str] = None
    judgment: Optional[str] = None         # ok|inaccuracy|mistake|blunder
    mate_in: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PositionRecord:
    """A board state seen at least once in the corpus.

    Deduplicated by pos_key. `seen_count` etc. are aggregates maintained at
    write time so we never have to scan the move table to answer "how often
    does this player reach this position".
    """
    pos_key: str
    fen: str                   # a representative full FEN
    ply: int                   # representative ply it was first seen at
    side_to_move: str
    material_signature: str = ""   # e.g. "KQRRBBNNPPPPPPPP" per side
    phase: str = "opening"         # opening | middlegame | endgame
    eco: Optional[str] = None
    opening: Optional[str] = None
    seen_count: int = 0
    features: dict = field(default_factory=dict)  # structural tags, Phase 3

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
