"""Central configuration.

Everything that a human might want to tune lives here, so no other module
hardcodes a path, a depth, or a threshold.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"       # downloaded PGN, untouched
DB = DATA / "db"         # sqlite files
CACHE = DATA / "cache"   # engine evals, embeddings — expensive to recompute

for _p in (RAW, DB, CACHE):
    _p.mkdir(parents=True, exist_ok=True)


def _find_stockfish() -> str:
    """Locate the engine binary. Env var wins, then PATH, then common spots."""
    if env := os.environ.get("STOCKFISH_PATH"):
        return env
    if found := shutil.which("stockfish"):
        return found
    for candidate in ("/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish"):
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "Stockfish not found. Install it (`brew install stockfish`) "
        "or set STOCKFISH_PATH."
    )


@dataclass(frozen=True)
class EngineConfig:
    """How hard the engine thinks.

    `depth` is the main quality/time dial. Depth 12 is ~50ms per position and
    is enough to catch blunders reliably; depth 18+ is for verifying a single
    critical position. `multipv` = how many candidate moves to return — we need
    at least 2 so we can say "you played X, but Y was better."
    """
    path: str = field(default_factory=_find_stockfish)
    depth: int = 12
    multipv: int = 2
    threads: int = 2
    hash_mb: int = 256
    # A move is a "mistake" if it costs at least this many centipawns.
    # These match the thresholds Lichess itself uses, so our labels are
    # comparable to public annotations.
    inaccuracy_cp: int = 50
    mistake_cp: int = 100
    blunder_cp: int = 300


@dataclass(frozen=True)
class IngestConfig:
    """Limits on what we pull down."""
    max_games: int = 500
    # Lichess variants/speeds we care about. Bullet games are noisy — mistakes
    # there are time pressure, not misunderstanding — so default to slower ones.
    perf_types: tuple[str, ...] = ("blitz", "rapid", "classical")
    rated_only: bool = True


@dataclass(frozen=True)
class AnalysisConfig:
    """Which parts of a game we bother analysing.

    Analysing all 80 plies of 500 games at depth 12 is ~40k positions. That is
    fine but slow on first run, so `max_ply` lets you cut games short. Opening
    prep questions mostly live in the first 30 moves anyway.
    """
    skip_opening_plies: int = 8    # book moves — no useful mistakes here
    max_ply: int = 60              # ~move 30 for each side
    min_ply_for_mistake: int = 8


ENGINE = EngineConfig()
INGEST = IngestConfig()
ANALYSIS = AnalysisConfig()
