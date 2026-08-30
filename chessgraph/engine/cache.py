"""Persistent cache for engine evaluations.

Engine analysis is by far the most expensive thing this system does, roughly
50-200ms per position, and a 500-game corpus has ~40,000 positions. That is 30+
minutes of CPU you do not want to repeat every time you tweak a report.

Two properties make caching unusually effective here:
  1. Positions deduplicate hard. Every game starts from the same position, and
     a player with a narrow repertoire replays the same first 15 plies
     constantly. Real hit rates on opening-heavy corpora run 40-70%.
  2. Evaluations are deterministic for a fixed engine + depth, so a cache hit
     is exactly as good as a recomputation.

Keyed on (pos_key, depth, multipv) so raising the depth does not silently serve
you shallow numbers.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from chessgraph.config import CACHE

SCHEMA = """
CREATE TABLE IF NOT EXISTS position_eval (
    pos_key      TEXT NOT NULL,
    depth        INTEGER NOT NULL,
    multipv      INTEGER NOT NULL,
    score_cp     INTEGER,      -- centipawns, side-to-move perspective
    mate         INTEGER,      -- plies to mate, signed; NULL if no forced mate
    best_uci     TEXT,
    best_san     TEXT,
    pv           TEXT,         -- principal variation, space-separated UCI
    alternatives TEXT,         -- JSON list of {uci, san, score_cp, mate}
    engine       TEXT,
    PRIMARY KEY (pos_key, depth, multipv)
);
"""


class EvalCache:
    def __init__(self, path: Path | None = None):
        self.path = path or (CACHE / "evals.sqlite")
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        # WAL lets a reader and a writer coexist, which matters once we
        # parallelise analysis across processes.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.hits = 0
        self.misses = 0

    def get(self, pos_key: str, depth: int, multipv: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM position_eval WHERE pos_key=? AND depth=? AND multipv=?",
            (pos_key, depth, multipv),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        d = dict(row)
        d["alternatives"] = json.loads(d["alternatives"] or "[]")
        return d

    def put(self, pos_key: str, depth: int, multipv: int, **fields) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO position_eval
               (pos_key, depth, multipv, score_cp, mate, best_uci, best_san,
                pv, alternatives, engine)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                pos_key, depth, multipv,
                fields.get("score_cp"), fields.get("mate"),
                fields.get("best_uci"), fields.get("best_san"),
                fields.get("pv"),
                json.dumps(fields.get("alternatives", [])),
                fields.get("engine", "stockfish"),
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def size(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM position_eval").fetchone()[0]

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
