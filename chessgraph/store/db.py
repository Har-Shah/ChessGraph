"""SQLite storage: the system of record.

Why SQLite and not Postgres/Neo4j for the facts layer?
  A single-player corpus is ~500 games / 40k moves / 25k positions. That is a
  few tens of MB. SQLite handles it with zero operational overhead, it is a
  single file you can copy or delete to reset an experiment, and it gives us
  real indexes and joins. The knowledge graph (Phase 3) is built ON TOP of this
  table — the graph is a *view* optimised for traversal, not a second source of
  truth. Keeping one authoritative store and deriving the graph from it means
  the two can never disagree.

Index choices below are driven by the actual queries in the retrieval layer,
not sprinkled hopefully. Each one is annotated with the query it serves.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Iterable, Sequence

from chessgraph.config import DB
from chessgraph.models import GameRecord, MoveRecord, PositionRecord

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS games (
    game_id       TEXT PRIMARY KEY,
    url           TEXT,
    date          TEXT,
    white         TEXT,
    black         TEXT,
    white_elo     INTEGER,
    black_elo     INTEGER,
    result        TEXT,
    eco           TEXT,
    opening       TEXT,
    time_control  TEXT,
    speed         TEXT,
    variant       TEXT,
    termination   TEXT,
    event         TEXT,
    ply_count     INTEGER,
    subject       TEXT,
    subject_color TEXT,
    subject_elo   INTEGER,
    opponent      TEXT,
    opponent_elo  INTEGER,
    subject_score REAL
);

CREATE TABLE IF NOT EXISTS moves (
    game_id        TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    ply            INTEGER NOT NULL,
    move_number    INTEGER,
    color          TEXT,
    san            TEXT,
    uci            TEXT,
    fen_before     TEXT,
    pos_key        TEXT,
    fen_after      TEXT,
    pos_key_after  TEXT,
    clock_seconds  INTEGER,
    lichess_eval   REAL,
    is_subject_move INTEGER,
    eval_before_cp INTEGER,
    eval_after_cp  INTEGER,
    cp_loss        INTEGER,
    best_move_san  TEXT,
    best_move_uci  TEXT,
    judgment       TEXT,
    mate_in        INTEGER,
    PRIMARY KEY (game_id, ply)
);

CREATE TABLE IF NOT EXISTS positions (
    pos_key            TEXT PRIMARY KEY,
    fen                TEXT,
    ply                INTEGER,
    side_to_move       TEXT,
    material_signature TEXT,
    phase              TEXT,
    eco                TEXT,
    opening            TEXT,
    seen_count         INTEGER,
    features           TEXT
);

-- "which openings does this player play as White?"
CREATE INDEX IF NOT EXISTS ix_games_subject ON games(subject, subject_color);
-- "all games in the Alekhine" / opening-prediction eval
CREATE INDEX IF NOT EXISTS ix_games_eco     ON games(eco);
CREATE INDEX IF NOT EXISTS ix_games_opening ON games(opening);
-- held-out splits by date
CREATE INDEX IF NOT EXISTS ix_games_date    ON games(date);
-- "every mistake this player made"  <- the single hottest query in the system
CREATE INDEX IF NOT EXISTS ix_moves_judg    ON moves(judgment, is_subject_move);
-- "every game that passed through this position" <- graph edge construction
CREATE INDEX IF NOT EXISTS ix_moves_poskey  ON moves(pos_key);
CREATE INDEX IF NOT EXISTS ix_moves_game    ON moves(game_id, ply);
CREATE INDEX IF NOT EXISTS ix_pos_opening   ON positions(opening);
CREATE INDEX IF NOT EXISTS ix_pos_phase     ON positions(phase);
"""


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else (DB / "chessgraph.sqlite")
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ write
    def _insert_many(self, table: str, cls, records: Sequence) -> int:
        if not records:
            return 0
        cols = [f.name for f in dc_fields(cls)]
        placeholders = ",".join("?" * len(cols))
        rows = []
        for r in records:
            d = r.to_dict()
            rows.append(tuple(
                json.dumps(d[c]) if isinstance(d[c], (dict, list))
                else (int(d[c]) if isinstance(d[c], bool) else d[c])
                for c in cols
            ))
        self.conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
            rows,
        )
        return len(rows)

    def add_games(self, games: Sequence[GameRecord]) -> int:
        return self._insert_many("games", GameRecord, games)

    def add_moves(self, moves: Sequence[MoveRecord]) -> int:
        return self._insert_many("moves", MoveRecord, moves)

    def add_positions(self, positions: Iterable[PositionRecord]) -> int:
        """Upsert positions, summing seen_count rather than overwriting it.

        Positions arrive in batches across many games, so a blind REPLACE would
        reset the count each time and silently break every frequency-based
        query downstream.
        """
        n = 0
        for p in positions:
            self.conn.execute(
                """INSERT INTO positions
                     (pos_key, fen, ply, side_to_move, material_signature,
                      phase, eco, opening, seen_count, features)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(pos_key) DO UPDATE SET
                     seen_count = seen_count + excluded.seen_count""",
                (p.pos_key, p.fen, p.ply, p.side_to_move, p.material_signature,
                 p.phase, p.eco, p.opening, p.seen_count, json.dumps(p.features)),
            )
            n += 1
        return n

    def commit(self):
        self.conn.commit()

    # ------------------------------------------------------------------- read
    def q(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple = ()):
        row = self.conn.execute(sql, params).fetchone()
        return row

    def stats(self) -> dict:
        return {
            "games": self.one("SELECT COUNT(*) c FROM games")["c"],
            "moves": self.one("SELECT COUNT(*) c FROM moves")["c"],
            "positions": self.one("SELECT COUNT(*) c FROM positions")["c"],
            "analyzed_moves": self.one(
                "SELECT COUNT(*) c FROM moves WHERE cp_loss IS NOT NULL")["c"],
            "subjects": [r["subject"] for r in self.q(
                "SELECT DISTINCT subject FROM games WHERE subject IS NOT NULL")],
        }

    def close(self):
        self.conn.commit()
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
