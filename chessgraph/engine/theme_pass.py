"""Second analysis pass: label every mistake with tactical themes.

Runs AFTER the Stockfish pass, and needs no engine of its own — it reuses the
principal variations already sitting in the eval cache.

The join that makes this work:
    moves.pos_key_after  ->  position_eval.pos_key

A move's *refutation* is the best line from the position it created, which is
exactly the PV we cached when we evaluated that position to compute cp_loss.
So the expensive work is already done; this pass is pure geometry over data we
have, and takes seconds rather than minutes.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from rich.console import Console

from chessgraph.config import CACHE, ENGINE
from chessgraph.engine.themes import detect_themes
from chessgraph.store.db import Store

console = Console()


def run_theme_pass(store: Store, *, cache_path: Path | None = None,
                   min_cp_loss: int = 100, depth: int | None = None) -> dict:
    """Label every move losing >= min_cp_loss with its tactical themes."""
    cache_path = cache_path or (CACHE / "evals.sqlite")
    depth = depth or ENGINE.depth

    # ATTACH lets SQLite join across two database files in one query, which
    # beats pulling both tables into Python and joining by hand.
    store.conn.execute("ATTACH DATABASE ? AS evalcache", (str(cache_path),))
    try:
        rows = store.conn.execute(
            """
            SELECT m.game_id, m.ply, m.fen_before, m.uci, m.cp_loss,
                   m.is_subject_move, p.phase, e.pv
            FROM moves m
            JOIN positions p ON p.pos_key = m.pos_key
            LEFT JOIN evalcache.position_eval e
                   ON e.pos_key = m.pos_key_after AND e.depth = ?
            WHERE m.cp_loss >= ?
            """,
            (depth, min_cp_loss),
        ).fetchall()
    finally:
        store.conn.execute("DETACH DATABASE evalcache")

    console.print(f"  labelling {len(rows)} mistakes...")
    inserts, missing_pv, counts = [], 0, {}
    for r in rows:
        if not r["pv"]:
            missing_pv += 1
        result = detect_themes(
            r["fen_before"], r["uci"], r["pv"] or "",
            phase=r["phase"] or "middlegame",
            cp_loss=r["cp_loss"],
        )
        for theme in result.themes:
            inserts.append((r["game_id"], r["ply"], theme,
                            result.material_swing, result.refutation_san))
            counts[theme] = counts.get(theme, 0) + 1

    store.conn.executemany(
        """INSERT OR REPLACE INTO move_themes
           (game_id, ply, theme, material_swing, refutation_san)
           VALUES (?,?,?,?,?)""",
        inserts,
    )
    store.commit()
    return {
        "mistakes_labelled": len(rows),
        "theme_labels_written": len(inserts),
        "missing_pv": missing_pv,
        "theme_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }
