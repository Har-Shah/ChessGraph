"""End-to-end ingestion: download -> parse -> analyse -> store.

Kept deliberately linear and restartable. Every stage is idempotent:
  - download caches to data/raw and skips if present
  - parse is pure
  - analysis hits the eval cache, so a re-run after a crash is cheap
  - inserts are INSERT OR REPLACE keyed on natural IDs

That means you can Ctrl-C a 25-minute analysis run, restart it, and it picks up
almost where it left off. Long-running data jobs that are not restartable are
the single biggest time sink in this kind of project.
"""
from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn,
)

from chessgraph.config import ANALYSIS, INGEST
from chessgraph.engine.analyzer import Analyzer, average_cp_loss
from chessgraph.ingest.lichess import download_player_games
from chessgraph.ingest.parse import parse_pgn_file, collect_positions
from chessgraph.store.db import Store

console = Console()


def ingest_player(
    username: str,
    *,
    max_games: int = INGEST.max_games,
    store_path: Path | None = None,
    depth: int | None = None,
    max_ply: int | None = None,
    analyze: bool = True,
    force_download: bool = False,
    perf_types: tuple[str, ...] = INGEST.perf_types,
) -> dict:
    """Run the full pipeline for one player and return a summary."""
    t0 = time.time()
    console.rule(f"[bold]Ingesting {username}")

    # --- 1. download ------------------------------------------------------
    console.print("[bold cyan]1/4[/] Downloading games")
    pgn_path = download_player_games(
        username, max_games=max_games, force=force_download,
        perf_types=perf_types,
    )

    # --- 2. parse ---------------------------------------------------------
    console.print("[bold cyan]2/4[/] Parsing PGN")
    parsed = list(parse_pgn_file(pgn_path, subject=username))
    games = [g for g, _ in parsed]
    moves_by_game = [m for _, m in parsed]
    total_moves = sum(len(m) for m in moves_by_game)
    console.print(f"  {len(games)} games, {total_moves} moves")

    if not games:
        console.print("[red]  no games parsed, check the username[/]")
        return {"games": 0}

    # --- 3. analyse -------------------------------------------------------
    if analyze:
        console.print(
            f"[bold cyan]3/4[/] Stockfish analysis "
            f"(depth {depth or 'default'}, first {max_ply or ANALYSIS.max_ply} plies)"
        )
        with Analyzer() as az:
            start_cached = az.cache.size()
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                BarColumn(), TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(), TimeRemainingColumn(),
                console=console,
            ) as prog:
                task = prog.add_task("  analysing", total=len(moves_by_game))
                for i, moves in enumerate(moves_by_game):
                    az.annotate_game(moves, depth=depth, max_ply=max_ply)
                    # Commit periodically so a crash does not lose the work.
                    if i % 20 == 0:
                        az.cache.commit()
                    prog.advance(task)
            console.print(
                f"  cache: {az.cache.size()} positions "
                f"(+{az.cache.size() - start_cached} new), "
                f"hit rate {az.cache.hit_rate:.0%}"
            )
    else:
        console.print("[bold cyan]3/4[/] [dim]skipping analysis[/]")

    # --- 4. store ---------------------------------------------------------
    console.print("[bold cyan]4/4[/] Writing to SQLite")
    openings = {g.game_id: (g.eco, g.opening) for g in games}
    positions = collect_positions(moves_by_game, openings)

    with Store(store_path) as store:
        store.add_games(games)
        for moves in moves_by_game:
            store.add_moves(moves)
        store.add_positions(positions.values())
        store.commit()
        stats = store.stats()

    elapsed = time.time() - t0
    flat_moves = [m for ms in moves_by_game for m in ms]
    summary = {
        **stats,
        "username": username,
        "elapsed_s": round(elapsed, 1),
        "unique_positions": len(positions),
        "dedup_ratio": round(1 - len(positions) / max(total_moves, 1), 3),
    }
    if analyze:
        summary["acpl_subject"] = round(average_cp_loss(flat_moves, subject_only=True), 1)
        summary["blunders"] = sum(
            1 for m in flat_moves if m.is_subject_move and m.judgment == "blunder")
        summary["mistakes"] = sum(
            1 for m in flat_moves if m.is_subject_move and m.judgment == "mistake")

    console.print()
    for k, v in summary.items():
        console.print(f"  [dim]{k:18s}[/] {v}")
    console.print(f"\n[green]done in {elapsed:.1f}s[/]")
    return summary
