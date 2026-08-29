#!/usr/bin/env python
"""CLI: python scripts/ingest.py <username> [--max-games N] [--depth D]"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from chessgraph.pipeline import ingest_player

app = typer.Typer(add_completion=False)


@app.command()
def main(
    username: str,
    max_games: int = typer.Option(500, help="How many games to pull"),
    depth: int = typer.Option(12, help="Stockfish search depth"),
    max_ply: int = typer.Option(60, help="Analyse only the first N plies"),
    no_analyze: bool = typer.Option(False, "--no-analyze", help="Skip Stockfish"),
    force: bool = typer.Option(False, "--force", help="Re-download even if cached"),
    perf: str = typer.Option("blitz,rapid,classical", help="Comma-separated speeds"),
):
    ingest_player(
        username,
        max_games=max_games,
        depth=depth,
        max_ply=max_ply,
        analyze=not no_analyze,
        force_download=force,
        perf_types=tuple(p.strip() for p in perf.split(",") if p.strip()),
    )


if __name__ == "__main__":
    app()
