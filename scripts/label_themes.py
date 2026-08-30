#!/usr/bin/env python
"""Label every mistake with tactical themes. Run after ingest.

    python scripts/label_themes.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from chessgraph.engine.theme_pass import run_theme_pass
from chessgraph.store.db import Store

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(min_cp_loss: int = typer.Option(100), depth: int = typer.Option(12)):
    with Store() as store:
        stats = run_theme_pass(store, min_cp_loss=min_cp_loss, depth=depth)
    for k, v in stats.items():
        console.print(f"  [dim]{k:22s}[/] {v}")


if __name__ == "__main__":
    app()
