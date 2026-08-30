#!/usr/bin/env python
"""Generate a cited preparation report.

    python scripts/report.py <subject-username> [--out report.md]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console

from chessgraph.config import DATA
from chessgraph.report.generate import build_report, render_markdown
from chessgraph.store.db import Store
from chessgraph.store.graph import ChessKnowledgeGraph

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(
    subject: str,
    out: str = typer.Option("", help="Markdown output path"),
    json_out: str = typer.Option("", help="Structured JSON output path"),
    top_openings: int = typer.Option(5),
    top_weaknesses: int = typer.Option(5),
    training: int = typer.Option(12, help="How many training positions"),
):
    with Store() as store:
        kg = ChessKnowledgeGraph.build(store, subject)
        report = build_report(store, subject, kg,
                              top_openings=top_openings,
                              top_weaknesses=top_weaknesses,
                              training_count=training)
    md = render_markdown(report)
    md_path = Path(out) if out else (DATA / f"report_{subject.lower()}.md")
    md_path.write_text(md)
    js_path = Path(json_out) if json_out else (DATA / f"report_{subject.lower()}.json")
    js_path.write_text(json.dumps(report.to_dict(), indent=2))
    console.print(md)
    console.print(f"\n[green]wrote {md_path} and {js_path}[/]")


if __name__ == "__main__":
    app()
