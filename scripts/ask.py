#!/usr/bin/env python
"""Ask the preparation agent a question.

    python scripts/ask.py <subject> "which opening should I fix first?"

Requires ANTHROPIC_API_KEY. Everything else in ChessGraph runs without it.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from rich.markdown import Markdown

console = Console()
app = typer.Typer(add_completion=False)


@app.command()
def main(subject: str, question: str,
         model: str = typer.Option("claude-opus-5"),
         quiet: bool = typer.Option(False, "--quiet")):
    from chessgraph.agent.runner import ask
    answer = ask(subject, question, model=model, verbose=not quiet)
    console.print()
    console.print(Markdown(answer))


if __name__ == "__main__":
    app()
