# ChessGraph

An AI system that analyses a player's historical games, identifies recurring
weaknesses and opening tendencies, retrieves relevant positions, and generates
a personalised preparation plan — with a **graph vs vector retrieval study** at
its core.

## Why a graph?

The motivating question is:

> *"Which opening variation should I prepare against this opponent, and what
> recurring mistakes do they make in those positions?"*

Answering it requires a multi-hop join across heterogeneous entities:

```
Opponent --plays--> Opening --leads_to--> Position <--blunders_at-- Opponent
```

Vector retrieval finds text *similar* to a query. It cannot compute the
intersection of "openings this opponent plays" and "positions where they go
wrong". That structural requirement is what justifies a graph here, rather than
bolting GraphRAG onto a project that did not need it. Whether the graph
actually wins is an empirical question, which is what the evaluation harness
measures.

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
brew install stockfish     # or set STOCKFISH_PATH
```

## Usage

```bash
./.venv/bin/python scripts/ingest.py <lichess-username> --max-games 500
```

## Architecture

| Module | Responsibility |
|---|---|
| `chessgraph/config.py` | Paths, engine settings, mistake thresholds |
| `chessgraph/models.py` | `GameRecord` / `MoveRecord` / `PositionRecord` |
| `chessgraph/ingest/` | Lichess API client, PGN parsing |
| `chessgraph/engine/` | Stockfish wrapper, eval cache, theme detection |
| `chessgraph/store/` | SQLite schema (source of truth) |
| `chessgraph/retrieval/` | Keyword / vector / graph / hybrid retrievers |
| `chessgraph/evaluation/` | Recall@K, ACPL, grounding accuracy |

## Design notes

**Centipawn loss is computed as `e[i] + e[i+1]`, not a subtraction.** Engine
evaluations are always from the side-to-move's perspective, so they flip sign
every ply. Evaluating each position once and adding consecutive evals gives the
loss in N+1 searches instead of 2N, and the sign convention collapses into an
addition that is much harder to get backwards. See
`chessgraph/engine/analyzer.py`.

**Positions are deduplicated by FEN prefix.** Dropping the halfmove and
fullmove counters means the same structure reached by different move orders
collapses to one node, which is what makes "every time you reach this position,
you play X" a lookup rather than a scan.

**Tactical themes are detected mechanically, not by an LLM.** Every label comes
from a geometric or material condition checked with `python-chess`, so the
labels are reproducible, countable, and falsifiable. See
`chessgraph/engine/themes.py`.
