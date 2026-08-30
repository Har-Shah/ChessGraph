# ChessGraph

Analyses a player's game history with Stockfish, builds a knowledge graph of
their openings, positions and recurring mistakes, and generates a cited
preparation report. The core of the project is a controlled comparison of four
retrieval strategies over the same corpus.

Everything below was measured on 497 real Lichess games. No numbers are
estimated.

## 1. The question

The motivating query is:

> Which opening variation should I prepare against this opponent, and what
> recurring mistakes do they make in those positions?

Answering it requires joining across entities:

```
opponent --plays--> opening --leads_to--> position <--blunders_at-- opponent
```

Vector search retrieves text similar to a query. It cannot compute the
intersection of "openings this opponent plays" and "positions where they go
wrong", because neither set appears in the question. That structural
requirement is the reason to use a graph here rather than adding GraphRAG to a
project that did not need it.

Whether the graph actually wins is an empirical question. The answer turned out
to be "on two of four question types, and it loses badly on the other two".

## 2. Headline results

| Metric | Result | Notes |
|---|---|---|
| Retrieval, thematic queries | graph nDCG@10 **0.704** vs BM25 0.099 | 7x better |
| Retrieval, lexical queries | BM25 nDCG@10 **0.788** vs graph 0.087 | graph loses badly |
| Retrieval, relational queries | BM25 nDCG@10 **0.730** vs graph 0.418 | not the expected result, see the evaluation doc |
| Weakness persistence | Spearman **0.988**, top-5 overlap **1.0** | weaknesses recur in held-out games |
| Opening prediction (family) | top-1 **0.419** vs 0.379 baseline | lift of only +0.040 |
| Opening ACPL persistence | Spearman **0.182** | which openings you play worse does not transfer |
| Recommendation agreement | **85%** at depth 20 vs depth 12 | mean cost 4.7cp when they differ |
| Position similarity | **89.4%** of linked pairs share an opening family | vs 32.7% for matched random pairs |
| Similarity as a retrieval booster | **no effect** | honest negative, see the evaluation doc |
| Grounding | **100%** resolution, support and numeric fidelity | 0 issues across 60 citations |

Full tables, method and reproduction steps: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

## 3. Corpus

| | |
|---|---|
| Games | 497 (Lichess, blitz/rapid/classical, 2020-12 to 2026-04) |
| Moves parsed | 28,809 |
| Positions analysed | 25,019 unique |
| Engine | Stockfish 18, depth 12, first 60 plies |
| Analysis time | 17 minutes, cached thereafter |
| Mistakes found (>=100cp) | 3,790 |
| Player ACPL | 66.7 |
| Graph | 4,856 nodes, 29,944 edges (8 relation types) |

## 4. Why dense retrieval fails here

Vector retrieval scored 0.031 nDCG@10 on paraphrased theme queries. Before
concluding anything about graphs, the obvious question is whether the embedding
model works at all.

`experiments/vocabulary_gap.py` measures this. For each theme, it takes the
paraphrase, retrieves the top 50 documents, and compares the rate of that theme
against its corpus base rate.

| theme | lift | paraphrase shares a content word |
|---|---|---|
| king_safety | 10.7x | yes ("king") |
| hanging_piece | 5.7x | yes ("piece") |
| positional_error | 2.9x | stem only ("position") |
| back_rank_weakness | 2.0x | partial |
| hangs_material | 1.6x | yes ("material") |
| endgame_technique | 0.8x | no |
| pin | 0.4x | no |
| discovered_attack | 0.4x | weak |
| fork | 0.0x | no |
| absolute_pin | 0.0x | no |
| skewer | 0.0x | no |
| accepts_unsound_sacrifice | 0.0x | no |

Mean lift when the paraphrase shares a content word with the label: **4.60x**.
When it does not: **0.76x**. A six-fold difference.

The four themes scoring exactly zero are the four purely geometric motifs. The
model is doing soft lexical matching, not conceptual mapping. It cannot connect
"one enemy piece attacking two of mine at once" to "fork", because that
requires chess knowledge rather than English semantics.

This is the mechanism behind the headline result. The graph wins thematic
queries not because graphs retrieve better in general, but because the theme
labels are computed from board geometry and stored as explicit edges. The
mapping from concept to instance is precomputed instead of inferred at query
time.

It also predicts the fix. Either use a domain-adapted embedding model, or
expand theme labels into descriptions inside the document text so the
vocabulary gap closes. The second is cheap and is the obvious next experiment.

## 5. Honest notes

- **One player.** Every number here comes from a single 497 game corpus. The
  weakness persistence result in particular needs a population baseline to
  separate "this player's recurring weaknesses" from "what everyone at 1300
  does", and one player cannot provide it. Multi-player ingestion is the first
  thing to add.
- **The temporal split is uneven.** 373 training games span roughly six months
  and 124 test games span five years, because play volume dropped sharply after
  mid-2021. The persistence result survives that gap, which is arguably a
  stronger claim, but the windows are not comparable in duration.
- **BM25 beats the graph on the query the graph was built for.** Section 5
  explains why. The result stands as measured, and the graph's genuine win is
  on thematic and aggregate questions instead.
- **Bullet games are excluded by default.** Blunders under 30 seconds measure
  time pressure, not understanding. Included via `--perf`.
- **The engine cache hit rate was only 16%.** Lower than expected, because a
  1300 rated player's games diverge from theory early. A stronger player with a
  narrow repertoire would see far higher reuse.
- **The Lichess `max` parameter is advisory.** A request for 25 games returned
  36. Capped client-side so corpus size is reproducible.
- **Similarity weights are hand-set, not learned.** They encode a chess
  judgement about what makes positions alike. A GNN over the position graph is
  the natural successor and is deliberately out of scope until the evaluation
  pipeline is settled.
- **The agent runner is untested.** `scripts/ask.py` has never been executed
  end to end, because no API key was available. The seven tools underneath it
  are tested directly; the Claude wiring is not.

## 6. Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
brew install stockfish
```

Set `STOCKFISH_PATH` if the binary is not on `PATH`. The agent below
needs `ANTHROPIC_API_KEY`. Nothing else does.

## 7. Usage

Ingest and analyse a player. Roughly 2 seconds per game on first run, cached
after that.

```bash
./.venv/bin/python scripts/ingest.py <lichess-username> --max-games 500
```

Label every mistake with tactical themes. Takes about a second.

```bash
./.venv/bin/python scripts/label_themes.py
```

Generate the cited preparation report.

```bash
./.venv/bin/python scripts/report.py <lichess-username>
```

Run every evaluation and write a JSON report.

```bash
./.venv/bin/python scripts/evaluate_all.py <lichess-username>
```

Reproduce the vocabulary gap experiment from section 4.

```bash
./.venv/bin/python experiments/vocabulary_gap.py
```

Ask the agent a question. Requires an API key.

```bash
./.venv/bin/python scripts/ask.py <lichess-username> "which opening should I fix first?"
```

## 8. Layout

```
chessgraph/
  ingest/       Lichess API client and PGN parsing
  engine/       Stockfish wrapper, evaluation cache, tactical theme detection
  store/        SQLite schema, knowledge graph, position similarity
  retrieval/    shared corpus plus BM25, vector, graph and hybrid retrievers
  evaluation/   query set, ranking metrics, held-out split, grounding
  report/       structured claims with citations, rendered to Markdown
  agent/        eight bounded tools and the Anthropic tool runner
experiments/    standalone findings, currently the vocabulary gap
scripts/        ingest, label_themes, report, evaluate_all, ask
tests/          59 tests
```

SQLite is the source of truth and the graph is rebuilt from it, so the two
cannot drift apart. Design rationale for the non-obvious choices is in
[docs/EVALUATION.md](docs/EVALUATION.md).

## 9. Next

- Multi-player ingestion, so weakness persistence can be compared against a
  population baseline.
- Expand theme labels into descriptions in the document text and re-run the
  vocabulary gap experiment. This is the cheapest available retrieval win.
- A structural query family for the evaluation, so position similarity can be
  measured on questions it is actually the right tool for.
- Chess.com ingestion behind the same parser.

