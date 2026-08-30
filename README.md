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
| Retrieval, relational queries | BM25 nDCG@10 **0.730** vs graph 0.418 | not the expected result, see section 5 |
| Weakness persistence | Spearman **0.988**, top-5 overlap **1.0** | weaknesses recur in held-out games |
| Opening prediction (family) | top-1 **0.419** vs 0.379 baseline | lift of only +0.040 |
| Opening ACPL persistence | Spearman **0.182** | which openings you play worse does not transfer |
| Recommendation agreement | **85%** at depth 20 vs depth 12 | mean cost 4.7cp when they differ |
| Grounding | **100%** resolution, support and numeric fidelity | 0 issues across 60 citations |

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
| Graph | 4,856 nodes, 14,700 edges |

## 4. Method

### 4.1 Measuring a mistake

Engine evaluations are always from the side-to-move's perspective, so they flip
sign every ply. Centipawn loss is computed as `e[i] + e[i+1]` over consecutive
evaluations rather than by re-searching after each move. That needs N+1 searches
instead of 2N, and the perspective flip becomes an addition instead of a sign
bug.

Validated two ways:

1. On the Fried Liver. Black is -17 before `5...Nxd5`, White is +79 after, so
   the move costs 62cp, and the engine independently finds `6.Nxf7` as the
   refutation.
2. By calibration. Magnus Carlsen scores ACPL 20.5 over 36 games, inside the
   expected grandmaster band of under 25.

### 4.2 Tactical themes are detected mechanically

Every theme label comes from a geometric or material condition checked with
`python-chess`, not from a language model. Forks, pins, skewers, discovered
attacks, back rank weakness and hanging pieces are each a specific board
predicate. The labels are therefore reproducible, countable and falsifiable.
You can open any cited position and check whether the label is right.

The pass runs in one second over 3,790 mistakes because it reuses principal
variations already sitting in the engine cache, joined through
`moves.pos_key_after = position_eval.pos_key`.

### 4.3 Ground truth without human labelling

Relevance is a predicate over structured fields the database already stores.
"My mistakes in the Caro-Kann" has an exact answer set. No annotators, no LLM
judge, and a rerun reproduces the labels exactly.

That introduces a bias toward graph traversal, since the graph walks those same
relations. Four mitigations, all implemented:

1. Queries do not name the fields the predicate uses when a realistic question
   would not. Relational queries name an opponent and an intent, never the
   openings.
2. Paraphrase queries never use theme vocabulary. A query about forks says
   "attacks two pieces at once" and never the word fork.
3. A lexical family is included specifically because BM25 should win it. A
   comparison where one method wins everything is a broken comparison.
4. Results are reported per family, never as a single aggregate.

Relevant sets are bounded to between 5 and 60 documents. An earlier version
averaged 284 relevant documents out of 3,790, which capped Recall@10 at 0.035
and made the metric unable to move.

## 5. Retrieval results in full

44 queries, median 29 relevant documents each. Best value per column in bold.

**Lexical** (names the opening literally, 10 queries)

| retriever | recall@10 | recall@20 | precision@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.179 | **0.375** | 0.820 | 0.788 |
| vector | 0.186 | 0.342 | 0.850 | 0.864 |
| graph | 0.013 | 0.052 | 0.060 | 0.087 |
| hybrid (all) | **0.194** | 0.368 | **0.890** | **0.893** |

**Thematic** (names the tactical theme, 10 queries)

| retriever | recall@10 | recall@20 | precision@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.032 | 0.072 | 0.090 | 0.099 |
| vector | 0.052 | 0.084 | 0.150 | 0.141 |
| graph | **0.199** | **0.465** | **0.700** | **0.704** |
| hybrid (vector+graph) | 0.132 | 0.254 | 0.430 | 0.451 |

**Relational** (names an opponent and an intent, 14 queries)

| retriever | recall@10 | recall@20 | precision@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 | **0.681** | **0.939** | **0.686** | **0.730** |
| vector | 0.024 | 0.051 | 0.014 | 0.031 |
| graph | 0.366 | 0.582 | 0.386 | 0.418 |
| hybrid (all) | 0.425 | 0.676 | 0.443 | 0.472 |

**Paraphrase** (describes the theme without naming it, 10 queries)

| retriever | recall@10 | recall@20 | precision@10 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.000 | 0.003 | 0.000 | 0.000 |
| vector | 0.015 | 0.022 | 0.040 | 0.031 |
| graph | 0.023 | **0.065** | 0.080 | 0.080 |
| hybrid (vector+graph) | **0.034** | 0.041 | **0.090** | **0.095** |

Three things worth stating plainly about this table.

**BM25 wins relational queries, which is not what I expected.** The reason is
that the query names the opponent's username, and usernames are rare, highly
discriminative tokens that appear verbatim in the documents. Lexical match
solves most of the problem before any reasoning is needed. The graph's
advantage is confined to filtering that opponent's games down to the opening
family they actually play often, and that filtering is worth less than the
username match is worth.

**Every retriever fails on paraphrase queries.** The best precision@10 in that
family is 0.090. This is the most interesting failure in the project and
section 6 explains it.

**The graph loses lexical queries by an order of magnitude.** This is correct
behaviour. When a query names exactly what it wants, entity linking adds a
failure mode and nothing else. The graph retriever returns nothing when entity
linking finds no match, rather than falling back to text search, so the
evaluation sees the real failure rate.

## 6. Why dense retrieval fails here

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

## 7. Held-out evaluation

The split is temporal, never random. Train on older games, test on newer. A
random split leaks the same opponents and the same repertoire phase into both
sides and inflates every metric. Split here is 373 train, 124 test.

**Weakness persistence.** Do the weaknesses found in training recur later?

| | |
|---|---|
| Spearman correlation of theme rates | 0.988 |
| Top-5 theme overlap | 1.0 |
| Train top 5 | hangs_material, endgame_technique, fork, positional_error, hanging_piece |
| Test top 5 | hangs_material, endgame_technique, positional_error, hanging_piece, fork |

The same five themes, in near-identical proportions, across a five year gap.
This is the result that makes a training plan worth building at all. If the
profile did not persist, every recommendation would be describing noise.

**Opening prediction.** Predicting the held-out opening from training
frequencies, conditioned on colour, against a baseline that ignores colour.

| level | top-1 | top-3 | baseline | lift |
|---|---|---|---|---|
| family | 0.419 | 0.589 | 0.379 | +0.040 |
| opening | 0.145 | 0.234 | 0.089 | +0.057 |
| ECO | 0.145 | 0.331 | 0.097 | +0.048 |

Modest. Conditioning on colour buys 4 to 6 points over just predicting the most
frequent opening. Honest reading: the repertoire is predictable mostly because
it is narrow, not because the model is clever.

**Opening ACPL persistence.** Does per-opening centipawn loss measured in
training predict test ACPL? Spearman 0.182 across 14 openings. Effectively no.
Aggregate weaknesses transfer, but "which of your openings you play worse" does
not. That is a real negative result and it constrains what the report should
claim.

## 8. Grounding and recommendation quality

**Grounding.** Every claim in the report is a structured object with citations
attached, so grounding is checkable rather than spot-checked.

| check | result |
|---|---|
| Claims with citations | 20 / 20 |
| Citations that resolve to a real move | 60 / 60 |
| Citations that support their claim | 60 / 60 |
| Numbers that match a recomputation | 100% |
| Issues | 0 |

The MVP report is template generated, so resolution and numeric fidelity are
close to tautological today. They exist now so that when a language model
writes the prose there is an established 100% baseline to measure the drop
against. The support check is not tautological even today, since citation
selection and claim text come from separate queries and can disagree.

**Recommendation quality.** The system analyses at depth 12 for throughput.
Re-searching 40 report positions at depth 20:

| | |
|---|---|
| Agreement at 1 | 0.850 |
| Agreement at 2 | 0.925 |
| Mean cost of the recommendation | 4.7cp |
| Median cost | 0cp |
| Recommendations worse than 100cp | 0 |

Depth 12 disagrees with depth 20 on 15% of positions, but the disagreements are
between moves that are nearly equally good. Mean cost of following the depth 12
recommendation is 4.7 centipawns, which is not detectable in a real game. Depth
12 is an adequate working depth and the 17 minute analysis budget is justified.

## 9. Honest notes

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

## 10. Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
brew install stockfish
```

Set `STOCKFISH_PATH` if the binary is not on `PATH`. The agent in section 11
needs `ANTHROPIC_API_KEY`. Nothing else does.

## 11. Usage

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

Reproduce the vocabulary gap experiment from section 6.

```bash
./.venv/bin/python experiments/vocabulary_gap.py
```

Ask the agent a question. Requires an API key.

```bash
./.venv/bin/python scripts/ask.py <lichess-username> "which opening should I fix first?"
```

## 12. Layout

```
chessgraph/
  config.py           paths, engine settings, mistake thresholds
  models.py           GameRecord / MoveRecord / PositionRecord
  pipeline.py         download -> parse -> analyse -> store
  ingest/
    lichess.py        streaming API client, rate limit handling
    parse.py          PGN to records, speed and phase classification
  engine/
    analyzer.py       Stockfish wrapper, centipawn loss, ACPL
    cache.py          persistent evaluation cache
    themes.py         mechanical tactical theme detection
    theme_pass.py     labels every mistake, reusing cached PVs
  store/
    db.py             SQLite schema, source of truth
    graph.py          NetworkX knowledge graph, derived view
  retrieval/
    corpus.py         the shared document set all retrievers index
    keyword.py        BM25
    vector.py         bge-small via fastembed, cached embeddings
    graph_retriever.py entity linking plus multi-hop traversal
    hybrid.py         reciprocal rank fusion
  evaluation/
    queries.py        query set with programmatic ground truth
    metrics.py        recall, precision, MRR, nDCG
    harness.py        runs all retrievers, reports per family
    holdout.py        temporal split, opening prediction, persistence
    grounding.py      citation resolution, support, numeric fidelity
    recommendation.py depth 20 verification of depth 12 advice
  report/generate.py  structured claims with citations, then Markdown
  agent/
    tools.py          seven bounded tools, usable without a model
    runner.py         Anthropic tool runner wiring
experiments/
  vocabulary_gap.py   why dense retrieval fails on chess jargon
scripts/              ingest, label_themes, report, evaluate_all, ask
tests/                pytest suite
```

## 13. Design decisions worth explaining

**SQLite is the source of truth and the graph is derived.** The graph is
rebuilt from SQLite, so the two cannot drift apart. Most single-hop questions
are a GROUP BY and SQL wins them. The graph earns its place on multi-hop joins
whose depth is not known in advance, on hierarchical rollup from variation to
family when a specific variation has too few games, and on similarity chains.

**Positions are deduplicated by FEN prefix.** Dropping the halfmove and
fullmove counters means the same structure reached by different move orders
collapses to one node. That normalisation is what makes "every time you reach
this position, you play X" a lookup instead of a scan.

**The retrievable unit is a mistake instance, not a game.** It is the smallest
thing that is citable, countable and checkable. Aggregates are computed from
instances and never stored as free text, so no claim exists without instances
behind it.

**Report citations are filtered to competitive positions.** Ranking examples by
centipawn loss alone surfaces moves played while already three pieces down,
which have the largest losses and the least to teach. Cited examples are
restricted to positions within 3 pawns of level. Aggregate counts use all
mistakes and are unaffected.

**Every pipeline stage is idempotent and restartable.** Downloads cache to
disk, engine evaluations hit a persistent cache, inserts key on natural IDs. A
17 minute analysis run can be interrupted and resumed without losing work.

## 14. Tests

```bash
./.venv/bin/python -m pytest tests/ -q            # 47 tests
./.venv/bin/python -m pytest tests/ -q -m "not slow"   # skip engine tests
```

| area | what is checked |
|---|---|
| Centipawn loss | sign convention against a constructed queen hang, book-move exclusion, colour and subject filtering |
| Theme detectors | fork, back rank mate, absolute pin, and a regression guard that Bb5 in the Ruy Lopez is a skewer and not a pin |
| Sacrifice detection | the Fried Liver returns negative material swing and is still labelled a blunder |
| Ranking metrics | recall, precision, MRR and nDCG against hand-computed values, including nDCG normalisation when relevant documents outnumber slots |
| BM25 | ranking order, empty results, and that IDF never goes negative |
| Rank fusion | that RRF depends on rank only and ignores score magnitude |
| Parsing | FEN prefix keying, Lichess speed buckets including the increment rule, material signatures |
| Engine robustness | illegal positions are rejected before reaching Stockfish, and a killed engine process is restarted rather than aborting the run |
| Temporal split | Spearman against ties, constant series and non-linear monotone relations |

Two of these were written to lock in bugs the tests themselves found. The speed
classifier's docstring claimed 1+2 was blitz when the formula makes it bullet,
and an early crash-recovery test asserted Stockfish had a bug on a legal
position when the position was in fact illegal (White to move with Black in
check). Both are now correct and guarded.

## 15. Next

- Multi-player ingestion, so weakness persistence can be compared against a
  population baseline.
- Expand theme labels into descriptions in the document text and re-run the
  vocabulary gap experiment. This is the cheapest available retrieval win.
- Position similarity edges, which are designed in the schema but not yet
  populated. That is the one graph capability text search fundamentally cannot
  replicate.
- Chess.com ingestion behind the same parser.
