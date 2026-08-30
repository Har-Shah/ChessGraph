# ChessGraph evaluation

Full results for [ChessGraph](../README.md). Everything here was measured
on 497 real Lichess games. Reproduce it all with:

```bash
./.venv/bin/python scripts/evaluate_all.py <lichess-username>
```

## 1. Method

### 1.1 Measuring a mistake

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

### 1.2 Tactical themes are detected mechanically

Every theme label comes from a geometric or material condition checked with
`python-chess`, not from a language model. Forks, pins, skewers, discovered
attacks, back rank weakness and hanging pieces are each a specific board
predicate. The labels are therefore reproducible, countable and falsifiable.
You can open any cited position and check whether the label is right.

The pass runs in one second over 3,790 mistakes because it reuses principal
variations already sitting in the engine cache, joined through
`moves.pos_key_after = position_eval.pos_key`.

### 1.3 Ground truth without human labelling

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

## 2. Retrieval results in full

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
family is 0.090. This is the most interesting failure in the project, and the
README explains the mechanism behind it.

**The graph loses lexical queries by an order of magnitude.** This is correct
behaviour. When a query names exactly what it wants, entity linking adds a
failure mode and nothing else. The graph retriever returns nothing when entity
linking finds no match, rather than falling back to text search, so the
evaluation sees the real failure rate.

## 3. Position similarity, the computed edge

Every other edge in the graph looks up something already recorded. `resembles`
is computed, and it is the only capability text retrieval cannot imitate. Two
positions can share no opening name, no player and no vocabulary and still be
the same position to play, because pawn structure decides how a position is
handled.

Similarity is `pawn_similarity * (0.6 + 0.4 * rest)`, where `rest` blends
material, king placement and minor piece configuration. Pawn structure **gates**
the score rather than contributing a weighted share. An additive version was
tried first and calibrated badly: two positions from unrelated openings scored
0.604, because material and king terms stay high for almost any pair of early
middlegames and put a floor under everything. Multiplying removes the floor,
which is the correct chess judgement. Different skeleton, different game.

**Does it work?** Compared against two baselines. The honest one is random pairs
drawn from the same blocking bucket, since blocking alone already forces phase
and material to agree.

| pair set | same opening family | same ECO | shared theme |
|---|---|---|---|
| `resembles` edges | **89.4%** | **82.0%** | 46.8% |
| random, same block | 32.7% | 25.9% | 39.1% |
| random, anywhere | 18.5% | 6.5% | 25.5% |

**The 89.4% is almost too good, and that is the point to interrogate.** If
similarity only recovered "same opening", it would be redundant with metadata
already in the graph. The value is the 10.6% of edges that cross opening
families, which no metadata filter can produce. Of those 6,922 cross-opening
edges, 71.8% link two opening-phase positions and only 9.0% are endgames where
positions converge trivially. 5,715 have both positions inside the first 30
plies, which are true transpositions. The highest scoring examples are
chess-correct: Italian Game to King's Knight Opening is a literal transposition,
and Sicilian to Caro-Kann is a genuine structural relative.

**Cost.** 25,019 positions is 313 million naive comparisons. Blocking on
(phase, side to move, coarse material) removes 92.9% of them. The remainder is
pruned exactly rather than heuristically: since `rest <= 1`, the score can never
exceed pawn similarity, so a pair whose pawn similarity is below the threshold
is a *proof* of failure, not a guess. Pawn similarity is computed per block with
bitboard popcounts, and the full score runs only on survivors. Total: 65,187
edges in 14.8 seconds, with no block-size cap and therefore no lost recall. The
invariant that makes the prune sound is asserted directly in
`tests/test_similarity.py`.

## 4. Held-out evaluation

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

## 5. Grounding and recommendation quality

### 5.1 Grounding

Every claim in the report is a structured object with citations attached, so
grounding is checkable rather than spot-checked.

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

### 5.2 Recommendation quality

The system analyses at depth 12 for throughput. Re-searching 40 report
positions at depth 20:

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

### 5.3 Similarity chains do not improve text retrieval

`resembles` edges were also wired into the graph retriever as an optional extra
hop, scored with decay, and evaluated as a separate variant.

| family | graph nDCG@10 | graph + similarity nDCG@10 |
|---|---|---|
| thematic | 0.704 | 0.704 |
| relational | 0.418 | 0.418 |
| paraphrase | 0.080 | 0.080 |
| lexical | 0.087 | 0.051 |

No effect on three families, slightly worse on the fourth. The result is not
surprising once stated plainly: none of the query families asks a structural
question. They ask about openings, themes and opponents, all of which the base
traversal already answers, so expansion only adds positions that are related to
the answer rather than part of it.

The conclusion is about scope, not about the feature. Position similarity earns
its place as a direct tool, "show me positions like this one", which is the
`retrieve_similar_positions` agent tool and the MVP requirement it satisfies. It
is not a retrieval booster for natural language questions, and the expansion is
left off by default.

Testing it properly would need a fifth query family whose ground truth is
structural, for example "positions where I face this same pawn structure". That
is the obvious next evaluation to add.

## 6. Design decisions worth explaining

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

## 7. Tests

```bash
./.venv/bin/python -m pytest tests/ -q            # 59 tests
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
| Position similarity | the pruning invariant that score never exceeds pawn similarity, pawn-structure gating, castling-side sensitivity, symmetry, no self or duplicate edges |

Two of these were written to lock in bugs the tests themselves found. The speed
classifier's docstring claimed 1+2 was blitz when the formula makes it bullet,
and an early crash-recovery test asserted Stockfish had a bug on a legal
position when the position was in fact illegal (White to move with Black in
check). Both are now correct and guarded.

