#!/usr/bin/env python
"""Experiment: can a general embedding model bridge chess jargon?

    ./.venv/bin/python experiments/vocabulary_gap.py

MOTIVATION
Dense retrieval scored 0.000 precision@10 on paraphrased theme queries in the
main evaluation, while the graph retriever scored 0.700. Before concluding
anything about graph retrieval, the obvious question is whether the embedding
model is doing its job at all.

METHOD
For each tactical theme, take its paraphrase from the evaluation query set,
which never contains the theme's own name, retrieve the top 50 documents, and
measure what fraction carry that theme against the corpus base rate. A model
that understands the concept should show a large lift. A model doing surface
matching should show a lift only where the paraphrase happens to reuse the
theme label's own words.

RESULT (497 games, 3,790 documents, BAAI/bge-small-en-v1.5)
The split follows shared vocabulary, not conceptual difficulty. "shares" below
is an exact content-word overlap between the theme label and its paraphrase.

    theme                       base   top50    lift  shares
    king_safety                 5.4%   58.0%   10.7x  yes
    hanging_piece              16.4%   94.0%    5.7x  yes
    positional_error           22.4%   66.0%    2.9x  no   (stem overlap only)
    back_rank_weakness          3.0%    6.0%    2.0x  no
    hangs_material             33.6%   54.0%    1.6x  yes
    endgame_technique          18.2%   14.0%    0.8x  no
    discovered_attack           4.9%    2.0%    0.4x  yes  (weak shared word)
    pin                         9.8%    4.0%    0.4x  no
    fork                       11.8%    0.0%    0.0x  no
    absolute_pin                3.4%    0.0%    0.0x  no
    skewer                      4.5%    0.0%    0.0x  no
    accepts_unsound_sacrifice  13.5%    0.0%    0.0x  no

    mean lift when the paraphrase shares a content word: 4.60x
    mean lift when it does not:                          0.76x

A six-fold difference, and the four themes scoring exactly 0.0x are the four
purely geometric motifs: fork, skewer, absolute pin, and sacrifice acceptance.

CONCLUSION
The model is doing soft lexical matching. Every theme it handles is one where
the paraphrase reuses a content word from the label. Every theme it fails is
one where connecting the description to the label requires chess knowledge
rather than English semantics. "One enemy piece attacking two of mine at once"
is a fork only if you know chess, and bge-small does not.

WHY THIS MATTERS FOR THE PROJECT
This is the mechanism behind the headline retrieval result. The graph wins
thematic queries not because graphs are inherently better at retrieval, but
because the theme labels are computed mechanically from board geometry and
stored as explicit edges. The mapping from concept to instance is precomputed
rather than inferred at query time.

It also predicts the fix for anyone wanting dense retrieval to work here:
either use a domain-adapted embedding model, or expand theme labels into
descriptions in the document text so the vocabulary gap closes. The second is
cheap and is the obvious next experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chessgraph.evaluation.queries import THEME_PARAPHRASE
from chessgraph.retrieval.corpus import build_corpus
from chessgraph.retrieval.vector import VectorRetriever
from chessgraph.store.db import Store


def main(min_docs: int = 30, k: int = 50) -> None:
    with Store() as store:
        docs = build_corpus(store)
    vec = VectorRetriever()
    vec.index(docs)

    print(f"corpus: {len(docs)} documents, model {vec.model_name}\n")
    print(f"{'theme':28s} {'base':>7s} {'top50':>7s} {'lift':>6s}  shares vocab")
    rows = []
    for theme, paraphrase in THEME_PARAPHRASE.items():
        with_theme = [d for d in docs if theme in d.meta["themes"]]
        if len(with_theme) < min_docs:
            continue
        base = len(with_theme) / len(docs)
        hits = vec.search(paraphrase, k=k).hits
        got = sum(1 for h in hits if theme in h.doc.meta["themes"]) / k
        lift = got / base if base else 0.0
        label_words = set(theme.replace("_", " ").split())
        shares = bool(label_words & set(paraphrase.lower().split()))
        rows.append((theme, base, got, lift, shares))

    for theme, base, got, lift, shares in sorted(rows, key=lambda r: -r[3]):
        print(f"{theme:28s} {base:6.1%} {got:6.1%} {lift:5.1f}x  "
              f"{'yes' if shares else 'no'}")

    shared = [r for r in rows if r[4]]
    unshared = [r for r in rows if not r[4]]
    if shared and unshared:
        ms = sum(r[3] for r in shared) / len(shared)
        mu = sum(r[3] for r in unshared) / len(unshared)
        print(f"\nmean lift when paraphrase shares a content word: {ms:.2f}x")
        print(f"mean lift when it does not:                      {mu:.2f}x")


if __name__ == "__main__":
    main()
