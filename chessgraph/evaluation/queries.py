"""Evaluation query set with programmatic ground truth.

GROUND TRUTH WITHOUT HUMAN LABELLING
Most RAG evaluations need annotators, or use an LLM judge that is expensive and
not reproducible. This domain avoids both. Relevance here is a predicate over
structured fields that the database already stores. "Mistakes I make in the
Caro-Kann" has an exact answer set: every document where player is the subject
and opening family is Caro-Kann. No judgement required, and a rerun produces
identical labels.

THE BIAS THIS INTRODUCES, STATED PLAINLY
Defining relevance by structured predicates risks favouring the graph
retriever, because the graph traverses those same relations. Ignoring that
would make the whole comparison worthless. Four mitigations:

1. Queries never name the fields the predicate uses when the realistic form of
   the question would not. The relational queries name an opponent and an
   intent, never the openings, so a retriever has to find those.
2. The paraphrase family deliberately avoids theme vocabulary. A query about
   forks says "attacks two pieces at once" and never the word fork, so lexical
   match on the label cannot succeed.
3. The lexical family is included specifically because BM25 should win it. A
   comparison where one method wins every category is a broken comparison.
4. Results are reported per family, never as a single aggregate. The
   interesting claim is where each method wins, not which has the higher mean.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from chessgraph.retrieval.corpus import Document


@dataclass
class EvalQuery:
    qid: str
    text: str
    family: str            # lexical | paraphrase | relational | thematic
    relevant: set[str]
    description: str
    meta: dict = field(default_factory=dict)


# Theme paraphrases that avoid the theme label entirely, so lexical retrieval
# cannot match on the word itself and has to rely on the surrounding wording.
THEME_PARAPHRASE = {
    "fork": "positions where one enemy piece ends up attacking two of mine at once",
    "hanging_piece": "moves that leave a piece undefended and free to be taken",
    "hangs_material": "moves where I simply give away material for nothing",
    "back_rank_mate": "getting checkmated on the last row behind my own pawns",
    "back_rank_weakness": "trouble on the last row with my king boxed in",
    "absolute_pin": "a piece of mine stuck in front of my king and unable to move",
    "pin": "a piece of mine frozen because something more valuable sits behind it",
    "skewer": "my valuable piece forced to move so the one behind it is captured",
    "discovered_attack": "an enemy piece stepping aside to unleash an attack from behind",
    "accepts_unsound_sacrifice": "grabbing offered material and ending up worse for it",
    "endgame_technique": "errors I make once most pieces have come off the board",
    "king_safety": "situations where my king gets exposed to attack",
    "positional_error": "quiet moves that slowly make my position worse",
}


def _family_of(doc: Document) -> str | None:
    return doc.meta.get("family")


def build_query_set(docs: list[Document], subject: str, *,
                    min_relevant: int = 5, max_per_family: int = 8,
                    seed: int = 17) -> list[EvalQuery]:
    """Instantiate query templates against entities that actually have data."""
    rng = random.Random(seed)
    subject_lc = subject.lower()
    queries: list[EvalQuery] = []

    subject_docs = [d for d in docs if (d.meta.get("player") or "").lower() == subject_lc]

    # ---------------------------------------------------------------- indexes
    by_family: dict[str, list[Document]] = defaultdict(list)
    by_theme: dict[str, list[Document]] = defaultdict(list)
    by_opponent: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        if fam := _family_of(d):
            by_family[fam].append(d)
        for t in d.meta.get("themes", []):
            by_theme[t].append(d)
    for d in docs:
        player = (d.meta.get("player") or "").lower()
        if player and player != subject_lc:
            by_opponent[player].append(d)

    # --------------------------------------------------- A. lexical lookup
    # Names the opening explicitly. BM25 is expected to do well here.
    fams = [f for f, ds in by_family.items() if len(ds) >= min_relevant]
    fams.sort(key=lambda f: -len(by_family[f]))
    for i, fam in enumerate(fams[:max_per_family]):
        rel = {d.doc_id for d in by_family[fam]}
        queries.append(EvalQuery(
            qid=f"lex{i}", family="lexical",
            text=f"blunders and mistakes in the {fam}",
            relevant=rel,
            description=f"Names the opening family literally. {len(rel)} relevant.",
            meta={"opening_family": fam},
        ))

    # ------------------------------------------------ B. paraphrased theme
    # Never uses the theme label. Tests semantic rather than lexical matching.
    themes = [t for t, ds in by_theme.items()
              if len(ds) >= min_relevant and t in THEME_PARAPHRASE]
    themes.sort(key=lambda t: -len(by_theme[t]))
    for i, theme in enumerate(themes[:max_per_family]):
        rel = {d.doc_id for d in by_theme[theme]}
        queries.append(EvalQuery(
            qid=f"para{i}", family="paraphrase",
            text=THEME_PARAPHRASE[theme],
            relevant=rel,
            description=f"Describes '{theme}' without naming it. {len(rel)} relevant.",
            meta={"theme": theme},
        ))

    # ------------------------------------------------------ C. relational
    # The motivating question. Names an opponent and an intent, never the
    # openings or the positions, which are what the answer consists of.
    opponents = [(o, ds) for o, ds in by_opponent.items() if len(ds) >= min_relevant]
    opponents.sort(key=lambda kv: -len(kv[1]))
    for i, (opp, opp_docs) in enumerate(opponents[:max_per_family]):
        # Relevance is a genuine two-way join: this opponent's mistakes, AND
        # only in the opening families they actually play often.
        fam_counts = Counter(_family_of(d) for d in opp_docs if _family_of(d))
        top_fams = {f for f, _ in fam_counts.most_common(3)}
        rel = {d.doc_id for d in opp_docs if _family_of(d) in top_fams}
        if len(rel) < min_relevant:
            continue
        name = opp_docs[0].meta.get("player")
        queries.append(EvalQuery(
            qid=f"rel{i}", family="relational",
            text=(f"Which opening variation should I prepare against {name}, "
                  f"and what recurring mistakes do they make in those positions?"),
            relevant=rel,
            description=(f"Two-hop join: {name}'s mistakes restricted to their "
                         f"most-played families {sorted(top_fams)}. {len(rel)} relevant."),
            meta={"opponent": name, "families": sorted(top_fams)},
        ))

    # Same shape aimed at the subject, which always has data even when no
    # opponent recurs often enough.
    subj_fam_counts = Counter(_family_of(d) for d in subject_docs if _family_of(d))
    top_subj_fams = {f for f, _ in subj_fam_counts.most_common(5)}
    for color in ("white", "black"):
        rel = {d.doc_id for d in subject_docs
               if d.meta.get("color") == color and _family_of(d) in top_subj_fams}
        if len(rel) >= min_relevant:
            queries.append(EvalQuery(
                qid=f"relself_{color}", family="relational",
                text=(f"In the openings I play most often as {color}, "
                      f"what mistakes do I keep repeating?"),
                relevant=rel,
                description=(f"Join of my top-5 families with my {color} mistakes. "
                             f"{len(rel)} relevant."),
                meta={"color": color, "families": sorted(top_subj_fams)},
            ))

    # -------------------------------------------------------- D. thematic
    # Names the theme and a colour. Both lexical and graph have a route in.
    for i, theme in enumerate(themes[:max_per_family]):
        for color in ("white", "black"):
            rel = {d.doc_id for d in by_theme[theme]
                   if d.meta.get("color") == color
                   and (d.meta.get("player") or "").lower() == subject_lc}
            if len(rel) < min_relevant:
                continue
            label = theme.replace("_", " ")
            queries.append(EvalQuery(
                qid=f"them{i}_{color}", family="thematic",
                text=f"my recurring {label} problems as {color}",
                relevant=rel,
                description=f"Subject's {theme} mistakes as {color}. {len(rel)} relevant.",
                meta={"theme": theme, "color": color},
            ))

    rng.shuffle(queries)
    return queries


def query_set_stats(queries: list[EvalQuery]) -> dict:
    by_fam = Counter(q.family for q in queries)
    return {
        "total": len(queries),
        "by_family": dict(by_fam),
        "avg_relevant": round(
            sum(len(q.relevant) for q in queries) / max(len(queries), 1), 1),
    }
