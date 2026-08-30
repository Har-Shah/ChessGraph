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
                    min_relevant: int = 5, max_relevant: int = 60,
                    max_per_family: int = 10,
                    seed: int = 17) -> list[EvalQuery]:
    """Instantiate query templates against entities that actually have data.

    RELEVANT SET SIZING
    Sets are capped to [min_relevant, max_relevant]. This is not cosmetic. With
    3,790 documents, a predicate like "every mistake in the Sicilian" selects
    ~300 of them, and Recall@10 is then bounded above by 10/300 = 0.033. The
    metric cannot move, so it measures nothing. Capping relevant sets near 60
    keeps Recall@20 able to reach 0.33 and makes the number informative.
    """
    rng = random.Random(seed)
    subject_lc = subject.lower()
    queries: list[EvalQuery] = []

    subject_docs = [d for d in docs if (d.meta.get("player") or "").lower() == subject_lc]

    by_opening: dict[str, list[Document]] = defaultdict(list)
    by_theme: dict[str, list[Document]] = defaultdict(list)
    by_opponent: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        if d.meta.get("opening"):
            by_opening[d.meta["opening"]].append(d)
        for t in d.meta.get("themes", []):
            by_theme[t].append(d)
        player = (d.meta.get("player") or "").lower()
        if player and player != subject_lc:
            by_opponent[player].append(d)

    def sized(rel: set[str]) -> bool:
        return min_relevant <= len(rel) <= max_relevant

    # --------------------------------------------------- A. lexical lookup
    # Specific opening variations rather than families, both to keep the
    # relevant set in range and because naming a precise variation is what a
    # lexical query realistically looks like.
    opens = [o for o, ds in by_opening.items() if sized({d.doc_id for d in ds})]
    opens.sort(key=lambda o: -len(by_opening[o]))
    for i, op in enumerate(opens[:max_per_family]):
        rel = {d.doc_id for d in by_opening[op]}
        queries.append(EvalQuery(
            qid=f"lex{i}", family="lexical",
            text=f"blunders and mistakes in the {op}",
            relevant=rel,
            description=f"Names the variation literally. {len(rel)} relevant.",
            meta={"opening": op, "n_relevant": len(rel)},
        ))

    # ----------------------------- B/D. paraphrase and thematic, same targets
    # These two families deliberately share relevance sets and differ only in
    # how the query is worded. The thematic version names the theme; the
    # paraphrase version never does. Holding the target fixed isolates lexical
    # matching from semantic matching, instead of confounding it with a change
    # in what is being asked for.
    severe_by_theme_color: dict[tuple[str, str], set[str]] = {}
    for theme, ds in by_theme.items():
        for color in ("white", "black"):
            rel = {d.doc_id for d in ds
                   if d.meta.get("is_subject_move")
                   and d.meta.get("color") == color
                   and (d.meta.get("cp_loss") or 0) >= 200}
            if sized(rel) and theme in THEME_PARAPHRASE:
                severe_by_theme_color[(theme, color)] = rel

    ordered = sorted(severe_by_theme_color.items(), key=lambda kv: -len(kv[1]))
    for i, ((theme, color), rel) in enumerate(ordered[:max_per_family]):
        label = theme.replace("_", " ")
        queries.append(EvalQuery(
            qid=f"them{i}", family="thematic",
            text=f"my serious {label} mistakes as {color}",
            relevant=rel,
            description=f"Names '{theme}' directly. {len(rel)} relevant.",
            meta={"theme": theme, "color": color, "n_relevant": len(rel)},
        ))
        queries.append(EvalQuery(
            qid=f"para{i}", family="paraphrase",
            text=f"as {color}, {THEME_PARAPHRASE[theme]}",
            relevant=rel,
            description=(f"Same target as them{i}, but never says '{theme}'. "
                         f"{len(rel)} relevant."),
            meta={"theme": theme, "color": color, "n_relevant": len(rel),
                  "paired_with": f"them{i}"},
        ))

    # ------------------------------------------------------ C. relational
    # Relevance is this opponent's mistakes restricted to their SINGLE most
    # played opening family. Top-1 rather than top-3 because most opponents in
    # this corpus appear in only two or three families, so a top-3 predicate
    # selects everything they have and leaves nothing to filter out. With
    # top-1, the opponent's other games act as distractors, and a retriever
    # that can only match the username cannot separate them.
    opponents = [(o, ds) for o, ds in by_opponent.items() if len(ds) >= 8]
    opponents.sort(key=lambda kv: -len(kv[1]))
    for i, (opp, opp_docs) in enumerate(opponents[:max_per_family * 2]):
        fam_counts = Counter(_family_of(d) for d in opp_docs if _family_of(d))
        if len(fam_counts) < 2:
            continue                      # no distractors, so no join to test
        top_fam = fam_counts.most_common(1)[0][0]
        rel = {d.doc_id for d in opp_docs if _family_of(d) == top_fam}
        distractors = len(opp_docs) - len(rel)
        if not sized(rel) or distractors < 3:
            continue
        name = opp_docs[0].meta.get("player")
        queries.append(EvalQuery(
            qid=f"rel{i}", family="relational",
            text=(f"Which opening variation should I prepare against {name}, "
                  f"and what recurring mistakes do they make in those positions?"),
            relevant=rel,
            description=(f"{name}'s mistakes in their most played family "
                         f"({top_fam}). {len(rel)} relevant, {distractors} "
                         f"same-opponent distractors."),
            meta={"opponent": name, "family": top_fam,
                  "n_relevant": len(rel), "n_distractors": distractors},
        ))
        if len(queries) > 60:
            break

    # Subject-facing relational: a three way join of most played families,
    # colour and game phase. None of those three are stated as filters in the
    # question, which is what makes it relational rather than lookup.
    subj_fam_counts = Counter(_family_of(d) for d in subject_docs if _family_of(d))
    top_subj_fams = {f for f, _ in subj_fam_counts.most_common(4)}
    for color in ("white", "black"):
        for phase in ("middlegame", "endgame"):
            rel = {d.doc_id for d in subject_docs
                   if d.meta.get("color") == color
                   and d.meta.get("phase") == phase
                   and _family_of(d) in top_subj_fams}
            if not sized(rel):
                continue
            queries.append(EvalQuery(
                qid=f"relself_{color}_{phase}", family="relational",
                text=(f"In the openings I play most often as {color}, what do I "
                      f"keep getting wrong once the {phase} starts?"),
                relevant=rel,
                description=(f"Three way join: top-4 families, {color}, {phase}. "
                             f"{len(rel)} relevant."),
                meta={"color": color, "phase": phase,
                      "families": sorted(top_subj_fams), "n_relevant": len(rel)},
            ))

    rng.shuffle(queries)
    return queries


def query_set_stats(queries: list[EvalQuery]) -> dict:
    by_fam = Counter(q.family for q in queries)
    sizes = sorted(len(q.relevant) for q in queries)
    return {
        "total": len(queries),
        "by_family": dict(by_fam),
        "avg_relevant": round(sum(sizes) / max(len(sizes), 1), 1),
        "min_relevant": sizes[0] if sizes else 0,
        "max_relevant": sizes[-1] if sizes else 0,
        "median_relevant": sizes[len(sizes) // 2] if sizes else 0,
    }
