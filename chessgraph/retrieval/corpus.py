"""The shared retrieval corpus.

WHY THIS FILE EXISTS
--------------------
The experiment compares four retrieval strategies. That comparison is only
valid if all four search the SAME candidate set. If the graph retriever can
reach documents the vector retriever cannot even see, any win it posts measures
corpus construction, not retrieval.

So: one corpus, built once, here. Every retriever indexes these exact
documents. Keyword and vector use `doc.text`; the graph retriever uses
`doc.meta` to locate the same documents by traversal. Same units, same IDs,
different access paths.

THE RETRIEVABLE UNIT
--------------------
One document = one mistake instance: a specific move, in a specific game, that
lost at least 100 centipawns.

That choice is deliberate. It is the smallest thing that is:
  - citable  (game URL + move number, so a report can ground every claim)
  - countable (recurring patterns are repeated documents, not prose)
  - checkable (you can open the position and verify the claim)

Aggregates like "you lose to forks" are computed FROM these, never stored as
free text, so no claim in the final report exists without instances behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from chessgraph.store.db import Store


@dataclass
class Document:
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


ORDINAL_COLOR = {"white": "as White", "black": "as Black"}


def render_mistake(row, themes: list[str]) -> str:
    """Natural-language rendering of one mistake.

    Themes and opening lead, then the move detail. Front-loading the
    discriminative fields helps BM25 and costs nothing elsewhere.

    A NOTE ON A HYPOTHESIS THAT DID NOT SURVIVE
    An earlier version wrapped every field in a full sentence, and dense
    retrieval performed badly on it. The suspicion was embedding collapse from
    shared boilerplate: measured mean pairwise cosine was 0.809 across 3,790
    documents. Rewriting to this denser form moved it to 0.819, which is to say
    not at all. High absolute cosine is simply normal for bge-small on short
    same-domain text, and absolute magnitude was the wrong thing to measure.
    Relative ordering is what matters.

    The real cause is recorded in `experiments/vocabulary_gap.py`: the model
    does soft lexical matching, not conceptual mapping, so it cannot connect a
    chess motif to its description unless they share ordinary English words.
    """
    theme_txt = ", ".join(t.replace("_", " ") for t in themes) or "no tactical motif"
    opening = row["opening"] or "unnamed opening"
    family = (row["opening"] or "").split(":")[0].strip() or "unknown"
    color = row["color"]
    sev = row["judgment"] or "mistake"
    better = row["best_move_san"] or "unknown"
    return (
        f"{theme_txt}. {opening}. {family}. "
        f"{color} {sev} at move {row['move_number']}, {row['phase']} phase. "
        f"Played {row['san']}, better was {better}, lost {row['cp_loss']} "
        f"centipawns. ECO {row['eco'] or 'unknown'}. Player {row['player']}."
    )


def build_corpus(store: Store, *, min_cp_loss: int = 100,
                 subject_only: bool = False) -> list[Document]:
    """Materialise every mistake instance as a Document."""
    where_subject = "AND m.is_subject_move = 1" if subject_only else ""
    rows = store.q(
        f"""
        SELECT m.game_id, m.ply, m.move_number, m.color, m.san, m.uci,
               m.fen_before, m.pos_key, m.cp_loss, m.best_move_san, m.judgment,
               m.is_subject_move, m.clock_seconds,
               g.opening, g.eco, g.url, g.date, g.speed,
               g.subject, g.opponent, g.subject_color, g.subject_score,
               CASE WHEN m.is_subject_move = 1 THEN g.subject ELSE g.opponent END AS player,
               p.phase
        FROM moves m
        JOIN games g ON g.game_id = m.game_id
        LEFT JOIN positions p ON p.pos_key = m.pos_key
        WHERE m.cp_loss >= ? {where_subject}
        ORDER BY m.game_id, m.ply
        """, (min_cp_loss,))

    theme_rows = store.q("SELECT game_id, ply, theme FROM move_themes")
    themes_by_move: dict[tuple[str, int], list[str]] = {}
    for t in theme_rows:
        themes_by_move.setdefault((t["game_id"], t["ply"]), []).append(t["theme"])

    docs = []
    for r in rows:
        key = (r["game_id"], r["ply"])
        themes = themes_by_move.get(key, [])
        docs.append(Document(
            doc_id=f"{r['game_id']}:{r['ply']}",
            text=render_mistake(r, themes),
            meta={
                "game_id": r["game_id"], "ply": r["ply"],
                "move_number": r["move_number"], "color": r["color"],
                "player": r["player"], "is_subject_move": bool(r["is_subject_move"]),
                "opening": r["opening"],
                "family": (r["opening"] or "").split(":")[0].strip() or None,
                "eco": r["eco"], "phase": r["phase"],
                "pos_key": r["pos_key"], "fen": r["fen_before"],
                "san": r["san"], "best": r["best_move_san"],
                "cp_loss": r["cp_loss"], "judgment": r["judgment"],
                "themes": themes,
                "url": r["url"], "date": r["date"], "speed": r["speed"],
                "opponent": r["opponent"], "subject": r["subject"],
                "clock_seconds": r["clock_seconds"],
            },
        ))
    return docs


def corpus_stats(docs: Iterable[Document]) -> dict:
    docs = list(docs)
    from collections import Counter
    themes = Counter(t for d in docs for t in d.meta.get("themes", []))
    return {
        "documents": len(docs),
        "subject_mistakes": sum(1 for d in docs if d.meta["is_subject_move"]),
        "distinct_openings": len({d.meta["opening"] for d in docs if d.meta["opening"]}),
        "distinct_families": len({d.meta["family"] for d in docs if d.meta["family"]}),
        "top_themes": dict(themes.most_common(10)),
        "avg_text_len": round(sum(len(d.text) for d in docs) / max(len(docs), 1)),
    }
