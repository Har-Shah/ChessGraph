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

    Written to be genuinely searchable rather than to look pretty. It repeats
    the opening name, the family, the themes and the phase in plain words,
    because those are the terms a real question will use. A FEN string is
    included but contributes almost nothing to lexical matching, which is
    itself part of what the experiment measures.
    """
    color = ORDINAL_COLOR.get(row["color"], row["color"])
    opening = row["opening"] or "an unnamed opening"
    family = (row["opening"] or "").split(":")[0].strip() or "unknown"
    move_no = row["move_number"]
    sev = row["judgment"] or "mistake"
    theme_txt = ", ".join(t.replace("_", " ") for t in themes) or "no tactical motif"
    better = row["best_move_san"] or "a better move"
    return (
        f"{row['player']} playing {color} in the {opening} "
        f"({family}, ECO {row['eco'] or 'unknown'}) reached a {row['phase']} "
        f"position at move {move_no}. They played {row['san']}, a {sev} "
        f"losing {row['cp_loss']} centipawns. {better} was stronger. "
        f"Tactical themes: {theme_txt}. "
        f"Position FEN {row['fen_before']}."
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
