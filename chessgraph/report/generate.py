"""Preparation report generation.

DESIGN: CLAIMS AND CITATIONS, NOT PROSE
The report is built as structured data first. Every claim carries the citations
that support it and the numbers it was derived from. Rendering to Markdown is
the last step and adds nothing.

This matters for two reasons.

1. Grounding becomes checkable. Because a claim cannot exist without citations
   attached, `evaluation/grounding.py` can verify that every citation resolves
   to a real move and that the move actually supports what the claim says.
   A report written as free text can only be spot-checked by a human.

2. No language model is required for the MVP, so the baseline report has a
   hallucination rate of zero by construction. When an LLM is added later to
   write the prose, grounding accuracy measures how much fidelity that layer
   costs. Without this structured baseline there is nothing to measure against.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from chessgraph.engine.analyzer import average_cp_loss
from chessgraph.store.db import Store
from chessgraph.store.graph import ChessKnowledgeGraph

THEME_LABEL = {
    "fork": "losing material to forks",
    "hanging_piece": "leaving pieces undefended",
    "hangs_material": "giving away material outright",
    "back_rank_mate": "back rank mates",
    "back_rank_weakness": "back rank vulnerability",
    "absolute_pin": "pieces pinned against the king",
    "pin": "pinned pieces",
    "skewer": "skewers",
    "discovered_attack": "discovered attacks",
    "accepts_unsound_sacrifice": "accepting unsound sacrifices",
    "endgame_technique": "endgame technique",
    "king_safety": "king safety",
    "positional_error": "positional drift",
}

THEME_ADVICE = {
    "fork": "Drill knight fork patterns. Before each move, check which of your "
            "pieces share a square a knight could reach.",
    "hanging_piece": "Run a two second undefended piece scan before every move. "
                     "This is the single highest value habit at this level.",
    "hangs_material": "Same scan as above, applied to the square you are moving to "
                      "as well as the one you are leaving.",
    "back_rank_mate": "Make luft early once queens and rooks are still on.",
    "back_rank_weakness": "Watch the back rank whenever your rooks leave the first rank.",
    "absolute_pin": "Avoid lining pieces up in front of your king. Break pins early.",
    "pin": "Notice when a piece is frozen and add a defender or break the line.",
    "skewer": "Keep your queen and king off the same rank, file or diagonal.",
    "discovered_attack": "Check what an opponent's piece is standing in front of "
                         "before assuming it is harmless.",
    "accepts_unsound_sacrifice": "When material is offered, ask what it opens up "
                                 "before taking it. This is your most expensive habit.",
    "endgame_technique": "Study basic rook endings and king activity.",
    "king_safety": "Castle earlier and keep the pawn shield intact.",
    "positional_error": "Focus on piece activity and pawn structure over concrete lines.",
}


@dataclass
class Citation:
    game_id: str
    ply: int
    move_number: int
    color: str
    san: str
    best: str | None
    cp_loss: int | None
    url: str
    date: str
    fen: str
    themes: list[str] = field(default_factory=list)

    def ref(self) -> str:
        dots = "." if self.color == "white" else "..."
        return f"[{self.url} move {self.move_number}{dots}{self.san}]"


@dataclass
class Claim:
    text: str
    kind: str
    evidence: dict = field(default_factory=dict)
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Section:
    title: str
    claims: list[Claim] = field(default_factory=list)
    note: str = ""


@dataclass
class PrepReport:
    subject: str
    generated_at: str
    profile: dict
    sections: list[Section] = field(default_factory=list)
    training_positions: list[dict] = field(default_factory=list)

    def all_claims(self) -> list[Claim]:
        return [c for s in self.sections for c in s.claims]

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------- helpers
def _citations_for_moves(store: Store, rows, limit: int = 3) -> list[Citation]:
    out = []
    for r in rows[:limit]:
        out.append(Citation(
            game_id=r["game_id"], ply=r["ply"], move_number=r["move_number"],
            color=r["color"], san=r["san"], best=r["best_move_san"],
            cp_loss=r["cp_loss"], url=r["url"], date=r["date"],
            fen=r["fen_before"],
            themes=(r["themes"].split(",") if r.keys().__contains__("themes")
                    and r["themes"] else []),
        ))
    return out


# -------------------------------------------------------------------- report
def build_report(store: Store, subject: str, kg: ChessKnowledgeGraph | None = None,
                 *, top_openings: int = 5, top_weaknesses: int = 5,
                 training_count: int = 12,
                 max_abs_eval_before: int = 300,
                 game_ids: set[str] | None = None) -> PrepReport:
    """Assemble the full preparation report from the store and graph.

    `game_ids` restricts the report to a subset, which the held-out evaluation
    uses to build a report from training games only and then test its claims
    against future games.

    `max_abs_eval_before` is a teaching-quality filter, not a statistical one.
    Ranking examples purely by centipawn loss surfaces moves played in already
    decided positions, where someone two pieces down shuffles into mate. Those
    have the largest losses and the least to teach. Restricting cited examples
    to positions still within 3 pawns of level means every example is a move
    that actually cost a playable game. Aggregate counts and averages are
    computed over ALL mistakes and are unaffected; only which instances get
    cited and drilled changes.
    """
    kg = kg or ChessKnowledgeGraph.build(store, subject)
    scope = ""
    params: list = [subject]
    if game_ids:
        placeholders = ",".join("?" * len(game_ids))
        scope = f" AND g.game_id IN ({placeholders})"
        params += list(game_ids)

    # ------------------------------------------------------------- profile
    prof = store.one(
        f"""SELECT COUNT(*) games, AVG(subject_score) score,
                   AVG(subject_elo) elo, MIN(date) first, MAX(date) last
            FROM games g WHERE subject = ?{scope}""", tuple(params))
    acpl_row = store.one(
        f"""SELECT AVG(m.cp_loss) acpl, COUNT(*) n
            FROM moves m JOIN games g ON g.game_id = m.game_id
            WHERE g.subject = ? AND m.is_subject_move = 1
              AND m.cp_loss IS NOT NULL AND m.judgment != 'book'{scope}""",
        tuple(params))
    blunder_row = store.one(
        f"""SELECT
              SUM(CASE WHEN m.judgment='blunder' THEN 1 ELSE 0 END) blunders,
              SUM(CASE WHEN m.judgment='mistake' THEN 1 ELSE 0 END) mistakes,
              SUM(CASE WHEN m.judgment='inaccuracy' THEN 1 ELSE 0 END) inaccuracies
            FROM moves m JOIN games g ON g.game_id = m.game_id
            WHERE g.subject = ? AND m.is_subject_move = 1{scope}""", tuple(params))

    n_games = prof["games"] or 0
    profile = {
        "games_analysed": n_games,
        "score": round(prof["score"] or 0, 3),
        "avg_rating": round(prof["elo"] or 0),
        "date_range": f"{prof['first']} to {prof['last']}",
        "acpl": round(acpl_row["acpl"] or 0, 1),
        "analysed_moves": acpl_row["n"] or 0,
        "blunders": blunder_row["blunders"] or 0,
        "mistakes": blunder_row["mistakes"] or 0,
        "inaccuracies": blunder_row["inaccuracies"] or 0,
        "blunders_per_game": round((blunder_row["blunders"] or 0) / max(n_games, 1), 2),
    }

    report = PrepReport(
        subject=subject,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        profile=profile,
    )

    # ------------------------------------------------- section 1: repertoire
    sec = Section("Repertoire",
                  note="Openings ranked by how often they occur, with the score "
                       "achieved in each. Score is out of 1.0.")
    for color in ("white", "black"):
        rows = store.q(
            f"""SELECT opening, COUNT(*) n, AVG(subject_score) score,
                       SUM(CASE WHEN subject_score=1 THEN 1 ELSE 0 END) wins,
                       SUM(CASE WHEN subject_score=0 THEN 1 ELSE 0 END) losses
                FROM games g
                WHERE subject = ? AND subject_color = ? AND opening IS NOT NULL{scope}
                GROUP BY opening ORDER BY n DESC LIMIT ?""",
            (subject, color, *params[1:], top_openings))
        for r in rows:
            fam = (r["opening"] or "").split(":")[0].strip()
            games = store.q(
                f"""SELECT game_id, url, date, subject_score FROM games g
                    WHERE subject = ? AND opening = ?{scope} LIMIT 3""",
                (subject, r["opening"], *params[1:]))
            cits = [Citation(game_id=g["game_id"], ply=0, move_number=0,
                             color=color, san="", best=None, cp_loss=None,
                             url=g["url"], date=g["date"], fen="")
                    for g in games]
            sec.claims.append(Claim(
                text=(f"As {color}, plays {r['opening']} in {r['n']} games "
                      f"scoring {r['score']:.2f} ({r['wins']}W {r['losses']}L)."),
                kind="repertoire",
                evidence={"opening": r["opening"], "family": fam, "color": color,
                          "games": r["n"], "score": round(r["score"], 3)},
                citations=cits))
    report.sections.append(sec)

    # ------------------------------------------------- section 2: weaknesses
    sec = Section("Recurring weaknesses",
                  note="Themes ranked by total centipawns lost, not by count. "
                       "A rare expensive mistake outranks a frequent cheap one.")
    theme_rows = store.q(
        f"""SELECT t.theme, COUNT(*) n, AVG(m.cp_loss) avg_cp, SUM(m.cp_loss) total_cp
            FROM move_themes t
            JOIN moves m ON m.game_id = t.game_id AND m.ply = t.ply
            JOIN games g ON g.game_id = m.game_id
            WHERE g.subject = ? AND m.is_subject_move = 1{scope}
            GROUP BY t.theme ORDER BY total_cp DESC LIMIT ?""",
        (subject, *params[1:], top_weaknesses))
    for r in theme_rows:
        examples = store.q(
            f"""SELECT m.game_id, m.ply, m.move_number, m.color, m.san,
                       m.best_move_san, m.cp_loss, m.fen_before, g.url, g.date
                FROM move_themes t
                JOIN moves m ON m.game_id = t.game_id AND m.ply = t.ply
                JOIN games g ON g.game_id = m.game_id
                WHERE g.subject = ? AND m.is_subject_move = 1 AND t.theme = ?
                  AND ABS(COALESCE(m.eval_before_cp, 0)) <= ?{scope}
                ORDER BY m.cp_loss DESC LIMIT 3""",
            (subject, r["theme"], max_abs_eval_before, *params[1:]))
        label = THEME_LABEL.get(r["theme"], r["theme"].replace("_", " "))
        sec.claims.append(Claim(
            text=(f"{label.capitalize()}: {r['n']} occurrences, "
                  f"average {round(r['avg_cp'])}cp lost, "
                  f"{round(r['total_cp'] / 100)} pawns of total damage. "
                  f"{THEME_ADVICE.get(r['theme'], '')}").strip(),
            kind="weakness",
            evidence={"theme": r["theme"], "count": r["n"],
                      "avg_cp_loss": round(r["avg_cp"]),
                      "total_cp_lost": r["total_cp"],
                      "per_game": round(r["n"] / max(n_games, 1), 2)},
            citations=_citations_for_moves(store, examples)))
    report.sections.append(sec)

    # -------------------------------------- section 3: opening trouble spots
    sec = Section("Where the openings go wrong",
                  note="Openings ranked by average centipawn loss, restricted "
                       "to those with enough games to be meaningful.")
    rows = store.q(
        f"""SELECT g.opening, COUNT(DISTINCT g.game_id) games,
                   AVG(m.cp_loss) acpl, COUNT(*) moves
            FROM moves m JOIN games g ON g.game_id = m.game_id
            WHERE g.subject = ? AND m.is_subject_move = 1
              AND m.cp_loss IS NOT NULL AND m.judgment != 'book'
              AND g.opening IS NOT NULL{scope}
            GROUP BY g.opening HAVING games >= 4
            ORDER BY acpl DESC LIMIT ?""",
        (subject, *params[1:], top_weaknesses))
    for r in rows:
        worst = store.q(
            f"""SELECT m.game_id, m.ply, m.move_number, m.color, m.san,
                       m.best_move_san, m.cp_loss, m.fen_before, g.url, g.date
                FROM moves m JOIN games g ON g.game_id = m.game_id
                WHERE g.subject = ? AND m.is_subject_move = 1
                  AND g.opening = ? AND m.cp_loss >= 100
                  AND ABS(COALESCE(m.eval_before_cp, 0)) <= ?{scope}
                ORDER BY m.cp_loss DESC LIMIT 3""",
            (subject, r["opening"], max_abs_eval_before, *params[1:]))
        sec.claims.append(Claim(
            text=(f"{r['opening']}: {round(r['acpl'], 1)} average centipawn loss "
                  f"across {r['games']} games."),
            kind="opening_weakness",
            evidence={"opening": r["opening"], "games": r["games"],
                      "acpl": round(r["acpl"], 1)},
            citations=_citations_for_moves(store, worst)))
    report.sections.append(sec)

    # ------------------------------------------- section 4: training set
    train_rows = store.q(
        f"""SELECT m.game_id, m.ply, m.move_number, m.color, m.san,
                   m.best_move_san, m.best_move_uci, m.cp_loss, m.fen_before,
                   g.url, g.date, g.opening
            FROM moves m JOIN games g ON g.game_id = m.game_id
            WHERE g.subject = ? AND m.is_subject_move = 1
              AND m.cp_loss >= 200 AND m.best_move_san IS NOT NULL
              AND ABS(COALESCE(m.eval_before_cp, 0)) <= ?{scope}
            ORDER BY m.cp_loss DESC""",
        (subject, max_abs_eval_before, *params[1:]))
    seen_pos: set[str] = set()
    for r in train_rows:
        if len(report.training_positions) >= training_count:
            break
        # Deduplicate by position so a repeated opening trap does not fill the
        # whole training set with the same puzzle.
        key = " ".join(r["fen_before"].split(" ")[:4])
        if key in seen_pos:
            continue
        seen_pos.add(key)
        report.training_positions.append({
            "fen": r["fen_before"],
            "side_to_move": r["color"],
            "you_played": r["san"],
            "best_move": r["best_move_san"],
            "best_uci": r["best_move_uci"],
            "cp_loss": r["cp_loss"],
            "opening": r["opening"],
            "source_url": r["url"],
            "move_number": r["move_number"],
            "game_id": r["game_id"],
            "ply": r["ply"],
        })

    return report


# -------------------------------------------------------------------- render
def render_markdown(report: PrepReport) -> str:
    p = report.profile
    lines = [
        f"# Preparation report: {report.subject}",
        "",
        f"Generated {report.generated_at} from {p['games_analysed']} analysed games "
        f"({p['date_range']}).",
        "",
        "## Profile",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Games analysed | {p['games_analysed']} |",
        f"| Average rating | {p['avg_rating']} |",
        f"| Score | {p['score']:.3f} |",
        f"| Average centipawn loss | {p['acpl']} |",
        f"| Blunders | {p['blunders']} ({p['blunders_per_game']} per game) |",
        f"| Mistakes | {p['mistakes']} |",
        f"| Inaccuracies | {p['inaccuracies']} |",
        "",
    ]
    for sec in report.sections:
        lines.append(f"## {sec.title}")
        lines.append("")
        if sec.note:
            lines.append(f"_{sec.note}_")
            lines.append("")
        for claim in sec.claims:
            lines.append(f"- {claim.text}")
            for c in claim.citations:
                if c.san:
                    lines.append(
                        f"    - {c.date} `{c.san}` lost {c.cp_loss}cp, "
                        f"{c.best} was better. {c.url}")
                else:
                    lines.append(f"    - {c.date} {c.url}")
        lines.append("")

    if report.training_positions:
        lines += ["## Training positions", "",
                  "_Positions from your own games where you lost at least 200 "
                  "centipawns. Set each up and find the move._", "",
                  "| # | Opening | You played | Best | Lost | FEN |",
                  "|---|---|---|---|---|---|"]
        for i, t in enumerate(report.training_positions, 1):
            lines.append(
                f"| {i} | {(t['opening'] or '')[:30]} | {t['you_played']} | "
                f"{t['best_move']} | {t['cp_loss']}cp | `{t['fen']}` |")
        lines.append("")
    return "\n".join(lines)
