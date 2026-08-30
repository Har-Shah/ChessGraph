"""Bounded tools for an interactive preparation agent.

DESIGN PRINCIPLE: EVERY TOOL RETURNS A SMALL, BOUNDED RESULT
The corpus holds 3,790 mistake documents and 25,000 positions. A tool that can
return all of them will eventually return all of them, and the agent's context
fills with data it cannot use. Every function here caps its output, and the cap
is a parameter with a conservative default rather than something the model can
raise without limit.

Results are JSON-serialisable dicts, not prose. The model does the reasoning;
the tools supply facts with citations attached, so anything the agent says can
be traced back to a game and a move number.

The tools are plain Python functions and are usable without any model. The
`@beta_tool` decorator is applied in `runner.py`, which keeps this module
importable and testable with no API key present.
"""
from __future__ import annotations

from typing import Any

import chess

from chessgraph.config import ENGINE
from chessgraph.engine.analyzer import Analyzer, win_probability
from chessgraph.report.generate import build_report, render_markdown
from chessgraph.retrieval.corpus import build_corpus
from chessgraph.retrieval.graph_retriever import GraphRetriever
from chessgraph.retrieval.hybrid import HybridRetriever
from chessgraph.retrieval.keyword import BM25Retriever
from chessgraph.store.db import Store
from chessgraph.store.graph import ChessKnowledgeGraph

MAX_RESULTS = 10


class ChessGraphTools:
    """Holds the store, graph and retrievers so tools do not rebuild them.

    Building the corpus and graph takes a few seconds. Doing it per tool call
    would dominate agent latency, so the agent constructs this once per session
    and the tools become cheap lookups.
    """

    def __init__(self, subject: str, store: Store | None = None):
        self.subject = subject
        self.store = store or Store()
        self.kg = ChessKnowledgeGraph.build(self.store, subject)
        self.docs = build_corpus(self.store)
        self._bm25 = BM25Retriever()
        self._graph = GraphRetriever(self.kg, subject=subject)
        self._bm25.index(self.docs)
        self._graph.index(self.docs)
        self.retriever = HybridRetriever([self._bm25, self._graph],
                                         name="hybrid_keyword_graph")

    # ------------------------------------------------------------------ tools
    def fetch_player_games(self, player: str = "", limit: int = MAX_RESULTS,
                           color: str = "", opening: str = "") -> dict:
        """Games for a player, most recent first, with results and openings."""
        player = player or self.subject
        limit = min(limit, 25)
        clauses = ["(subject = ? OR opponent = ?)"]
        params: list[Any] = [player, player]
        if color:
            clauses.append("subject_color = ?")
            params.append(color)
        if opening:
            clauses.append("opening LIKE ?")
            params.append(f"%{opening}%")
        rows = self.store.q(
            f"""SELECT game_id, url, date, white, black, result, opening, eco,
                       speed, subject_color, subject_score, ply_count
                FROM games WHERE {' AND '.join(clauses)}
                ORDER BY date DESC LIMIT ?""", (*params, limit))
        return {
            "player": player,
            "count": len(rows),
            "games": [dict(r) for r in rows],
        }

    def analyze_position(self, fen: str, depth: int = 16,
                         multipv: int = 3) -> dict:
        """Evaluate a position and return the top candidate moves."""
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            return {"error": f"invalid FEN: {exc}"}
        depth = min(depth, 22)
        from chessgraph.config import EngineConfig
        cfg = EngineConfig(depth=depth, multipv=min(multipv, 5))
        with Analyzer(config=cfg) as az:
            ev = az.evaluate(board, depth=depth)
        cp = ev.clamped_cp()
        return {
            "fen": fen,
            "side_to_move": "white" if board.turn else "black",
            "eval_cp": ev.score_cp,
            "mate_in": ev.mate,
            "win_probability_for_side_to_move": round(win_probability(cp), 3),
            "best_move": ev.best_san,
            "principal_variation": ev.pv,
            "candidates": [
                {"san": a["san"], "uci": a["uci"], "score_cp": a["score_cp"],
                 "mate": a["mate"]}
                for a in ev.alternatives[:multipv]
            ],
            "depth": depth,
        }

    def retrieve_similar_games(self, question: str,
                               limit: int = MAX_RESULTS) -> dict:
        """Search the mistake corpus. Returns cited instances, not prose."""
        limit = min(limit, MAX_RESULTS)
        res = self.retriever.search(question, k=limit)
        return {
            "question": question,
            "retriever": res.retriever,
            "latency_ms": round(res.latency_ms, 1),
            "results": [
                {
                    "game_url": h.doc.meta["url"],
                    "date": h.doc.meta["date"],
                    "player": h.doc.meta["player"],
                    "opening": h.doc.meta["opening"],
                    "move_number": h.doc.meta["move_number"],
                    "color": h.doc.meta["color"],
                    "played": h.doc.meta["san"],
                    "better": h.doc.meta["best"],
                    "cp_loss": h.doc.meta["cp_loss"],
                    "themes": h.doc.meta["themes"],
                    "fen": h.doc.meta["fen"],
                    "why_retrieved": h.explanation,
                }
                for h in res.hits if h.doc
            ],
        }

    def find_opening_weaknesses(self, player: str = "", color: str = "",
                                min_games: int = 4,
                                limit: int = MAX_RESULTS) -> dict:
        """Openings ranked by average centipawn loss, with score and volume."""
        player = player or self.subject
        limit = min(limit, MAX_RESULTS)
        clauses = ["g.subject = ?", "m.cp_loss IS NOT NULL",
                   "m.judgment != 'book'", "g.opening IS NOT NULL"]
        params: list[Any] = [player]
        if color:
            clauses.append("g.subject_color = ?")
            params.append(color)
        rows = self.store.q(
            f"""SELECT g.opening, g.eco, COUNT(DISTINCT g.game_id) games,
                       AVG(m.cp_loss) acpl, AVG(g.subject_score) score
                FROM moves m JOIN games g ON g.game_id = m.game_id
                WHERE {' AND '.join(clauses)} AND m.is_subject_move = 1
                GROUP BY g.opening HAVING games >= ?
                ORDER BY acpl DESC LIMIT ?""", (*params, min_games, limit))
        return {
            "player": player, "color": color or "both",
            "openings": [
                {"opening": r["opening"], "eco": r["eco"], "games": r["games"],
                 "acpl": round(r["acpl"], 1), "score": round(r["score"], 3)}
                for r in rows
            ],
        }

    def find_recurring_themes(self, player: str = "",
                              limit: int = MAX_RESULTS) -> dict:
        """Tactical themes ranked by total centipawns lost."""
        player = player or self.subject
        rows = self.store.q(
            """SELECT t.theme, COUNT(*) n, AVG(m.cp_loss) avg_cp,
                      SUM(m.cp_loss) total_cp
               FROM move_themes t
               JOIN moves m ON m.game_id = t.game_id AND m.ply = t.ply
               JOIN games g ON g.game_id = m.game_id
               WHERE g.subject = ? AND m.is_subject_move = 1
               GROUP BY t.theme ORDER BY total_cp DESC LIMIT ?""",
            (player, min(limit, MAX_RESULTS)))
        return {
            "player": player,
            "themes": [
                {"theme": r["theme"], "occurrences": r["n"],
                 "avg_cp_loss": round(r["avg_cp"]),
                 "total_pawns_lost": round(r["total_cp"] / 100)}
                for r in rows
            ],
        }

    def generate_training_positions(self, count: int = 8,
                                    theme: str = "",
                                    max_abs_eval: int = 300) -> dict:
        """Positions from the player's own games, as drillable puzzles.

        `max_abs_eval` keeps positions that were still competitive. A blunder
        made while three pieces down is not a useful exercise.
        """
        count = min(count, 20)
        clauses = ["g.subject = ?", "m.is_subject_move = 1", "m.cp_loss >= 200",
                   "m.best_move_san IS NOT NULL",
                   "ABS(COALESCE(m.eval_before_cp, 0)) <= ?"]
        params: list[Any] = [self.subject, max_abs_eval]
        join = ""
        if theme:
            join = "JOIN move_themes t ON t.game_id = m.game_id AND t.ply = m.ply"
            clauses.append("t.theme = ?")
            params.append(theme)
        rows = self.store.q(
            f"""SELECT DISTINCT m.fen_before, m.san, m.best_move_san,
                       m.cp_loss, m.move_number, m.color, g.url, g.opening
                FROM moves m JOIN games g ON g.game_id = m.game_id {join}
                WHERE {' AND '.join(clauses)}
                ORDER BY m.cp_loss DESC LIMIT ?""", (*params, count * 3))
        seen, out = set(), []
        for r in rows:
            key = " ".join(r["fen_before"].split(" ")[:4])
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "fen": r["fen_before"], "side_to_move": r["color"],
                "you_played": r["san"], "best_move": r["best_move_san"],
                "cp_loss": r["cp_loss"], "move_number": r["move_number"],
                "opening": r["opening"], "source": r["url"],
            })
            if len(out) >= count:
                break
        return {"theme": theme or "any", "count": len(out), "positions": out}

    def retrieve_similar_positions(self, fen: str = "", pos_key: str = "",
                                   limit: int = 5,
                                   cross_opening_only: bool = False) -> dict:
        """Structurally similar positions from the player's own history.

        Similarity is computed from pawn structure, material, king placement
        and piece configuration, not from names or text. `cross_opening_only`
        restricts results to a different opening family, which surfaces
        transpositions that no metadata filter can find.
        """
        from chessgraph.models import position_key
        from chessgraph.store.graph import position_node

        limit = min(limit, MAX_RESULTS)
        key = pos_key or (position_key(fen) if fen else "")
        if not key:
            return {"error": "supply a fen or a pos_key"}
        node = position_node(key)
        if node not in self.kg.g:
            return {"error": "position not present in this player's graph",
                    "pos_key": key}

        found = (self.kg.similar_across_openings(node, k=limit)
                 if cross_opening_only else
                 self.kg.similar_positions(node, k=limit))

        enriched = []
        for r in found:
            k = r["position"].split(":", 1)[1]
            row = self.store.one(
                """SELECT m.san, m.best_move_san, m.cp_loss, g.url, g.date,
                          g.opening
                   FROM moves m JOIN games g ON g.game_id = m.game_id
                   WHERE m.pos_key = ? ORDER BY m.cp_loss DESC LIMIT 1""", (k,))
            enriched.append({
                **{kk: r[kk] for kk in ("score", "fen", "opening", "phase", "ply")},
                "you_played": row["san"] if row else None,
                "better_was": row["best_move_san"] if row else None,
                "cp_loss": row["cp_loss"] if row else None,
                "game": row["url"] if row else None,
            })
        return {"query_pos_key": key, "cross_opening_only": cross_opening_only,
                "count": len(enriched), "similar": enriched}

    def build_opponent_report(self, opponent: str, limit: int = 5) -> dict:
        """Prep sheet for one opponent: repertoire, results, exploitable errors.

        This is the multi-hop query the knowledge graph exists for. It joins
        what the opponent plays against where they go wrong, and neither half
        is known before the query runs.
        """
        rows = self.store.q(
            """SELECT COUNT(*) games, AVG(1.0 - subject_score) opp_score
               FROM games WHERE opponent = ?""", (opponent,))
        header = dict(rows[0]) if rows else {}

        openings = self.store.q(
            """SELECT opening,
                      CASE subject_color WHEN 'white' THEN 'black' ELSE 'white' END color,
                      COUNT(*) n, AVG(1.0 - subject_score) score
               FROM games WHERE opponent = ? AND opening IS NOT NULL
               GROUP BY opening, color ORDER BY n DESC LIMIT ?""",
            (opponent, limit))

        mistakes = self.store.q(
            """SELECT g.opening, m.move_number, m.color, m.san, m.best_move_san,
                      m.cp_loss, m.fen_before, g.url, g.date,
                      GROUP_CONCAT(t.theme) themes
               FROM moves m
               JOIN games g ON g.game_id = m.game_id
               LEFT JOIN move_themes t ON t.game_id = m.game_id AND t.ply = m.ply
               WHERE g.opponent = ? AND m.is_subject_move = 0 AND m.cp_loss >= 150
                 AND ABS(COALESCE(m.eval_before_cp, 0)) <= 300
               GROUP BY m.game_id, m.ply
               ORDER BY m.cp_loss DESC LIMIT ?""", (opponent, limit * 2))

        themes = self.store.q(
            """SELECT t.theme, COUNT(*) n FROM move_themes t
               JOIN moves m ON m.game_id = t.game_id AND m.ply = t.ply
               JOIN games g ON g.game_id = m.game_id
               WHERE g.opponent = ? AND m.is_subject_move = 0
               GROUP BY t.theme ORDER BY n DESC LIMIT ?""", (opponent, limit))

        return {
            "opponent": opponent,
            "games_against": header.get("games", 0),
            "their_score_against_you": round(header.get("opp_score") or 0, 3),
            "their_openings": [dict(r) for r in openings],
            "their_recurring_themes": [dict(r) for r in themes],
            "exploitable_mistakes": [
                {"opening": r["opening"], "move": r["move_number"],
                 "color": r["color"], "they_played": r["san"],
                 "better_was": r["best_move_san"], "cp_loss": r["cp_loss"],
                 "themes": (r["themes"] or "").split(","),
                 "fen": r["fen_before"], "game": r["url"], "date": r["date"]}
                for r in mistakes
            ],
        }

    def generate_prep_report(self, markdown: bool = False) -> dict:
        """The full cited preparation report."""
        report = build_report(self.store, self.subject, self.kg)
        if markdown:
            return {"markdown": render_markdown(report)}
        return {
            "profile": report.profile,
            "sections": [
                {"title": s.title,
                 "claims": [{"text": c.text, "evidence": c.evidence,
                             "citations": [cit.ref() for cit in c.citations]}
                            for c in s.claims]}
                for s in report.sections
            ],
            "training_positions": report.training_positions[:8],
        }

    def close(self) -> None:
        self.store.close()
