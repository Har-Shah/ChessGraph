"""Knowledge graph over the SQLite facts.

WHAT THE GRAPH BUYS US OVER SQL
-------------------------------
Being honest about this matters, because "we added a graph" is not a result.
Most single-hop questions here ("which openings does X play?") are a GROUP BY
and SQL wins. The graph earns its place on three specific shapes:

  1. Multi-hop joins whose depth is not known in advance.
     "Prepare against this opponent" walks
        opponent -> openings they play -> positions those reach -> mistakes
        they make there -> themes those mistakes share
     and then back out to *other* positions exhibiting the same themes. Written
     as SQL that is a five-way join with a self-join at the end, rewritten every
     time the question changes shape.

  2. Hierarchical rollup with fallback.
     Opening names are granular ("Sicilian Defense: Najdorf, English Attack").
     A specific variation may have 3 games, too few to conclude anything, while
     its family has 60. The graph models variation -> family explicitly, so a
     retriever can climb until it has enough evidence. In SQL that is string
     prefix matching, which is fragile.

  3. Similarity chains and transpositions.
     Positions link to each other by resemblance. Following those links two or
     three hops finds relevant material that shares no metadata with the query
     at all.

The graph is a DERIVED VIEW. SQLite stays the source of truth, and the graph is
rebuilt from it. They cannot drift apart.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from chessgraph.store.db import Store

# --------------------------------------------------------------- node helpers
def player_node(name: str) -> str:      return f"player:{name.lower()}"
def opening_node(name: str) -> str:     return f"opening:{name}"
def family_node(name: str) -> str:      return f"family:{name}"
def eco_node(code: str) -> str:         return f"eco:{code}"
def position_node(key: str) -> str:     return f"position:{key}"
def theme_node(name: str) -> str:       return f"theme:{name}"


def opening_family(opening: str | None) -> str | None:
    """'Sicilian Defense: Najdorf Variation, English Attack' -> 'Sicilian Defense'.

    Lichess opening names are `Family: Variation, Sub-variation`. Splitting on
    the first colon gives a rollup level with enough games to be significant
    when a specific variation does not.
    """
    if not opening:
        return None
    return opening.split(":")[0].strip()


@dataclass
class GraphStats:
    nodes: int
    edges: int
    by_kind: dict
    by_relation: dict


class ChessKnowledgeGraph:
    def __init__(self, graph: nx.MultiDiGraph, subject: str | None = None):
        self.g = graph
        self.subject = subject

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, store: Store, subject: str,
              *, min_position_seen: int = 2,
              include_opponents: bool = True) -> "ChessKnowledgeGraph":
        g = nx.MultiDiGraph()
        subj = player_node(subject)
        g.add_node(subj, kind="player", name=subject, is_subject=True)

        # --- openings, families, ECO codes ---------------------------------
        rows = store.q("SELECT DISTINCT eco, opening FROM games WHERE opening IS NOT NULL")
        for r in rows:
            op, eco = r["opening"], r["eco"]
            g.add_node(opening_node(op), kind="opening", name=op, eco=eco)
            if fam := opening_family(op):
                g.add_node(family_node(fam), kind="family", name=fam)
                g.add_edge(opening_node(op), family_node(fam), rel="in_family")
            if eco:
                g.add_node(eco_node(eco), kind="eco", name=eco)
                g.add_edge(opening_node(op), eco_node(eco), rel="has_eco")

        # --- player -> opening, with performance ---------------------------
        # One row per (player, colour, opening) carrying frequency AND score.
        # Both matter: an opening you play often but score badly in is exactly
        # what a prep report should surface first.
        plays = store.q(
            """
            SELECT subject AS player, subject_color AS color, opening,
                   COUNT(*) AS games, AVG(subject_score) AS score,
                   AVG(subject_elo) AS elo
            FROM games
            WHERE subject IS NOT NULL AND opening IS NOT NULL
            GROUP BY subject, subject_color, opening
            """)
        for r in plays:
            p = player_node(r["player"])
            if p not in g:
                g.add_node(p, kind="player", name=r["player"])
            g.add_edge(p, opening_node(r["opening"]), rel="plays",
                       color=r["color"], games=r["games"],
                       score=round(r["score"] or 0, 3))

        if include_opponents:
            opp = store.q(
                """
                SELECT opponent AS player,
                       CASE subject_color WHEN 'white' THEN 'black' ELSE 'white' END AS color,
                       opening, COUNT(*) AS games,
                       AVG(1.0 - subject_score) AS score
                FROM games
                WHERE opponent IS NOT NULL AND opening IS NOT NULL
                GROUP BY opponent, color, opening
                """)
            for r in opp:
                p = player_node(r["player"])
                if p not in g:
                    g.add_node(p, kind="player", name=r["player"])
                g.add_edge(p, opening_node(r["opening"]), rel="plays",
                           color=r["color"], games=r["games"],
                           score=round(r["score"] or 0, 3))

        # --- positions worth keeping ---------------------------------------
        # Everything a mistake happened in, plus anything recurring. Including
        # all 25k one-off positions would bloat the graph with nodes no query
        # can ever reach.
        pos_rows = store.q(
            """
            SELECT p.pos_key, p.fen, p.ply, p.phase, p.eco, p.opening,
                   p.seen_count, p.side_to_move, p.material_signature
            FROM positions p
            WHERE p.seen_count >= ?
               OR p.pos_key IN (SELECT DISTINCT pos_key FROM moves
                                WHERE cp_loss >= 100)
            """, (min_position_seen,))
        for r in pos_rows:
            g.add_node(position_node(r["pos_key"]), kind="position",
                       fen=r["fen"], ply=r["ply"], phase=r["phase"],
                       eco=r["eco"], opening=r["opening"],
                       seen_count=r["seen_count"],
                       side_to_move=r["side_to_move"],
                       material=r["material_signature"])

        # --- opening -> position -------------------------------------------
        leads = store.q(
            """
            SELECT gm.opening, m.pos_key, COUNT(*) AS n, MIN(m.ply) AS first_ply
            FROM moves m JOIN games gm ON gm.game_id = m.game_id
            WHERE gm.opening IS NOT NULL AND m.ply <= 30
            GROUP BY gm.opening, m.pos_key
            """)
        for r in leads:
            pn = position_node(r["pos_key"])
            if pn in g:
                g.add_edge(opening_node(r["opening"]), pn, rel="leads_to",
                           count=r["n"], ply=r["first_ply"])

        # --- player -> position (mistakes) ----------------------------------
        blunders = store.q(
            """
            SELECT CASE WHEN m.is_subject_move = 1 THEN gm.subject ELSE gm.opponent END AS player,
                   m.pos_key, m.game_id, m.ply, m.san, m.cp_loss,
                   m.best_move_san, m.judgment
            FROM moves m JOIN games gm ON gm.game_id = m.game_id
            WHERE m.cp_loss >= 100
            """)
        agg: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"count": 0, "total_cp": 0, "examples": []})
        for r in blunders:
            if not r["player"]:
                continue
            k = (r["player"], r["pos_key"])
            a = agg[k]
            a["count"] += 1
            a["total_cp"] += r["cp_loss"] or 0
            if len(a["examples"]) < 3:
                a["examples"].append({
                    "game_id": r["game_id"], "ply": r["ply"], "played": r["san"],
                    "better": r["best_move_san"], "cp_loss": r["cp_loss"],
                    "judgment": r["judgment"],
                })
        for (player, pos_key), a in agg.items():
            pn, pl = position_node(pos_key), player_node(player)
            if pn not in g:
                continue
            if pl not in g:
                g.add_node(pl, kind="player", name=player)
            g.add_edge(pl, pn, rel="blunders_at", count=a["count"],
                       avg_cp_loss=round(a["total_cp"] / a["count"]),
                       examples=a["examples"])

        # --- player -> theme, position -> theme ------------------------------
        themes = store.q(
            """
            SELECT CASE WHEN m.is_subject_move = 1 THEN gm.subject ELSE gm.opponent END AS player,
                   t.theme, m.pos_key, COUNT(*) AS n, AVG(m.cp_loss) AS avg_cp
            FROM move_themes t
            JOIN moves m ON m.game_id = t.game_id AND m.ply = t.ply
            JOIN games gm ON gm.game_id = m.game_id
            GROUP BY player, t.theme, m.pos_key
            """)
        player_theme: dict[tuple[str, str], dict] = defaultdict(
            lambda: {"count": 0, "cp": 0.0})
        for r in themes:
            if not r["player"]:
                continue
            tn = theme_node(r["theme"])
            if tn not in g:
                g.add_node(tn, kind="theme", name=r["theme"])
            pn = position_node(r["pos_key"])
            if pn in g:
                g.add_edge(pn, tn, rel="exhibits", count=r["n"])
            pt = player_theme[(r["player"], r["theme"])]
            pt["count"] += r["n"]
            pt["cp"] += (r["avg_cp"] or 0) * r["n"]
        for (player, theme), v in player_theme.items():
            pl = player_node(player)
            if pl not in g:
                g.add_node(pl, kind="player", name=player)
            g.add_edge(pl, theme_node(theme), rel="struggles_with",
                       count=v["count"],
                       avg_cp_loss=round(v["cp"] / max(v["count"], 1)))

        return cls(g, subject=subject)

    # ------------------------------------------------------------------ stats
    def stats(self) -> GraphStats:
        by_kind, by_rel = defaultdict(int), defaultdict(int)
        for _, d in self.g.nodes(data=True):
            by_kind[d.get("kind", "?")] += 1
        for _, _, d in self.g.edges(data=True):
            by_rel[d.get("rel", "?")] += 1
        return GraphStats(
            nodes=self.g.number_of_nodes(),
            edges=self.g.number_of_edges(),
            by_kind=dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
            by_relation=dict(sorted(by_rel.items(), key=lambda kv: -kv[1])),
        )

    # -------------------------------------------------------------- traversal
    def out(self, node: str, rel: str | None = None):
        """Outgoing edges, optionally filtered by relation type."""
        for _, tgt, d in self.g.out_edges(node, data=True):
            if rel is None or d.get("rel") == rel:
                yield tgt, d

    def inn(self, node: str, rel: str | None = None):
        for src, _, d in self.g.in_edges(node, data=True):
            if rel is None or d.get("rel") == rel:
                yield src, d

    def openings_played(self, player: str, color: str | None = None,
                        min_games: int = 1) -> list[dict]:
        """Openings a player uses, ranked by frequency."""
        out = []
        for tgt, d in self.out(player_node(player), rel="plays"):
            if color and d.get("color") != color:
                continue
            if d.get("games", 0) < min_games:
                continue
            out.append({
                "opening": self.g.nodes[tgt].get("name"),
                "node": tgt,
                "color": d.get("color"),
                "games": d.get("games"),
                "score": d.get("score"),
                "eco": self.g.nodes[tgt].get("eco"),
            })
        return sorted(out, key=lambda r: -r["games"])

    def weaknesses(self, player: str, top_k: int = 10) -> list[dict]:
        """Themes this player loses points to, ranked by total damage.

        Ranked by count * avg_cp_loss rather than raw count: a theme that
        happens 5 times at 400cp each costs more than one that happens 30 times
        at 110cp, and the report should lead with the expensive one.
        """
        out = []
        for tgt, d in self.out(player_node(player), rel="struggles_with"):
            count, avg = d.get("count", 0), d.get("avg_cp_loss", 0)
            out.append({
                "theme": self.g.nodes[tgt].get("name"),
                "count": count,
                "avg_cp_loss": avg,
                "total_cp_lost": count * avg,
            })
        return sorted(out, key=lambda r: -r["total_cp_lost"])[:top_k]

    def positions_for_opening(self, opening: str, min_count: int = 1) -> list[str]:
        return [t for t, d in self.out(opening_node(opening), rel="leads_to")
                if d.get("count", 0) >= min_count]

    def mistakes_in_opening(self, player: str, opening: str) -> list[dict]:
        """THE MULTI-HOP QUERY.

        player -> plays -> opening -> leads_to -> position <- blunders_at <- player

        This is the shape that motivates the whole graph: it intersects "where
        this player goes" with "where this player goes wrong", and neither set
        is known ahead of time.
        """
        pos_nodes = set(self.positions_for_opening(opening))
        if not pos_nodes:
            return []
        results = []
        for tgt, d in self.out(player_node(player), rel="blunders_at"):
            if tgt not in pos_nodes:
                continue
            node = self.g.nodes[tgt]
            themes = [self.g.nodes[t].get("name")
                      for t, _ in self.out(tgt, rel="exhibits")]
            results.append({
                "position": tgt,
                "fen": node.get("fen"),
                "ply": node.get("ply"),
                "phase": node.get("phase"),
                "count": d.get("count"),
                "avg_cp_loss": d.get("avg_cp_loss"),
                "examples": d.get("examples", []),
                "themes": themes,
            })
        return sorted(results, key=lambda r: -(r["avg_cp_loss"] * r["count"]))

    def save(self, path) -> None:
        nx.write_gexf(self.g, str(path))
