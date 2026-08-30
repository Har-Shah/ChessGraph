"""Graph retrieval by entity linking plus multi-hop traversal.

HOW IT WORKS
1. Entity linking. Parse the query against entities that actually exist in the
   graph: player names, opening names, opening families, tactical themes,
   colours, phases. Nothing is inferred that the graph cannot confirm.
2. Traversal. Each linked entity seeds a walk that collects candidate
   positions.
3. Intersection scoring. Positions reached by more than one independent path
   score highest. This is the whole point. A position that a player reaches in
   an opening they play AND blunders in AND that exhibits a theme they
   repeatedly lose to is exactly the answer to a preparation question, and it
   is found by joining, not by matching text.
4. Expansion. Positions map back to the shared corpus documents.

WHY THIS CAN BEAT TEXT SEARCH
A question like "which variation should I prepare against this opponent, and
what mistakes do they make there" names an opponent and an intent. It does not
name the opening, the position, or the theme. Those are what the answer is. A
lexical or dense retriever can only match words the query already contains, so
it cannot reach a document whose relevance comes from a relationship rather
than from shared vocabulary.

WHERE IT SHOULD LOSE
Queries that name a specific thing directly, for example "show me the Nxd5
blunder in the Alekhine", are pure lookup. Entity linking adds a failure mode
and nothing else. The evaluation is built to expose both cases.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from chessgraph.retrieval.base import Hit, RetrievalResult, tokenize
from chessgraph.retrieval.corpus import Document
from chessgraph.store.graph import (
    ChessKnowledgeGraph, player_node, opening_node, family_node, theme_node,
)

# Scoring weights. Absolute values do not matter, only their ratios. The
# intersection bonus is set well above any single path so that a position
# confirmed by two independent relationships always outranks one supported by a
# single strong edge.
W_OPENING_PATH = 1.0
W_BLUNDER_PATH = 1.5
W_THEME_PATH = 1.2
W_INTERSECTION = 3.0
W_FAMILY_FALLBACK = 0.6


@dataclass
class LinkedEntities:
    players: list[str] = field(default_factory=list)
    openings: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)

    def any_found(self) -> bool:
        return bool(self.players or self.openings or self.families
                    or self.themes or self.colors or self.phases)


class GraphRetriever:
    name = "graph"

    def __init__(self, graph: ChessKnowledgeGraph, subject: str | None = None):
        self.kg = graph
        self.subject = subject or graph.subject
        self.docs: list[Document] = []
        self.by_pos: dict[str, list[Document]] = defaultdict(list)
        self._entity_index: dict[str, list[tuple[str, str]]] = {}

    # ------------------------------------------------------------------ index
    def index(self, docs: list[Document]) -> None:
        self.docs = docs
        self.by_pos = defaultdict(list)
        for d in docs:
            self.by_pos[d.meta["pos_key"]].append(d)
        self._build_entity_index()

    def _build_entity_index(self) -> None:
        """Map lowercase surface forms to graph nodes for entity linking.

        Indexed on both the full name and its individual tokens. A query saying
        "Najdorf" has to reach "Sicilian Defense: Najdorf Variation" without the
        user typing the full Lichess name.
        """
        idx: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for node, data in self.kg.g.nodes(data=True):
            kind, name = data.get("kind"), data.get("name")
            if not name or kind not in ("player", "opening", "family", "theme"):
                continue
            surface = name.lower()
            idx[surface].append((kind, node))
            if kind in ("opening", "family", "theme"):
                for tok in tokenize(name):
                    # Skip generic words that would match nearly everything.
                    if tok in ("defense", "defence", "opening", "game", "attack",
                               "variation", "system", "the", "of", "and"):
                        continue
                    if len(tok) >= 4:
                        idx[tok].append((kind, node))
        self._entity_index = dict(idx)

    # --------------------------------------------------------- entity linking
    def link(self, query: str) -> LinkedEntities:
        q = query.lower()
        toks = set(tokenize(query))
        found = LinkedEntities()
        seen_nodes: set[str] = set()

        # Longest surface forms first so "Sicilian Defense: Najdorf Variation"
        # wins over the bare token "sicilian" when both are present.
        for surface in sorted(self._entity_index, key=len, reverse=True):
            hit = surface in q if " " in surface else surface in toks
            if not hit:
                continue
            for kind, node in self._entity_index[surface]:
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                name = self.kg.g.nodes[node].get("name")
                if kind == "player":
                    found.players.append(name)
                elif kind == "opening":
                    found.openings.append(name)
                elif kind == "family":
                    found.families.append(name)
                elif kind == "theme":
                    found.themes.append(name)

        for c in ("white", "black"):
            if c in toks:
                found.colors.append(c)
        for ph in ("opening", "middlegame", "endgame"):
            if ph in toks:
                found.phases.append(ph)

        # A question about "my" or "me" is about the subject player, who is
        # rarely named explicitly in a real question.
        if not found.players and self.subject:
            if toks & {"my", "me", "i", "mine"}:
                found.players.append(self.subject)
        return found

    # -------------------------------------------------------------- traversal
    def _positions_from_openings(self, names: list[str], is_family: bool
                                 ) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        weight = W_FAMILY_FALLBACK if is_family else W_OPENING_PATH
        for name in names:
            if is_family:
                fam = family_node(name)
                if fam not in self.kg.g:
                    continue
                # family <- in_family <- opening -> leads_to -> position
                openings = [src for src, _ in self.kg.inn(fam, rel="in_family")]
            else:
                openings = [opening_node(name)]
            for op in openings:
                if op not in self.kg.g:
                    continue
                for pos, d in self.kg.out(op, rel="leads_to"):
                    out[pos] += weight * min(d.get("count", 1), 5) / 5.0
        return out

    def _positions_from_players(self, names: list[str]) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for name in names:
            pn = player_node(name)
            if pn not in self.kg.g:
                continue
            for pos, d in self.kg.out(pn, rel="blunders_at"):
                severity = min(d.get("avg_cp_loss", 100), 600) / 600.0
                freq = min(d.get("count", 1), 5) / 5.0
                out[pos] += W_BLUNDER_PATH * (0.5 * freq + 0.5 * severity)
        return out

    def _positions_from_themes(self, names: list[str]) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for name in names:
            tn = theme_node(name)
            if tn not in self.kg.g:
                continue
            for pos, d in self.kg.inn(tn, rel="exhibits"):
                out[pos] += W_THEME_PATH * min(d.get("count", 1), 3) / 3.0
        return out

    def search(self, query: str, k: int = 10) -> RetrievalResult:
        t0 = time.perf_counter()
        ents = self.link(query)

        paths: dict[str, dict[str, float]] = {}
        if ents.openings:
            paths["opening"] = self._positions_from_openings(ents.openings, False)
        if ents.families:
            paths["family"] = self._positions_from_openings(ents.families, True)
        if ents.players:
            paths["player"] = self._positions_from_players(ents.players)
        if ents.themes:
            paths["theme"] = self._positions_from_themes(ents.themes)

        if not paths:
            # No entity linked. Returning nothing is the honest outcome and the
            # evaluation should see it, rather than silently falling back to a
            # text search and reporting the result as a graph win.
            return RetrievalResult(query, self.name, [],
                                   (time.perf_counter() - t0) * 1000)

        combined: dict[str, float] = defaultdict(float)
        support: dict[str, list[str]] = defaultdict(list)
        for path_name, scores in paths.items():
            for pos, s in scores.items():
                combined[pos] += s
                support[pos].append(path_name)

        # The join bonus. Independent paths only, so opening and family
        # agreeing is not treated as two pieces of evidence.
        for pos, srcs in support.items():
            independent = {s for s in srcs}
            if "family" in independent and "opening" in independent:
                independent.discard("family")
            if len(independent) >= 2:
                combined[pos] += W_INTERSECTION * (len(independent) - 1)

        ranked_positions = sorted(combined.items(), key=lambda kv: -kv[1])

        # Expand positions to documents, filtering for consistency with the
        # linked entities so a position shared by two players does not return
        # the wrong player's mistakes.
        want_players = {p.lower() for p in ents.players}
        want_themes = set(ents.themes)
        want_colors = set(ents.colors)
        hits: list[Hit] = []
        for pos_node, score in ranked_positions:
            pos_key = pos_node.split(":", 1)[1]
            for doc in self.by_pos.get(pos_key, []):
                if want_players and (doc.meta.get("player") or "").lower() not in want_players:
                    continue
                if want_colors and doc.meta.get("color") not in want_colors:
                    continue
                doc_score = score
                if want_themes and want_themes & set(doc.meta.get("themes", [])):
                    doc_score += 0.5
                hits.append(Hit(doc_id=doc.doc_id, score=round(doc_score, 4),
                                rank=0, doc=doc,
                                explanation=f"paths: {'+'.join(support[pos_node])}"))
            if len(hits) >= k * 3:
                break

        hits.sort(key=lambda h: -h.score)
        hits = hits[:k]
        for i, h in enumerate(hits):
            h.rank = i + 1
        return RetrievalResult(query, self.name, hits,
                               (time.perf_counter() - t0) * 1000)
