"""Interactive preparation agent built on the Anthropic tool runner.

The tools in `tools.py` are plain functions and stay usable without a model.
This module is the only place that imports `anthropic`, so the rest of the
project runs with no API key.

WHY THE TOOL RUNNER RATHER THAN A HAND-WRITTEN LOOP
`client.beta.messages.tool_runner` drives the request, execute, feed-back cycle
for us. The tools here are pure lookups against a local SQLite file with no side
effects, so there is nothing to gate or approve per turn, which is the main
reason to write the loop by hand. Schemas are generated from the type hints and
docstrings, so a tool's signature is its contract and the two cannot drift.

TOOL DESIGN NOTES
Every tool description states its bound explicitly. Models will request large
limits when a question feels broad, and a description that says "at most 10"
is more effective than silently clamping, because the model then plans around
the limit rather than expecting data it will not receive.
"""
from __future__ import annotations

import os

from chessgraph.agent.tools import ChessGraphTools

SYSTEM_PROMPT = """You are a chess preparation assistant with direct access to \
one player's analysed game history.

Ground every claim in tool output. When you state that a player has a weakness, \
cite the specific games and move numbers the tools returned. Never invent a \
game, a move, an evaluation, or a URL. If the tools do not support a claim, say \
so plainly rather than filling the gap.

Centipawn loss is how much a move threw away, measured by Stockfish. 100cp is \
a pawn. At club level a single move losing 300cp usually decides the game.

Be concrete and specific. "You lose material to knight forks in the Sicilian, \
17 times in 42 games, most often around move 15" is useful. "You should work on \
tactics" is not. Prefer the smallest number of highest-value recommendations \
over a long list."""


def build_tools(tools: ChessGraphTools) -> list:
    """Wrap the bound methods as decorated tools for the runner.

    The decorator reads the signature and docstring of each function, so the
    wrappers below are where the model-facing contract is written. The bounds
    are stated in the descriptions on purpose.
    """
    from anthropic import beta_tool

    @beta_tool
    def fetch_player_games(player: str = "", limit: int = 10,
                           color: str = "", opening: str = "") -> dict:
        """Look up a player's games with results, openings and dates.

        Args:
            player: Username. Defaults to the player being studied.
            color: Filter to "white" or "black". Empty means both.
            opening: Substring match on the opening name, e.g. "Sicilian".
            limit: Maximum games to return. At most 25.
        """
        return tools.fetch_player_games(player=player, limit=limit,
                                        color=color, opening=opening)

    @beta_tool
    def analyze_position(fen: str, depth: int = 16, multipv: int = 3) -> dict:
        """Run Stockfish on a position and return the top candidate moves.

        Args:
            fen: Position in Forsyth-Edwards Notation.
            depth: Search depth. At most 22. Higher is slower.
            multipv: How many candidate moves to return. At most 5.
        """
        return tools.analyze_position(fen=fen, depth=depth, multipv=multipv)

    @beta_tool
    def retrieve_similar_games(question: str, limit: int = 10) -> dict:
        """Search the analysed mistake corpus with a natural language question.

        Returns specific mistakes with game URLs, move numbers, what was played,
        what was better, and the tactical themes involved.

        Args:
            question: What to look for, e.g. "back rank problems as white".
            limit: Maximum results. At most 10.
        """
        return tools.retrieve_similar_games(question=question, limit=limit)

    @beta_tool
    def find_opening_weaknesses(player: str = "", color: str = "",
                                min_games: int = 4, limit: int = 10) -> dict:
        """Rank a player's openings by average centipawn loss.

        Args:
            player: Username. Defaults to the player being studied.
            color: "white" or "black". Empty means both.
            min_games: Ignore openings with fewer games than this.
            limit: Maximum openings to return. At most 10.
        """
        return tools.find_opening_weaknesses(player=player, color=color,
                                             min_games=min_games, limit=limit)

    @beta_tool
    def find_recurring_themes(player: str = "", limit: int = 10) -> dict:
        """Rank a player's tactical weaknesses by total centipawns lost.

        Args:
            player: Username. Defaults to the player being studied.
            limit: Maximum themes to return. At most 10.
        """
        return tools.find_recurring_themes(player=player, limit=limit)

    @beta_tool
    def generate_training_positions(count: int = 8, theme: str = "") -> dict:
        """Produce drillable positions from the player's own losses.

        Only positions that were still competitive before the mistake are
        returned, so each one is a genuine missed opportunity.

        Args:
            count: How many positions. At most 20.
            theme: Restrict to one theme, e.g. "fork". Empty means any.
        """
        return tools.generate_training_positions(count=count, theme=theme)

    @beta_tool
    def retrieve_similar_positions(fen: str = "", limit: int = 5,
                                   cross_opening_only: bool = False) -> dict:
        """Find positions structurally like this one in the player's history.

        Matches on pawn structure, material and king placement rather than on
        names, so it finds transpositions between differently named openings.

        Args:
            fen: The position to match, in Forsyth-Edwards Notation.
            limit: Maximum results. At most 10.
            cross_opening_only: Only return matches from a different opening
                family. Use this to surface transpositions.
        """
        return tools.retrieve_similar_positions(
            fen=fen, limit=limit, cross_opening_only=cross_opening_only)

    @beta_tool
    def build_opponent_report(opponent: str, limit: int = 5) -> dict:
        """Prep sheet for one opponent: repertoire, results and exploitable errors.

        Args:
            opponent: The opponent's username.
            limit: How many openings, themes and mistakes to include. At most 10.
        """
        return tools.build_opponent_report(opponent=opponent, limit=limit)

    return [fetch_player_games, analyze_position, retrieve_similar_games,
            retrieve_similar_positions, find_opening_weaknesses,
            find_recurring_themes, generate_training_positions,
            build_opponent_report]


def ask(subject: str, question: str, *, model: str = "claude-opus-5",
        max_tokens: int = 16000, verbose: bool = True) -> str:
    """Answer one question about a player, using the tools to ground it."""
    import anthropic

    if not (os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY, or run "
            "`ant auth login`. The rest of ChessGraph works without this."
        )

    cg = ChessGraphTools(subject)
    try:
        client = anthropic.Anthropic()
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            tools=build_tools(cg),
            messages=[{
                "role": "user",
                "content": f"The player being studied is {subject}.\n\n{question}",
            }],
        )
        final_text = ""
        for message in runner:
            for block in message.content:
                if block.type == "tool_use" and verbose:
                    print(f"  [tool] {block.name}({block.input})")
                elif block.type == "text":
                    final_text = block.text
        return final_text
    finally:
        cg.close()
