"""Fetch a single player's games from the Lichess API.

Why the API and not the Open Database dump?
    The monthly dumps are ~30GB compressed and contain *every* game played on
    the site. To get one player's games you would stream-decompress the whole
    file and throw away 99.999% of it. The per-user export endpoint gives us
    exactly the games we want in seconds. `database.py` covers the dump case
    for when we need population-level statistics later.

Rate limits: anonymous requests are throttled to roughly 20 games/second and
will return HTTP 429 if you hammer them. We stream, we set a real User-Agent,
and we back off on 429. Be a good citizen. This is a free public service.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import requests

from chessgraph.config import RAW, INGEST

API = "https://lichess.org/api"
USER_AGENT = "ChessGraph/0.1 (educational research project)"


class LichessError(RuntimeError):
    pass


def _get(url: str, *, params: dict | None = None, stream: bool = False,
         accept: str = "application/x-chess-pgn", retries: int = 3):
    """One HTTP call with backoff on rate limiting."""
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    for attempt in range(retries):
        resp = requests.get(url, params=params, headers=headers,
                            stream=stream, timeout=60)
        if resp.status_code == 429:
            # Lichess asks you to wait a full minute after a 429.
            wait = 60
            print(f"  rate limited, waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            # Lichess sometimes routes a throttled API request to the HTML
            # 404 page rather than returning a clean 429, so an HTML body on
            # a 404 for a user we believe exists means "slow down", not
            # "no such user".
            if "text/html" in resp.headers.get("Content-Type", ""):
                raise LichessError(
                    f"Got an HTML 404 from {url}. This usually means you are "
                    "being rate limited, not that the user is missing. "
                    "Wait 60s and retry."
                )
            raise LichessError(f"Not found: {url}")
        resp.raise_for_status()
        return resp
    raise LichessError(f"Gave up after {retries} attempts: {url}")


def fetch_player_profile(username: str) -> dict:
    """Ratings, game counts, account flags. Cheap, one small JSON."""
    resp = _get(f"{API}/user/{username}", accept="application/json")
    return resp.json()


def stream_player_pgn(
    username: str,
    *,
    max_games: int = INGEST.max_games,
    perf_types: tuple[str, ...] = INGEST.perf_types,
    rated_only: bool = INGEST.rated_only,
    since: int | None = None,
    until: int | None = None,
    color: str | None = None,
) -> Iterator[str]:
    """Yield raw PGN text chunks for one player, newest game first.

    The API streams, so we never hold the whole export in memory. Query params
    that matter:
      opening=true  -> adds [Opening] and [ECO] tags. Without this we would
                       have to classify openings ourselves from the moves.
      clocks=true   -> per-move clock times, which let us later separate
                       "blunder because they misunderstood the position" from
                       "blunder because they had 8 seconds left".
      evals=true    -> Lichess's own server-side analysis, when it exists.
                       Useful as a cross-check on our Stockfish numbers.
    """
    params = {
        "max": max_games,
        "rated": str(rated_only).lower(),
        "perfType": ",".join(perf_types),
        "opening": "true",
        "clocks": "true",
        "evals": "true",
        "moves": "true",
        "tags": "true",
    }
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    if color:
        params["color"] = color

    resp = _get(f"{API}/games/user/{username}", params=params, stream=True)
    # Lichess serves application/x-chess-pgn with no charset, so requests will
    # not guess an encoding and decode_unicode would hand back raw bytes.
    resp.encoding = "utf-8"
    for chunk in resp.iter_content(chunk_size=8192, decode_unicode=True):
        if chunk:
            yield chunk


def download_player_games(
    username: str,
    *,
    max_games: int = INGEST.max_games,
    force: bool = False,
    **kwargs,
) -> Path:
    """Download to data/raw/<username>.pgn and return the path.

    Cached by default: re-running an experiment should not re-hit the network.
    Delete the file or pass force=True to refresh.
    """
    out = RAW / f"{username.lower()}.pgn"
    if out.exists() and not force:
        size_kb = out.stat().st_size / 1024
        print(f"  cached: {out.name} ({size_kb:.0f} KB), pass force=True to refresh")
        return out

    print(f"  downloading up to {max_games} games for {username}...")
    written = 0
    games_seen = 0
    # Client-side cap. The server's `max` parameter is advisory in practice , 
    # an observed request for 25 games came back with 36, and an experiment
    # whose corpus size depends on server behaviour is not reproducible. We
    # count [Event tags as they stream past and stop ourselves. This also caps
    # bandwidth rather than downloading-then-discarding.
    with out.open("w", encoding="utf-8") as fh:
        buffer = ""
        for chunk in stream_player_pgn(username, max_games=max_games, **kwargs):
            buffer += chunk
            games_seen += chunk.count("[Event ")
            fh.write(chunk)
            written += len(chunk)
            if games_seen > max_games:
                break
    # Trim any game past the cap so the file holds exactly max_games.
    text = out.read_text(encoding="utf-8")
    parts = text.split("[Event ")
    if len(parts) - 1 > max_games:
        text = "[Event ".join(parts[: max_games + 1])
        out.write_text(text, encoding="utf-8")
    final = text.count("[Event ")
    print(f"  wrote {len(text) / 1024:.0f} KB to {out} ({final} games)")
    return out
