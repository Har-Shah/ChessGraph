"""Held-out temporal evaluation.

Two questions, both of which the system has to answer to be worth anything:

1. OPENING PREDICTION. Given the player and colour, can we predict what they
   will open with in games we have not seen?

2. WEAKNESS PERSISTENCE. This is the important one. A weakness profile built on
   past games is only useful if those weaknesses actually recur. If the themes
   ranked highest in the training window do not rank highly in the future
   window, the profile is describing noise and every training plan built on it
   is worthless.

THE SPLIT IS TEMPORAL, NEVER RANDOM
Train on older games, test on newer. A random split leaks the future into the
past: the same opponent, the same repertoire phase, sometimes the same day's
session end up on both sides, and every metric inflates. The only honest
question is whether the past predicts the future.

BASELINES MATTER MORE THAN SCORES
A weakness profile that does not beat the population base rate is not
personalised, it is just describing what every player at this level does. Both
evaluations below report a baseline that ignores the player, and the number to
look at is the gap, not the raw accuracy.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from chessgraph.store.db import Store


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without a scipy dependency. Ties get average ranks."""
    n = len(a)
    if n < 2:
        return float("nan")

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


@dataclass
class Split:
    train_ids: set[str]
    test_ids: set[str]
    cutoff_date: str


def temporal_split(store: Store, subject: str, test_frac: float = 0.25) -> Split:
    rows = store.q(
        "SELECT game_id, date FROM games WHERE subject = ? ORDER BY date ASC",
        (subject,))
    if not rows:
        return Split(set(), set(), "")
    n_test = max(1, int(len(rows) * test_frac))
    train = rows[:-n_test]
    test = rows[-n_test:]
    return Split({r["game_id"] for r in train},
                 {r["game_id"] for r in test},
                 test[0]["date"] if test else "")


# ------------------------------------------------------- opening prediction
def evaluate_opening_prediction(store: Store, subject: str,
                                split: Split, level: str = "family") -> dict:
    """Predict the opening of each held-out game from training frequencies.

    `level` is 'family', 'opening' or 'eco'. Family is the level a preparation
    plan actually operates at, since a specific Lichess variation name is often
    too granular to have any training data.
    """
    col = {"family": "opening", "opening": "opening", "eco": "eco"}[level]
    rows = store.q(
        f"SELECT game_id, subject_color, {col} AS label FROM games WHERE subject = ?",
        (subject,))

    def norm(label):
        if not label:
            return None
        return label.split(":")[0].strip() if level == "family" else label

    train_by_color: dict[str, Counter] = defaultdict(Counter)
    train_all = Counter()
    test_rows = []
    for r in rows:
        label = norm(r["label"])
        if not label:
            continue
        if r["game_id"] in split.train_ids:
            train_by_color[r["subject_color"]][label] += 1
            train_all[label] += 1
        elif r["game_id"] in split.test_ids:
            test_rows.append((r["subject_color"], label))

    if not test_rows or not train_all:
        return {"level": level, "n_test": len(test_rows), "error": "insufficient data"}

    # Baseline ignores colour and just predicts the player's single most common
    # opening. Colour-conditioned prediction has to beat it to justify itself.
    baseline_pred = train_all.most_common(1)[0][0]

    top1 = top3 = base_top1 = 0
    for color, truth in test_rows:
        ranked = [lbl for lbl, _ in train_by_color[color].most_common(3)]
        if ranked and ranked[0] == truth:
            top1 += 1
        if truth in ranked:
            top3 += 1
        if truth == baseline_pred:
            base_top1 += 1

    n = len(test_rows)
    coverage = sum(1 for _, t in test_rows if t in train_all) / n
    return {
        "level": level,
        "n_train_games": sum(train_all.values()),
        "n_test": n,
        "top1_accuracy": round(top1 / n, 4),
        "top3_accuracy": round(top3 / n, 4),
        "baseline_top1_ignores_color": round(base_top1 / n, 4),
        "lift_over_baseline": round((top1 - base_top1) / n, 4),
        "test_label_coverage": round(coverage, 4),
        "distinct_train_labels": len(train_all),
    }


# ------------------------------------------------------ weakness persistence
def evaluate_weakness_persistence(store: Store, subject: str, split: Split,
                                  top_k: int = 5) -> dict:
    """Do the weaknesses found in the training window recur in the test window?"""
    rows = store.q(
        """
        SELECT m.game_id, t.theme, m.cp_loss, m.is_subject_move
        FROM move_themes t
        JOIN moves m ON m.game_id = t.game_id AND m.ply = t.ply
        JOIN games g ON g.game_id = m.game_id
        WHERE g.subject = ? AND m.is_subject_move = 1
        """, (subject,))

    train_count, test_count = Counter(), Counter()
    train_cp, test_cp = Counter(), Counter()
    train_moves = test_moves = 0
    for r in rows:
        if r["game_id"] in split.train_ids:
            train_count[r["theme"]] += 1
            train_cp[r["theme"]] += r["cp_loss"] or 0
            train_moves += 1
        elif r["game_id"] in split.test_ids:
            test_count[r["theme"]] += 1
            test_cp[r["theme"]] += r["cp_loss"] or 0
            test_moves += 1

    if not train_count or not test_count:
        return {"error": "insufficient labelled mistakes in one window"}

    # Compare RATES, not counts. The test window has fewer games, so raw counts
    # would always be lower and the comparison would be meaningless.
    themes = sorted(set(train_count) | set(test_count))
    train_rate = [train_count[t] / max(train_moves, 1) for t in themes]
    test_rate = [test_count[t] / max(test_moves, 1) for t in themes]

    rho = spearman(train_rate, test_rate)

    train_top = [t for t, _ in Counter(
        {t: train_cp[t] for t in themes}).most_common(top_k)]
    test_top = [t for t, _ in Counter(
        {t: test_cp[t] for t in themes}).most_common(top_k)]
    overlap = len(set(train_top) & set(test_top)) / max(len(train_top), 1)

    # Baseline: how well does simply predicting the most common theme overall
    # do? If personalised ranking does not beat this, it is not personalised.
    return {
        "n_train_mistakes": train_moves,
        "n_test_mistakes": test_moves,
        "n_themes": len(themes),
        "spearman_rate_correlation": round(rho, 4) if not math.isnan(rho) else None,
        f"top{top_k}_overlap": round(overlap, 4),
        "train_top_themes": train_top,
        "test_top_themes": test_top,
        "train_rates": {t: round(train_count[t] / max(train_moves, 1), 4)
                        for t in train_top},
        "test_rates": {t: round(test_count[t] / max(test_moves, 1), 4)
                       for t in train_top},
    }


def evaluate_opening_weakness_persistence(store: Store, subject: str,
                                          split: Split, min_games: int = 4) -> dict:
    """Does per-opening ACPL measured on training games predict test ACPL?

    A stronger claim than theme persistence: it says we can tell you which of
    your openings you actually play worse, and be right about the next ones.
    """
    rows = store.q(
        """
        SELECT g.game_id, g.opening, m.cp_loss
        FROM moves m JOIN games g ON g.game_id = m.game_id
        WHERE g.subject = ? AND m.is_subject_move = 1
          AND m.cp_loss IS NOT NULL AND m.judgment != 'book'
        """, (subject,))

    train_sum, train_n = Counter(), Counter()
    test_sum, test_n = Counter(), Counter()
    train_games, test_games = defaultdict(set), defaultdict(set)
    for r in rows:
        fam = (r["opening"] or "").split(":")[0].strip()
        if not fam:
            continue
        if r["game_id"] in split.train_ids:
            train_sum[fam] += r["cp_loss"]; train_n[fam] += 1
            train_games[fam].add(r["game_id"])
        elif r["game_id"] in split.test_ids:
            test_sum[fam] += r["cp_loss"]; test_n[fam] += 1
            test_games[fam].add(r["game_id"])

    shared = [f for f in train_n
              if f in test_n
              and len(train_games[f]) >= min_games and len(test_games[f]) >= 2]
    if len(shared) < 3:
        return {"error": f"only {len(shared)} openings had enough games in both windows",
                "n_shared": len(shared)}

    tr = [train_sum[f] / train_n[f] for f in shared]
    te = [test_sum[f] / test_n[f] for f in shared]
    return {
        "n_openings_compared": len(shared),
        "spearman_acpl_correlation": round(spearman(tr, te), 4),
        "train_acpl": {f: round(train_sum[f] / train_n[f], 1) for f in shared},
        "test_acpl": {f: round(test_sum[f] / test_n[f], 1) for f in shared},
    }
