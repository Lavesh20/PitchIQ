"""Simulate a domestic league season, to test whether our ranges are honest.

The tournament simulator answers "how likely is Arsenal to reach the
quarter-final". This one answers a question we can actually check
thousands of times: "where will this club finish, and how sure are we?"

The reason to build it is sample size. UEFA replaced the group stage
with the 36-club league phase in 2024/25, so only two seasons have ever
used the format the tournament simulator knows — 72 clubs in total, far
too few to tell a well-calibrated forecast from an over-confident one.
Domestic leagues are the same shape of problem, and the record holds
548 complete double round-robin seasons: about 10,800 club-seasons.

If the machinery states honest ranges there, that is real evidence it
states honest ranges in Europe. Not proof — a domestic league is played
between clubs who meet twice a year and whose ratings are settled,
which is the easy case — but far better than 36 data points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tournament import build_grids, league_phase, rank


def _fixtures(season: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {key: i for i, key in enumerate(keys)}

    return (
        np.array([index[k] for k in season.home_key], dtype=int),
        np.array([index[k] for k in season.away_key], dtype=int),
    )


def positions(
    model,
    season: pd.DataFrame,
    runs: int = 2000,
    seed: int = 0,
) -> tuple[list[str], np.ndarray]:
    """Finishing position of every club, once per simulated season.

    Returns the club keys and a ``(runs, clubs)`` array of 1-based
    positions. The real fixture list is replayed rather than a generated
    one, so home and away are where they actually were.

    Ordering uses the UEFA tiebreakers from the tournament simulator:
    points, goal difference, goals scored, then away goals and wins.
    Most domestic leagues stop after goals scored and a few use
    head-to-head instead, which changes the order only for clubs level
    on every earlier criterion — a rounding error against the question
    being asked here.
    """
    keys = sorted(set(season.home_key) | set(season.away_key))
    home, away = _fixtures(season, keys)

    generator = np.random.default_rng(seed)
    grids = build_grids(model, keys)
    totals = league_phase(grids, home, away, runs, generator)

    order = rank(totals)

    # ``rank`` gives the club sitting at each position; invert it to the
    # position held by each club.
    placing = np.empty_like(order)
    np.put_along_axis(
        placing, order, np.arange(1, order.shape[1] + 1)[None, :].repeat(len(order), 0), axis=1
    )

    return keys, placing


def interval(placing: np.ndarray, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """Central interval of simulated finishing positions, per club.

    A 90% interval leaves 5% of simulations below and 5% above. The
    bounds are rounded outward — a half-position bound is not a league
    table — so if anything this understates over-confidence rather than
    inventing it.
    """
    tail = (1.0 - level) / 2.0

    low = np.floor(np.quantile(placing, tail, axis=0)).astype(int)
    high = np.ceil(np.quantile(placing, 1.0 - tail, axis=0)).astype(int)

    return low, high


def coverage(
    keys: list[str],
    placing: np.ndarray,
    actual: dict[str, int],
    levels=(0.5, 0.8, 0.9),
) -> pd.DataFrame:
    """Did each club finish inside its own interval?

    One row per club per level. An honest 90% interval should contain
    the truth for 90% of clubs; materially fewer means the simulator
    states ranges narrower than its knowledge supports, which is what
    puts too much probability on the favourites.
    """
    rows = []

    for level in levels:
        low, high = interval(placing, level)

        for i, key in enumerate(keys):
            if key not in actual:
                continue

            finish = actual[key]
            rows.append(
                {
                    "club": key,
                    "level": level,
                    "low": int(low[i]),
                    "high": int(high[i]),
                    "actual": finish,
                    "inside": bool(low[i] <= finish <= high[i]),
                    "expected": float(placing[:, i].mean()),
                }
            )

    return pd.DataFrame(rows)


def table(season: pd.DataFrame) -> dict[str, int]:
    """The season's real final table, as club key to position.

    Built with the same criteria the simulation is ranked by, so a
    difference between them is a difference in results and not in how
    the table was compiled.
    """
    keys = sorted(set(season.home_key) | set(season.away_key))
    home, away = _fixtures(season, keys)

    totals = {
        name: np.zeros((1, len(keys)))
        for name in ("points", "scored", "conceded", "away_scored", "wins", "away_wins")
    }

    for f, (h, a) in enumerate(zip(home, away)):
        goals_home = season.fthg.iloc[f]
        goals_away = season.ftag.iloc[f]

        home_won = goals_home > goals_away
        away_won = goals_away > goals_home
        drawn = goals_home == goals_away

        totals["points"][0, h] += 3 if home_won else (1 if drawn else 0)
        totals["points"][0, a] += 3 if away_won else (1 if drawn else 0)
        totals["scored"][0, h] += goals_home
        totals["scored"][0, a] += goals_away
        totals["conceded"][0, h] += goals_away
        totals["conceded"][0, a] += goals_home
        totals["away_scored"][0, a] += goals_away
        totals["wins"][0, h] += home_won
        totals["wins"][0, a] += away_won
        totals["away_wins"][0, a] += away_won

    order = rank(totals)[0]

    return {keys[club]: position + 1 for position, club in enumerate(order)}
