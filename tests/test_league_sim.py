"""Tests for the league-season simulator.

The measurement it feeds — do 90% intervals contain the truth 90% of
the time — is only as trustworthy as the position bookkeeping beneath
it. An off-by-one in the ranking, or an interval computed on the wrong
axis, would produce a coverage figure that looks entirely plausible and
is meaningless. So these pin the arithmetic against cases where the
answer is known by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchiq.sim import league


def round_robin(clubs: list[str], results: dict | None = None) -> pd.DataFrame:
    """Every club hosts every other once."""
    rows = []
    results = results or {}

    for home in clubs:
        for away in clubs:
            if home == away:
                continue

            goals = results.get((home, away), (0, 0))
            rows.append(
                {
                    "home_key": home,
                    "away_key": away,
                    "fthg": goals[0],
                    "ftag": goals[1],
                    "date": pd.Timestamp("2020-01-01"),
                }
            )

    return pd.DataFrame(rows)


class FakeModel:
    """A model where club strength is read straight off the name.

    ``a`` is strongest. Goal rates are chosen to be comfortably inside
    the grid so the simulation cannot be distorted by truncation.
    """

    def __init__(self, keys: list[str]):
        self.rank = {key: i for i, key in enumerate(sorted(keys))}
        self.attack = dict.fromkeys(keys, 0.0)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        from scipy.stats import poisson

        edge = (self.rank[away] - self.rank[home]) * 0.25
        lam = np.exp(0.25 + edge)
        mu = np.exp(0.25 - edge)

        goals = np.arange(8)
        grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))

        return grid / grid.sum()


def test_table_ranks_by_points_then_goal_difference():
    clubs = ["a", "b", "c"]
    # a beats both, b beats c, so a first, b second, c third.
    season = round_robin(
        clubs,
        {
            ("a", "b"): (2, 0), ("a", "c"): (2, 0),
            ("b", "a"): (0, 1), ("b", "c"): (3, 0),
            ("c", "a"): (0, 1), ("c", "b"): (0, 1),
        },
    )

    assert league.table(season) == {"a": 1, "b": 2, "c": 3}


def test_table_covers_every_club_exactly_once():
    clubs = list("abcdef")
    season = round_robin(clubs)

    positions = league.table(season)

    assert sorted(positions.values()) == list(range(1, len(clubs) + 1))


def test_every_simulated_season_is_a_valid_table():
    clubs = list("abcde")
    season = round_robin(clubs)

    keys, placing = league.positions(FakeModel(clubs), season, runs=50, seed=0)

    assert placing.shape == (50, len(clubs))
    # Each run must be a permutation of 1..n: no club may share a place
    # with another, and none may be missing.
    for run in placing:
        assert sorted(run) == list(range(1, len(clubs) + 1))


def test_the_stronger_club_finishes_higher_on_average():
    clubs = list("abcde")
    season = round_robin(clubs)

    keys, placing = league.positions(FakeModel(clubs), season, runs=400, seed=1)
    average = placing.mean(axis=0)

    # ``a`` is strongest by construction, ``e`` weakest.
    assert average[keys.index("a")] < average[keys.index("e")]
    assert list(np.argsort(average)) == [keys.index(k) for k in "abcde"]


def test_wider_levels_give_wider_intervals():
    clubs = list("abcdef")
    season = round_robin(clubs)

    _, placing = league.positions(FakeModel(clubs), season, runs=400, seed=2)

    narrow = league.interval(placing, 0.5)
    wide = league.interval(placing, 0.9)

    assert ((wide[1] - wide[0]) >= (narrow[1] - narrow[0])).all()


def test_intervals_stay_inside_the_table():
    clubs = list("abcdef")
    season = round_robin(clubs)

    _, placing = league.positions(FakeModel(clubs), season, runs=400, seed=3)
    low, high = league.interval(placing, 0.9)

    assert (low >= 1).all()
    assert (high <= len(clubs)).all()
    assert (low <= high).all()


def test_coverage_marks_inside_and_outside_correctly():
    clubs = list("abcd")
    season = round_robin(clubs)

    keys, placing = league.positions(FakeModel(clubs), season, runs=200, seed=4)
    low, high = league.interval(placing, 0.9)

    # Claim every club finished exactly where its lower bound sits, then
    # again one place outside it.
    inside = league.coverage(
        keys, placing, {k: int(low[i]) for i, k in enumerate(keys)}, (0.9,)
    )
    assert inside.inside.all()

    outside = league.coverage(
        keys, placing, {k: len(clubs) + 5 for k in keys}, (0.9,)
    )
    assert not outside.inside.any()


def test_coverage_is_honest_when_the_truth_comes_from_the_simulator():
    """The end-to-end check on the measurement itself.

    Draw the "real" table from the simulator's own output. A 90%
    interval must then contain it about 90% of the time. If this drifts,
    the coverage figures in the backtest cannot be trusted either.
    """
    clubs = list("abcdefgh")
    season = round_robin(clubs)

    keys, placing = league.positions(FakeModel(clubs), season, runs=1500, seed=5)

    generator = np.random.default_rng(7)
    hits = []

    for level in (0.5, 0.9):
        low, high = league.interval(placing, level)

        for _ in range(150):
            drawn = placing[generator.integers(0, len(placing))]
            actual = {k: int(drawn[i]) for i, k in enumerate(keys)}
            result = league.coverage(keys, placing, actual, (level,))
            hits.append((level, result.inside.mean()))

        observed = np.mean([h for lv, h in hits if lv == level])
        # Rounding the bounds outward can only help coverage, so the
        # floor is the stated level and the ceiling allows for that.
        assert level - 0.05 <= observed <= min(1.0, level + 0.20)


def test_unknown_clubs_are_reported_not_hidden():
    clubs = list("abc")
    season = round_robin(clubs)

    keys, placing = league.positions(FakeModel(clubs), season, runs=50, seed=6)
    result = league.coverage(keys, placing, {"a": 1, "b": 2}, (0.9,))

    # "c" has no recorded finish, so it contributes no row rather than a
    # silently wrong one.
    assert set(result.club) == {"a", "b"}
