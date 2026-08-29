"""Tournament simulation.

Most of these are conservation checks. A bracket bug does not throw --
it quietly produces a competition with nine quarter-finalists, or lets a
club reach a semi-final it never qualified for.
"""

import numpy as np
import pandas as pd
import pytest

from pitchiq.models import dixon_coles as dc
from pitchiq.sim import tournament


class FakeModel:
    """Strength decreasing with index, so club 0 is the best."""

    def __init__(self, n=36, width=6):
        self.n, self.width = n, width

    def score_matrix(self, home, away):
        i, j = int(home.split("_")[1]), int(away.split("_")[1])
        # Both rates must stay positive across all 36 clubs.
        lam = 2.2 - 0.04 * i
        mu = 1.6 - 0.03 * j

        goals = np.arange(self.width)
        from scipy.stats import poisson

        grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        return grid / grid.sum()


def _fixtures(n=36):
    """Each club plays four home and four away, as the real draw does."""
    rows = []
    for i in range(n):
        for step in (1, 2, 3, 4):
            rows.append({"home": i, "away": (i + step) % n})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def simulation():
    keys = [f"club_{i}" for i in range(36)]
    return tournament.run(FakeModel(), keys, _fixtures(), runs=400, seed=0)


def test_every_run_produces_exactly_one_champion(simulation):
    assert simulation.reached["wins_it"].sum(axis=1).tolist() == [1] * simulation.runs


@pytest.mark.parametrize(
    "stage, places",
    [("last_16", 16), ("quarter_finals", 8), ("semi_finals", 4), ("final", 2)],
)
def test_each_round_has_the_right_number_of_clubs(simulation, stage, places):
    counts = simulation.reached[stage].sum(axis=1)

    assert (counts == places).all(), f"{stage} had {set(counts.tolist())}"


def test_positions_are_a_permutation(simulation):
    for run in range(simulation.runs):
        assert sorted(simulation.position[run]) == list(range(1, 37))


def test_reaching_a_later_round_implies_the_earlier_one(simulation):
    order = ["last_16", "quarter_finals", "semi_finals", "final", "wins_it"]

    for earlier, later in zip(order, order[1:]):
        assert not (simulation.reached[later] & ~simulation.reached[earlier]).any()


def test_top_eight_skip_the_playoffs(simulation):
    """Places 1-8 go straight to the last 16; 25-36 are out."""
    direct = simulation.position <= 8
    eliminated = simulation.position > 24

    assert not (direct & simulation.reached["playoffs"]).any()
    assert not (eliminated & simulation.reached["last_16"]).any()


def test_stronger_clubs_win_more_often(simulation):
    wins = simulation.reached["wins_it"].mean(axis=0)

    assert wins[:6].sum() > wins[-6:].sum()
    assert simulation.points[:, 0].mean() > simulation.points[:, -1].mean()


def test_points_conserved_across_the_league_phase(simulation):
    """144 matches award 3 points, less one for every draw."""
    per_season = simulation.points.sum(axis=1)

    assert (per_season <= 144 * 3).all()
    assert (per_season >= 144 * 2).all()


class TestRank:
    def _totals(self, **overrides):
        base = {
            name: np.zeros((1, 3))
            for name in ("points", "scored", "conceded", "away_scored", "wins", "away_wins")
        }
        base.update({k: np.array([v], dtype=float) for k, v in overrides.items()})
        return base

    def test_points_come_first(self):
        order = tournament.rank(self._totals(points=[3, 9, 6]))

        assert order[0].tolist() == [1, 2, 0]

    def test_goal_difference_breaks_level_points(self):
        order = tournament.rank(
            self._totals(points=[6, 6, 6], scored=[4, 9, 2], conceded=[4, 4, 4])
        )

        assert order[0].tolist() == [1, 0, 2]

    def test_goals_scored_breaks_level_difference(self):
        order = tournament.rank(
            self._totals(points=[6, 6, 6], scored=[5, 9, 2], conceded=[3, 7, 0])
        )

        # All +2; ranked by goals scored.
        assert order[0].tolist() == [1, 0, 2]


class TestTwoLeggedTies:
    def test_better_side_advances_more_often(self):
        keys = [f"club_{i}" for i in range(36)]
        grids = tournament.build_grids(FakeModel(), keys)
        rng = np.random.default_rng(0)

        strong = np.zeros(2000, dtype=int)
        weak = np.full(2000, 30)

        through = tournament._two_legged(grids, strong, weak, rng)

        assert (through == 0).mean() > 0.7

    def test_a_tie_always_produces_one_survivor(self):
        keys = [f"club_{i}" for i in range(36)]
        grids = tournament.build_grids(FakeModel(), keys)
        rng = np.random.default_rng(0)

        first = np.arange(10)
        second = np.arange(10, 20)

        through = tournament._two_legged(grids, first, second, rng)

        assert np.isin(through, np.concatenate([first, second])).all()
        assert ((through == first) | (through == second)).all()
