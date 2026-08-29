"""Combine the Elo and Dixon-Coles views into one predictor.

The two models fail differently, and which one wins depends entirely on
how often it is allowed to refit.

Elo is sequential. Updated after every match it tracks form closely and
scores 0.2052 RPS on held-out European matches; frozen at a cutoff and
asked to forecast the next three years it decays to 0.2191, because a
single rating per club goes stale. It also only knows who wins, and a
dominant club in a closed league hoards that league's points, which is
what the league-strength correction repairs.

Dixon-Coles is a joint fit over every match at once. That links leagues
through shared opponents far better than sequential Elo -- Celtic lands
44th in Europe rather than 12th, with no explicit correction -- and it
produces scorelines. It is also far more robust to going stale: 0.2075
refitted quarterly against 0.2092 frozen.

Which matters depends on the question. Forecasting a whole league phase
before a ball is kicked is the frozen case, and there Dixon-Coles is
clearly better. Updating predictions as a season unfolds is the rolling
case, and there Elo is slightly ahead. Blending helps in the rolling
case and does almost nothing in the frozen one, so ``weight`` should be
chosen on a validation window rather than assumed.

For simulation the outcome probabilities alone are not enough: the
league phase is decided on goal difference, so scorelines are needed,
and only Dixon-Coles produces them. :meth:`Predictor.score_matrix`
therefore keeps the Dixon-Coles grid but rescales its home / draw / away
blocks so their totals match the blended probabilities. The shape within
each block -- whether a home win is more likely 2-0 or 3-1 -- is
Dixon-Coles; the split between the three outcomes is the ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..eval.metrics import OUTCOMES, ranked_probability_score
from . import dixon_coles as dc
from . import elo as elo_model
from . import league_strength
from .outcome import OutcomeModel


@dataclass
class Predictor:
    """Blended Elo and Dixon-Coles forecasts."""

    elo: elo_model.EloResult
    strength: league_strength.LeagueStrength
    outcome: OutcomeModel
    goals: dc.DixonColesResult
    weight: float
    countries: dict[str, str]

    def _elo_probabilities(self, home: str, away: str) -> np.ndarray:
        rating_home = self.strength.adjust(
            np.array([self.elo.rating(home)]), [self.countries.get(home)]
        )
        rating_away = self.strength.adjust(
            np.array([self.elo.rating(away)]), [self.countries.get(away)]
        )
        difference = rating_home + self.strength.home_advantage - rating_away

        return self.outcome.predict(difference)[0]

    def outcome_probabilities(self, home: str, away: str) -> dict[str, float]:
        """Blended home / draw / away probabilities."""
        from_elo = self._elo_probabilities(home, away)
        from_goals = np.array([self.goals.predict(home, away)[o] for o in OUTCOMES])

        blended = self.weight * from_elo + (1.0 - self.weight) * from_goals

        return dict(zip(OUTCOMES, blended))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """Scoreline probabilities agreeing with the blended outcome split.

        Dixon-Coles supplies the shape; the ensemble supplies how much
        weight each of the three outcomes carries.
        """
        grid = self.goals.score_matrix(home, away)
        target = self.outcome_probabilities(home, away)

        blocks = {
            "H": np.tril(np.ones_like(grid), -1),
            "D": np.eye(len(grid)),
            "A": np.triu(np.ones_like(grid), 1),
        }

        out = np.zeros_like(grid)
        for name, mask in blocks.items():
            current = (grid * mask).sum()

            if current > 0:
                out += grid * mask * (target[name] / current)

        return out / out.sum()

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Mean goals for each side under the calibrated grid."""
        grid = self.score_matrix(home, away)
        goals = np.arange(len(grid))

        return float((grid.sum(axis=1) * goals).sum()), float(
            (grid.sum(axis=0) * goals).sum()
        )


def choose_weight(
    elo_probabilities: np.ndarray,
    goal_probabilities: np.ndarray,
    actual,
    step: float = 0.05,
) -> float:
    """Pick the blend weight that scores best on a validation set."""
    grid = np.arange(0.0, 1.0 + step, step)

    scores = [
        ranked_probability_score(w * elo_probabilities + (1 - w) * goal_probabilities, actual)
        for w in grid
    ]

    return float(grid[int(np.argmin(scores))])


def build(
    matches: pd.DataFrame,
    cutoff: pd.Timestamp,
    weight: float,
    config: dc.DixonColesConfig | None = None,
) -> Predictor:
    """Fit every component on matches before ``cutoff``."""
    history = elo_model.fit(matches).history
    before = history[history.date < cutoff]

    strength = league_strength.fit(
        before[before.kind == "uefa"], league_strength.league_means(before)
    )
    adjusted = league_strength.apply(before, strength)
    domestic = adjusted[adjusted.kind == "domestic"]

    countries: dict[str, str] = {}
    for club, country in zip(matches.home_key, matches.home_cc):
        countries[club] = country
    for club, country in zip(matches.away_key, matches.away_cc):
        countries.setdefault(club, country)

    return Predictor(
        elo=elo_model.fit(matches[matches.date < cutoff]),
        strength=strength,
        outcome=OutcomeModel.fit(domestic.elo_diff_adj, domestic.ftr),
        goals=dc.fit(
            matches[matches.date < cutoff],
            config or dc.DixonColesConfig(xi=0.0010, ridge=0.5, home_advantage_by=None),
            reference=cutoff,
        ),
        weight=weight,
        countries=countries,
    )
