"""Elo ratings over every match, domestic and continental.

One rating per club, updated match by match, points taken from the loser
and given to the winner. Because a club carries the same rating into
every competition, a Norwegian side beating an Italian one moves points
straight from Serie A into the Eliteserien. That is what makes Elo a
useful cross-league bridge: league strength emerges from continental
results rather than needing a parameter of its own.

Ratings are recorded *before* each match is applied, so the history this
produces can be backtested without leaking the result being predicted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Ratings for a club entering European competition from a country we
# have no domestic data for, keyed by the tier of the competition it
# first appears in. A side entering through Conference League qualifying
# is a weaker proposition than one entering the Champions League proper,
# and starting both at the global average overrates the former badly.
ENTRY_BY_TIER = {1: 1550.0, 2: 1450.0, 3: 1350.0, 4: 1250.0}


@dataclass
class EloConfig:
    """Parameters of the rating update."""

    # Step size. ClubElo uses 20 for club football; larger reacts faster
    # and is noisier.
    k: float = 20.0

    # Home advantage, in rating points, added to the home side before
    # computing the expectation.
    home_advantage: float = 65.0

    # Where a club starts when we know nothing else about it.
    initial: float = 1500.0

    # A club appearing for the first time in a league we have already
    # seen starts at that league's mean less this much: newly promoted
    # and newly qualified sides are typically below their new peers.
    newcomer_penalty: float = 50.0

    # Fraction of the gap to the global mean given back at each season
    # boundary. ClubElo does not regress at all; squads do turn over,
    # so this is exposed but left off by default.
    season_regression: float = 0.0

    # Per-competition multipliers on k, keyed by competition code. A
    # continental tie can be made to move ratings further than a
    # fourth-tier league match. Left flat by default.
    k_multiplier: dict[str, float] = field(default_factory=dict)


def goal_multiplier(goal_difference: int) -> float:
    """Scale the update by margin of victory.

    The World Football Elo convention: a two-goal win counts half again
    as much as a one-goal win, and the increment tapers after that so a
    rout does not dominate. A draw keeps the full weight, which matters
    because a draw between mismatched sides is informative.
    """
    margin = abs(int(goal_difference))

    if margin < 2:
        return 1.0

    if margin == 2:
        return 1.5

    return (11.0 + margin) / 8.0


def expected_score(rating_home: float, rating_away: float, home_advantage: float) -> float:
    """Expected score for the home side, between 0 and 1."""
    difference = rating_home + home_advantage - rating_away

    return 1.0 / (1.0 + 10.0 ** (-difference / 400.0))


@dataclass
class EloResult:
    """Fitted ratings plus the pre-match state of every match."""

    ratings: dict[str, float]
    history: pd.DataFrame
    config: EloConfig

    def rating(self, club: str) -> float:
        return self.ratings.get(club, self.config.initial)

    def table(self, top: int | None = None) -> pd.DataFrame:
        df = pd.DataFrame(
            sorted(self.ratings.items(), key=lambda kv: -kv[1]),
            columns=["club", "elo"],
        )
        df.insert(0, "rank", range(1, len(df) + 1))

        return df.head(top) if top else df


class _Initialiser:
    """Decides what rating a club starts on.

    Tracks the mean rating of clubs already seen in each (country, tier)
    group so a newcomer enters near its peers rather than at the global
    average. Falls back to the country, then to the tier of the
    competition it first appears in.
    """

    def __init__(self, config: EloConfig):
        self.config = config
        self.by_group: dict[tuple[str, object], list[str]] = defaultdict(list)
        self.by_country: dict[str, list[str]] = defaultdict(list)

    def _mean(self, members: list[str], ratings: dict[str, float]) -> float | None:
        if len(members) < 3:
            return None

        return float(np.mean([ratings[m] for m in members]))

    def start(
        self,
        club: str,
        country: str,
        tier: object,
        kind: str,
        ratings: dict[str, float],
    ) -> float:
        group = self._mean(self.by_group[(country, tier)], ratings)

        if group is not None:
            return group - self.config.newcomer_penalty

        national = self._mean(self.by_country[country], ratings)

        if national is not None:
            return national - self.config.newcomer_penalty

        if kind == "uefa" and tier in ENTRY_BY_TIER:
            return ENTRY_BY_TIER[tier]

        return self.config.initial

    def record(self, club: str, country: str, tier: object) -> None:
        if club not in self.by_group[(country, tier)]:
            self.by_group[(country, tier)].append(club)

        if club not in self.by_country[country]:
            self.by_country[country].append(club)


def fit(matches: pd.DataFrame, config: EloConfig | None = None) -> EloResult:
    """Run the rating pass over a date-ordered match frame."""
    config = config or EloConfig()

    ratings: dict[str, float] = {}
    initialiser = _Initialiser(config)

    home_before = np.empty(len(matches))
    away_before = np.empty(len(matches))
    expected = np.empty(len(matches))
    change = np.empty(len(matches))

    season = None

    columns = zip(
        matches["home_key"].to_numpy(),
        matches["away_key"].to_numpy(),
        matches["home_country"].to_numpy(),
        matches["away_country"].to_numpy(),
        matches["tier"].to_numpy(),
        matches["kind"].to_numpy(),
        matches["competition"].to_numpy(),
        matches["season"].to_numpy(),
        matches["fthg"].to_numpy(),
        matches["ftag"].to_numpy(),
    )

    for i, (home, away, home_country, away_country, tier, kind, competition,
            current_season, goals_home, goals_away) in enumerate(columns):

        if config.season_regression and current_season != season:
            if season is not None:
                for club in ratings:
                    ratings[club] += config.season_regression * (
                        config.initial - ratings[club]
                    )
            season = current_season

        for club, country in ((home, home_country), (away, away_country)):
            if club not in ratings:
                ratings[club] = initialiser.start(
                    club, country, tier, kind, ratings
                )
            initialiser.record(club, country, tier)

        rating_home = ratings[home]
        rating_away = ratings[away]

        expectation = expected_score(
            rating_home, rating_away, config.home_advantage
        )

        if goals_home > goals_away:
            actual = 1.0
        elif goals_home < goals_away:
            actual = 0.0
        else:
            actual = 0.5

        step = config.k * config.k_multiplier.get(competition, 1.0)
        delta = step * goal_multiplier(goals_home - goals_away) * (actual - expectation)

        home_before[i] = rating_home
        away_before[i] = rating_away
        expected[i] = expectation
        change[i] = delta

        # Zero sum: whatever the home side gains, the away side loses.
        ratings[home] = rating_home + delta
        ratings[away] = rating_away - delta

    history = matches.copy()
    history["elo_home"] = home_before
    history["elo_away"] = away_before
    history["elo_diff"] = (
        home_before + config.home_advantage - away_before
    )
    history["elo_expected"] = expected
    history["elo_change"] = change

    return EloResult(ratings=ratings, history=history, config=config)
