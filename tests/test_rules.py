"""Tests for competition format as data.

The value of moving the bracket out of the simulator is that a wrong
format now fails loudly instead of quietly producing a competition with
the wrong number of quarter-finalists. Most of these check that it does.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pitchiq.sim import rules, tournament


def test_the_current_format_is_self_consistent():
    rules.LEAGUE_PHASE_36.validate()


@pytest.mark.parametrize("season", ["2024/25", "2025/26", "2026/27"])
def test_registered_seasons_resolve(season):
    assert rules.for_season(season).clubs == 36


def test_an_unregistered_season_raises_rather_than_guessing():
    """Silently applying this year's rules to an unknown season would
    produce a confident answer to a question nobody has checked."""
    with pytest.raises(KeyError, match="no format registered"):
        rules.for_season("2031/32")


def test_places_add_up():
    fmt = rules.LEAGUE_PHASE_36

    assert fmt.direct_qualifiers + fmt.playoff_places == 24
    assert fmt.eliminated_from == 24
    assert fmt.clubs - fmt.eliminated_from == 12


def test_play_off_bands_must_cover_their_positions_exactly():
    broken = dataclasses.replace(
        rules.LEAGUE_PHASE_36,
        playoff_bands=rules.LEAGUE_PHASE_36.playoff_bands[:-1],
    )

    with pytest.raises(ValueError, match="play-off bands do not cover"):
        broken.validate()


def test_round_of_16_must_use_every_direct_qualifier():
    broken = dataclasses.replace(
        rules.LEAGUE_PHASE_36,
        round_of_16=(((0, 1), 3), ((0, 1), 0), ((2, 3), 2), ((4, 5), 1)),
    )

    with pytest.raises(ValueError, match="round of 16 hosts do not cover"):
        broken.validate()


def test_each_play_off_band_feeds_the_bracket_once():
    broken = dataclasses.replace(
        rules.LEAGUE_PHASE_36,
        round_of_16=(((0, 1), 3), ((6, 7), 3), ((2, 3), 2), ((4, 5), 1)),
    )

    with pytest.raises(ValueError, match="band must feed the round of 16 once"):
        broken.validate()


def test_more_places_than_clubs_is_refused():
    broken = dataclasses.replace(rules.LEAGUE_PHASE_36, clubs=20)

    with pytest.raises(ValueError, match="exceed 20 clubs"):
        broken.validate()


def test_the_simulator_refuses_a_field_of_the_wrong_size():
    fixtures = pd.DataFrame([{"home": 0, "away": 1}])

    with pytest.raises(ValueError, match="expects 36 clubs, got 4"):
        tournament.run(
            tournament_model(4), [f"c{i}" for i in range(4)], fixtures, runs=2
        )


def tournament_model(n: int):
    from scipy.stats import poisson

    class Flat:
        def score_matrix(self, home, away):
            goals = np.arange(6)
            grid = np.outer(poisson.pmf(goals, 1.4), poisson.pmf(goals, 1.1))
            return grid / grid.sum()

    return Flat()


def _fixtures(n: int = 36) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"home": i, "away": (i + step) % n}
            for i in range(n)
            for step in (1, 2, 3, 4)
        ]
    )


def test_the_legacy_module_constants_still_match_the_format():
    """Kept as aliases so nothing that imported them broke."""
    assert tournament.PLAYOFF_BANDS == list(rules.DEFAULT.playoff_bands)
    assert tournament.ROUND_OF_16 == list(rules.DEFAULT.round_of_16)
    assert tournament.EXTRA_TIME_SHARE == rules.DEFAULT.extra_time_share


def test_away_goals_changes_who_goes_through():
    """The abolished rule is still implementable from the config alone.

    Not used for any registered season, but it is the reason the field
    exists: an older format should be addable without editing the
    simulator.
    """
    keys = [f"c{i}" for i in range(36)]
    fixtures = _fixtures()
    model = tournament_model(36)

    modern = tournament.run(model, keys, fixtures, runs=300, seed=0)
    old = tournament.run(
        model, keys, fixtures, runs=300, seed=0,
        rules=dataclasses.replace(rules.LEAGUE_PHASE_36, away_goals=True),
    )

    # Same league phase, so the same clubs qualify; the bracket resolves
    # differently once away goals settle level aggregates.
    assert (modern.position == old.position).all()
    assert not (modern.reached["wins_it"] == old.reached["wins_it"]).all()
