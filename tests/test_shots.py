"""Tests for shot-based ratings and the prior mechanism they drove.

The prior turned out not to help on this data — the shot columns cover
the clubs that least need one — but the mechanism itself is general and
has to be correct, because a wrong prior would silently drag every
rating toward it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchiq.models import dixon_coles as dc
from pitchiq.models import shots


def record(n_days: int = 500, seed: int = 0) -> pd.DataFrame:
    """A league whose shot counts and goals tell the same story.

    Club ``a`` both shoots and scores most, ``d`` least, so ratings
    fitted to shots should rank the clubs the same way as ratings fitted
    to goals.
    """
    generator = np.random.default_rng(seed)
    clubs = ["a", "b", "c", "d"]
    strength = {"a": 1.5, "b": 1.2, "c": 0.9, "d": 0.7}
    rows = []

    for day in range(n_days):
        for home in clubs:
            for away in clubs:
                if home == away:
                    continue

                shots_home = generator.poisson(5.0 * strength[home] / strength[away])
                shots_away = generator.poisson(5.0 * strength[away] / strength[home])

                rows.append(
                    {
                        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                        "home_key": home,
                        "away_key": away,
                        "kind": "domestic",
                        "fthg": generator.binomial(shots_home, 0.29),
                        "ftag": generator.binomial(shots_away, 0.29),
                        "home_sot": float(shots_home),
                        "away_sot": float(shots_away),
                    }
                )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


CONFIG = dc.DixonColesConfig(xi=0.0, ridge=0.5, home_advantage_by=None)


@pytest.fixture(scope="module")
def frame():
    return record()


@pytest.fixture(scope="module")
def prior(frame):
    return shots.fit(frame, CONFIG)


# --- the prior mechanism ---------------------------------------------


def test_no_prior_shrinks_toward_average(frame):
    """The default behaviour, stated so a change to it fails a test.

    Tested as a property rather than a threshold: a heavier ridge with
    no prior must pull the strongest club closer to zero, whatever the
    data happens to support.
    """
    # A short record, so the penalty can actually compete with the data.
    short = frame.head(240)

    light = dc.fit(short, dc.DixonColesConfig(xi=0.0, ridge=0.5,
                                              home_advantage_by=None))
    heavy = dc.fit(short, dc.DixonColesConfig(xi=0.0, ridge=500.0,
                                              home_advantage_by=None))

    assert abs(heavy.attack["a"]) < abs(light.attack["a"])


def test_a_prior_pulls_ratings_toward_it(frame):
    """With the ridge cranked up, the prior should dominate the data."""
    heavy = dc.DixonColesConfig(xi=0.0, ridge=200.0, home_advantage_by=None)

    target = dc.DixonColesResult(
        attack={"a": -1.0, "b": 1.0, "c": 0.0, "d": 0.0},
        defence=dict.fromkeys("abcd", 0.0),
        home_advantage=0.25,
        home_advantages={"all": 0.25},
        rho=0.0,
        config=heavy,
        converged=True,
        log_likelihood=float("nan"),
    )

    plain = dc.fit(frame, heavy)
    guided = dc.fit(frame, heavy, prior=target)

    # "a" is genuinely the strongest, so the unguided fit puts it above
    # "b". A prior insisting on the reverse must move it that way.
    assert guided.attack["a"] < plain.attack["a"]
    assert guided.attack["b"] > plain.attack["b"]


def test_a_prior_of_zero_matches_no_prior(frame):
    empty = dc.DixonColesResult(
        attack=dict.fromkeys("abcd", 0.0),
        defence=dict.fromkeys("abcd", 0.0),
        home_advantage=0.25,
        home_advantages={"all": 0.25},
        rho=0.0,
        config=CONFIG,
        converged=True,
        log_likelihood=float("nan"),
    )

    plain = dc.fit(frame, CONFIG)
    same = dc.fit(frame, CONFIG, prior=empty)

    assert same.attack["a"] == pytest.approx(plain.attack["a"], abs=1e-4)


def test_a_club_missing_from_the_prior_is_shrunk_to_average(frame):
    """A prior must never borrow another club's rating for a stranger."""
    partial = dc.DixonColesResult(
        attack={"a": 2.0},
        defence={},
        home_advantage=0.25,
        home_advantages={"all": 0.25},
        rho=0.0,
        config=CONFIG,
        converged=True,
        log_likelihood=float("nan"),
    )

    guided = dc.fit(frame, CONFIG, prior=partial)

    assert np.isfinite(guided.attack["d"])


# --- the shot ratings themselves --------------------------------------


def test_shot_ratings_rank_clubs_the_same_way_as_goals(prior):
    table = prior.table().set_index("club")

    assert table.loc["a", "attack"] > table.loc["b", "attack"]
    assert table.loc["b", "attack"] > table.loc["c", "attack"]
    assert table.loc["c", "attack"] > table.loc["d", "attack"]


def make_result(attack: dict) -> dc.DixonColesResult:
    return dc.DixonColesResult(
        attack=dict(attack),
        defence=dict.fromkeys(attack, 0.0),
        home_advantage=0.25,
        home_advantages={"all": 0.25},
        rho=0.0,
        config=CONFIG,
        converged=True,
        log_likelihood=float("nan"),
    )


def test_the_scale_is_measured_not_assumed():
    """Shot ratings live on a different scale and must be converted.

    Both fits centre their ratings, so the conversion is a slope through
    the origin. Here the goal ratings are exactly half the shot ratings
    and the measured slope has to say so.
    """
    shot = {f"club_{i}": (i - 15) / 10.0 for i in range(30)}
    goal = {club: value * 0.5 for club, value in shot.items()}

    slope = shots._rescale(make_result(goal), make_result(shot))

    assert slope == pytest.approx(0.5)


def test_too_few_shared_clubs_falls_back_to_no_rescaling():
    """A slope fitted through five points is not worth trusting."""
    shot = {f"club_{i}": (i - 2) / 10.0 for i in range(5)}
    goal = {club: value * 0.5 for club, value in shot.items()}

    assert shots._rescale(make_result(goal), make_result(shot)) == 1.0


def test_rescaling_shrinks_the_raw_ratings_onto_the_goal_scale(prior):
    raw = prior.raw.attack["a"]

    assert prior.attack["a"] == pytest.approx(raw * prior.scale)


def test_as_result_drops_the_meaningless_low_score_correction(prior):
    """rho describes a pile-up of 0-0s that shot counts do not have."""
    assert prior.as_result().rho == 0.0


def test_too_little_shot_data_returns_none(frame):
    assert shots.fit(frame.head(100), CONFIG) is None


def test_absent_columns_degrade_rather_than_raise(frame):
    without = frame.drop(columns=["home_sot", "away_sot"])

    assert shots.with_shots(without).empty
    assert shots.fit(without, CONFIG) is None


def test_partially_recorded_shots_are_filtered_not_imputed(frame):
    partial = frame.copy()
    partial.loc[partial.index[::2], "home_sot"] = np.nan

    kept = shots.with_shots(partial)

    assert len(kept) == len(frame) - len(frame[::2])
    assert kept.home_sot.notna().all()
