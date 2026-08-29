"""Tests for parameter uncertainty and how it reaches the simulator.

The property that matters is the one the whole exercise is for: clubs
we have seen a lot of should come out firmly pinned down, and clubs we
have barely seen should not. A bootstrap that returns the same spread
for both would be worse than useless — it would look like we had
accounted for uncertainty while leaving the forecast exactly as
over-confident as before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchiq.models import dixon_coles as dc
from pitchiq.models import uncertainty
from pitchiq.sim import draws


def record(seed: int = 0) -> pd.DataFrame:
    """A league where three clubs play often and one barely plays.

    ``rare`` appears in a handful of matches, so its rating should be
    the least certain of the four by a clear margin.
    """
    generator = np.random.default_rng(seed)
    rows = []
    regulars = ["a", "b", "c"]
    strength = {"a": 1.6, "b": 1.2, "c": 0.9, "rare": 1.2}

    for day in range(200):
        for home in regulars:
            for away in regulars:
                if home == away or generator.random() < 0.6:
                    continue

                rows.append(
                    {
                        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                        "home_key": home,
                        "away_key": away,
                        "fthg": generator.poisson(strength[home] * 1.2),
                        "ftag": generator.poisson(strength[away]),
                        "kind": "domestic",
                    }
                )

    for day in (10, 60, 120, 180):
        rows.append(
            {
                "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                "home_key": "rare",
                "away_key": "a",
                "fthg": generator.poisson(1.2),
                "ftag": generator.poisson(1.9),
                "kind": "domestic",
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


CONFIG = dc.DixonColesConfig(xi=0.0, ridge=0.5, home_advantage_by=None)


@pytest.fixture(scope="module")
def samples():
    return uncertainty.bootstrap(record(), CONFIG, draws=12, seed=1, verbose=False)


def test_sample_weights_change_the_fit():
    frame = record()
    generator = np.random.default_rng(3)
    weights = generator.dirichlet(np.ones(len(frame))) * len(frame)

    point = dc.fit(frame, CONFIG)
    resampled = dc.fit(frame, CONFIG, sample_weights=weights, start_from=point)

    assert point.attack["a"] != resampled.attack["a"]


def test_uniform_sample_weights_reproduce_the_point_estimate():
    """Weights of one everywhere must be the same fit, not a similar one."""
    frame = record()

    point = dc.fit(frame, CONFIG)
    same = dc.fit(frame, CONFIG, sample_weights=np.ones(len(frame)))

    assert same.attack["a"] == pytest.approx(point.attack["a"], abs=1e-6)


def test_sample_weights_length_is_checked():
    frame = record()

    with pytest.raises(ValueError, match="sample_weights has"):
        dc.fit(frame, CONFIG, sample_weights=np.ones(len(frame) - 1))


def test_warm_start_lands_in_the_same_place():
    frame = record()
    point = dc.fit(frame, CONFIG)
    warm = dc.fit(frame, CONFIG, start_from=point)

    assert warm.attack["a"] == pytest.approx(point.attack["a"], abs=1e-3)


def test_a_rarely_seen_club_is_the_least_certain(samples):
    """The whole point: uncertainty must track how much we have seen."""
    spread = samples.spread().set_index("club")

    assert spread.loc["rare", "attack_sd"] > spread.loc["a", "attack_sd"]
    assert spread.loc["rare", "attack_sd"] > spread.loc["b", "attack_sd"]
    assert spread.loc["rare", "attack_sd"] > spread.loc["c", "attack_sd"]


def test_spread_is_not_flat(samples):
    """A bootstrap returning one number for everyone has not worked."""
    spread = samples.spread()

    assert spread.attack_sd.std() > 0
    assert (spread.attack_sd > 0).all()


def test_a_draw_is_a_usable_model(samples):
    model = samples.draw(0)

    grid = model.score_matrix("a", "b")

    assert np.isfinite(grid).all()
    assert grid.sum() == pytest.approx(1.0)


def test_draws_differ_from_one_another(samples):
    first = samples.draw(0).attack["a"]
    second = samples.draw(1).attack["a"]

    assert first != second


def test_save_and_load_round_trip(samples, tmp_path):
    path = tmp_path / "samples.npz"
    samples.save(path)

    restored = uncertainty.ParameterSamples.load(path, CONFIG)

    assert restored.clubs == samples.clubs
    assert restored.draws == samples.draws
    assert np.allclose(restored.attack, samples.attack)
    assert restored.point.attack["a"] == pytest.approx(samples.point.attack["a"])


# --- how the draws reach a simulation --------------------------------


def test_a_single_model_is_one_batch():
    model = dc.fit(record(), CONFIG)

    assert list(draws.batches(model, 500)) == [(model, 500)]


def test_batches_cover_every_run(samples):
    counts = [count for _, count in draws.batches(samples, 1000)]

    assert sum(counts) == 1000
    assert len(counts) == samples.draws


def test_batches_are_within_one_run_of_each_other(samples):
    """An uneven split would weight some parameter sets more than others."""
    counts = [count for _, count in draws.batches(samples, 1007)]

    assert max(counts) - min(counts) <= 1


def test_batches_never_exceed_the_draws_available(samples):
    counts = [count for _, count in draws.batches(samples, 1000, 5)]

    assert len(counts) == 5
    assert sum(counts) == 1000


def test_fewer_runs_than_draws_still_works(samples):
    counts = [count for _, count in draws.batches(samples, 3)]

    assert sum(counts) == 3
    assert all(c > 0 for c in counts)


def test_sampling_widens_the_spread_of_finishing_positions(samples):
    """The measurable consequence, on a league we control.

    Same fixtures, same number of seasons; the only difference is
    whether the ratings are held fixed or drawn. Drawing them must widen
    the distribution of where clubs finish, because it adds a second
    source of variation to the one already there.
    """
    from pitchiq.sim import league

    fixtures = pd.DataFrame(
        [
            {"home_key": h, "away_key": a, "fthg": 0, "ftag": 0,
             "date": pd.Timestamp("2021-01-01")}
            for h in ("a", "b", "c", "rare")
            for a in ("a", "b", "c", "rare")
            if h != a
        ]
    )

    _, fixed = league.positions(samples.point, fixtures, runs=1200, seed=0)
    _, drawn = league.positions(samples, fixtures, runs=1200, seed=0)

    assert drawn.std(axis=0).mean() > fixed.std(axis=0).mean()


# --- the tournament path ---------------------------------------------
# Its bracket is hard-wired to UEFA's 36-club format, so it needs a
# 36-club set of draws. Building one directly rather than bootstrapping
# a real fit keeps the test fast and lets the spread be chosen: club 0
# is both the strongest and the one we are surest about.


def synthetic_samples(n_clubs: int = 36, n_draws: int = 12, seed: int = 0):
    generator = np.random.default_rng(seed)
    clubs = [f"club_{i}" for i in range(n_clubs)]

    centre = np.linspace(0.5, -0.5, n_clubs)
    # Spread widens down the order, as it does in the real bootstrap:
    # the clubs we have seen least are the ones at the bottom.
    spread = np.linspace(0.02, 0.25, n_clubs)

    attack = centre + generator.normal(0, 1, (n_draws, n_clubs)) * spread
    defence = -centre + generator.normal(0, 1, (n_draws, n_clubs)) * spread

    point = dc.DixonColesResult(
        attack=dict(zip(clubs, centre)),
        defence=dict(zip(clubs, -centre)),
        home_advantage=0.25,
        home_advantages={"all": 0.25},
        rho=-0.04,
        config=dc.DixonColesConfig(max_goals=6),
        converged=True,
        log_likelihood=float("nan"),
    )

    return uncertainty.ParameterSamples(
        clubs=clubs,
        attack=attack,
        defence=defence,
        home_advantages={"all": np.full(n_draws, 0.25)},
        rho=np.full(n_draws, -0.04),
        point=point,
    )


def tournament_fixtures(n: int = 36) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"home": i, "away": (i + step) % n}
            for i in range(n)
            for step in (1, 2, 3, 4)
        ]
    )


def test_the_tournament_simulator_accepts_draws():
    """The knockout path batches separately from the league one."""
    from pitchiq.sim import tournament

    samples = synthetic_samples()
    drawn = tournament.run(
        samples, samples.clubs, tournament_fixtures(), runs=400, seed=0,
        parameter_draws=8,
    )

    assert drawn.position.shape == (400, 36)
    # Stitching blocks back together must not break the bracket.
    assert drawn.reached["wins_it"].sum() == 400
    assert drawn.reached["wins_it"].sum(axis=1).max() == 1
    assert drawn.reached["quarter_finals"].sum(axis=1).tolist() == [8] * 400


def test_drawing_ratings_softens_the_favourite():
    """The change the whole exercise exists to produce.

    With ratings held fixed the strongest club wins a fixed share of
    seasons. Once the ratings are drawn, some seasons are played by a
    version of that club we are less sure about, and its share of titles
    must come down rather than up.
    """
    from pitchiq.sim import tournament

    samples = synthetic_samples()
    fixtures = tournament_fixtures()

    fixed = tournament.run(samples.point, samples.clubs, fixtures, runs=4000, seed=0)
    drawn = tournament.run(samples, samples.clubs, fixtures, runs=4000, seed=0)

    best = 0
    assert (
        drawn.reached["wins_it"][:, best].mean()
        < fixed.reached["wins_it"][:, best].mean()
    )
