"""The blended predictor and its calibrated scoreline grid."""

import numpy as np
import pandas as pd
import pytest

from pitchiq.eval import metrics
from pitchiq.models import dixon_coles as dc
from pitchiq.models import elo as elo_model
from pitchiq.models import ensemble, league_strength
from pitchiq.models.outcome import OutcomeModel


def _matches(n=500, seed=0):
    rng = np.random.default_rng(seed)
    clubs = ["big_a", "big_b", "small_a", "small_b"]
    power = {"big_a": 1.7, "big_b": 1.5, "small_a": 0.8, "small_b": 0.6}

    rows = []
    for k in range(n):
        i, j = rng.choice(4, 2, replace=False)
        home, away = clubs[i], clubs[j]
        rows.append({
            "match_id": f"m{k}",
            "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(k)),
            "kind": "domestic", "competition": "E0", "tier": 1,
            "season": "2019/20",
            "home_key": home, "away_key": away,
            "home_team": home, "away_team": away,
            "home_country": "England", "away_country": "England",
            "home_cc": "ENG", "away_cc": "ENG",
            "fthg": int(rng.poisson(power[home] * 1.25)),
            "ftag": int(rng.poisson(power[away])),
        })
    frame = pd.DataFrame(rows)
    frame["ftr"] = np.where(frame.fthg > frame.ftag, "H",
                            np.where(frame.fthg < frame.ftag, "A", "D"))
    return frame


@pytest.fixture(scope="module")
def predictor():
    frame = _matches()
    return ensemble.build(frame, frame.date.max() + pd.Timedelta(days=1), weight=0.6)


def test_outcome_probabilities_sum_to_one(predictor):
    p = predictor.outcome_probabilities("big_a", "small_a")

    assert sum(p.values()) == pytest.approx(1.0)


def test_stronger_side_favoured(predictor):
    p = predictor.outcome_probabilities("big_a", "small_a")

    assert p["H"] > p["A"]


def test_score_matrix_is_a_distribution(predictor):
    grid = predictor.score_matrix("big_a", "small_a")

    assert grid.sum() == pytest.approx(1.0)
    assert (grid >= 0).all()


def test_score_matrix_marginals_match_the_blend(predictor):
    """The whole point of the rescaling: the grid must agree with the
    blended outcome probabilities, or simulation and forecast disagree."""
    grid = predictor.score_matrix("big_a", "small_b")
    target = predictor.outcome_probabilities("big_a", "small_b")

    assert np.tril(grid, -1).sum() == pytest.approx(target["H"], abs=1e-9)
    assert np.trace(grid) == pytest.approx(target["D"], abs=1e-9)
    assert np.triu(grid, 1).sum() == pytest.approx(target["A"], abs=1e-9)


def test_expected_goals_favour_the_stronger_side(predictor):
    home, away = predictor.expected_goals("big_a", "small_a")

    assert home > away
    assert 0.0 < away < home < 6.0


def test_weight_of_one_ignores_the_goals_model():
    frame = _matches()
    cutoff = frame.date.max() + pd.Timedelta(days=1)

    only_elo = ensemble.build(frame, cutoff, weight=1.0)
    blended = ensemble.build(frame, cutoff, weight=0.5)

    assert only_elo.outcome_probabilities("big_a", "small_a") != \
        blended.outcome_probabilities("big_a", "small_a")


def test_choose_weight_prefers_the_better_component():
    """Given one useless component, the search must ignore it."""
    actual = ["H"] * 80 + ["A"] * 20

    good = np.tile([0.8, 0.1, 0.1], (100, 1))
    useless = np.tile([0.1, 0.1, 0.8], (100, 1))

    assert ensemble.choose_weight(good, useless, actual) > 0.8
    assert ensemble.choose_weight(useless, good, actual) < 0.2


def test_choose_weight_returns_a_valid_weight():
    rng = np.random.default_rng(0)
    a = rng.dirichlet([2, 2, 2], size=200)
    b = rng.dirichlet([2, 2, 2], size=200)
    actual = rng.choice(metrics.OUTCOMES, size=200)

    weight = ensemble.choose_weight(a, b, actual)

    assert 0.0 <= weight <= 1.0
