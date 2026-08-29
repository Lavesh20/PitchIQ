"""Dixon-Coles goals model."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from pitchiq.models import dixon_coles as dc


def _frame(rows):
    return pd.DataFrame(rows)


def _synthetic(n=400, seed=0):
    """Two strong sides, two weak, so the fit has something to find."""
    rng = np.random.default_rng(seed)
    clubs = ["strong_a", "strong_b", "weak_a", "weak_b"]
    power = {"strong_a": 1.6, "strong_b": 1.5, "weak_a": 0.8, "weak_b": 0.7}

    rows = []
    for k in range(n):
        i, j = rng.choice(4, 2, replace=False)
        home, away = clubs[i], clubs[j]
        rows.append({
            "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(k)),
            "home_key": home, "away_key": away,
            "fthg": int(rng.poisson(power[home] * 1.2)),
            "ftag": int(rng.poisson(power[away])),
        })

    return _frame(rows)


def test_gradient_matches_numerical_difference(monkeypatch):
    """An analytic gradient that is subtly wrong still converges, to the
    wrong answer. Check it rather than trust it."""
    captured = {}
    real = dc.minimize

    def spy(fun, x0, **kwargs):
        captured["fun"] = fun
        captured["x0"] = x0
        return real(fun, x0, **kwargs)

    monkeypatch.setattr(dc, "minimize", spy)
    dc.fit(_synthetic(120), dc.DixonColesConfig(xi=0.002))

    fun, x0 = captured["fun"], captured["x0"]
    rng = np.random.default_rng(1)

    for _ in range(3):
        point = rng.normal(scale=0.3, size=len(x0))
        point[-1] = np.clip(point[-1], -0.3, 0.3)

        _, analytic = fun(point)
        numeric = approx_fprime(point, lambda p: fun(p)[0], 1e-6)

        assert np.max(np.abs(analytic - numeric)) < 1e-3


def test_recovers_relative_strength():
    model = dc.fit(_synthetic(600), dc.DixonColesConfig(xi=0.0, ridge=0.01))

    assert model.attack["strong_a"] > model.attack["weak_a"]
    assert model.attack["strong_b"] > model.attack["weak_b"]


def test_home_advantage_is_positive():
    model = dc.fit(_synthetic(600), dc.DixonColesConfig(xi=0.0, ridge=0.01))

    assert model.home_advantage > 0


def test_score_matrix_is_a_distribution():
    model = dc.fit(_synthetic(200), dc.DixonColesConfig(xi=0.002))
    grid = model.score_matrix("strong_a", "weak_a")

    assert grid.shape == (11, 11)
    assert grid.sum() == pytest.approx(1.0)
    assert (grid >= 0).all()


def test_outcome_probabilities_sum_to_one():
    model = dc.fit(_synthetic(200), dc.DixonColesConfig(xi=0.002))
    prediction = model.predict("strong_a", "weak_a")

    assert sum(prediction.values()) == pytest.approx(1.0)


def test_stronger_side_is_favoured():
    model = dc.fit(_synthetic(600), dc.DixonColesConfig(xi=0.0, ridge=0.01))

    strong_home = model.predict("strong_a", "weak_a")
    weak_home = model.predict("weak_a", "strong_a")

    assert strong_home["H"] > strong_home["A"]
    assert weak_home["A"] > weak_home["H"]


def test_home_side_is_favoured_between_equals():
    model = dc.fit(_synthetic(600), dc.DixonColesConfig(xi=0.0, ridge=0.01))
    prediction = model.predict("strong_a", "strong_b")

    assert prediction["H"] > prediction["A"]


def test_rho_lifts_low_scoring_draws():
    """The correction exists because independent Poisson under-predicts
    0-0 and 1-1. A negative rho must raise them."""
    model = dc.fit(_synthetic(400), dc.DixonColesConfig(xi=0.002))
    model.rho = -0.1

    corrected = model.score_matrix("strong_a", "strong_b")

    model.rho = 0.0
    plain = model.score_matrix("strong_a", "strong_b")

    assert corrected[0, 0] > plain[0, 0]
    assert corrected[1, 1] > plain[1, 1]


def test_unknown_club_falls_back_to_average():
    model = dc.fit(_synthetic(200), dc.DixonColesConfig(xi=0.002))

    prediction = model.predict("never_seen", "also_never_seen")

    assert sum(prediction.values()) == pytest.approx(1.0)
    assert prediction["H"] > prediction["A"]  # home advantage still applies


def test_time_decay_drops_ancient_matches():
    frame = _synthetic(200)
    frame.loc[:99, "date"] = pd.Timestamp("1995-01-01")

    model = dc.fit(frame, dc.DixonColesConfig(xi=0.01, weight_floor=1e-4))

    assert model.converged
