"""Tests for calibration measurement.

The trap here is a measure that looks plausible on real data and is
quietly wrong, because nothing about a calibration number is obviously
false. So these fix it against cases where the right answer is known by
construction: a forecast that is perfect, one that is over-confident by
a stated amount, and one that is under-confident by a stated amount.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitchiq.eval import calibration, metrics


def outcomes(probabilities: np.ndarray, seed: int = 0) -> list[str]:
    """Draw results *from* the forecast, so it is perfectly calibrated."""
    generator = np.random.default_rng(seed)
    drawn = [generator.choice(3, p=row) for row in probabilities]

    return [metrics.OUTCOMES[d] for d in drawn]


def spread(n: int = 20000, seed: int = 1) -> np.ndarray:
    generator = np.random.default_rng(seed)
    raw = generator.dirichlet([2.0, 2.0, 2.0], size=n)

    return raw


def test_brier_is_zero_for_a_perfect_forecast():
    certain = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    assert metrics.brier_score(certain, ["H", "A"]) == pytest.approx(0.0)


def test_brier_is_one_for_a_confidently_wrong_forecast():
    certain = np.array([[1.0, 0.0, 0.0]])

    assert metrics.brier_score(certain, ["A"]) == pytest.approx(1.0)


def test_brier_appears_in_the_summary():
    probabilities = np.array([[0.5, 0.3, 0.2]])

    assert "brier" in metrics.summary(probabilities, ["H"])


def test_honest_forecasts_are_well_calibrated():
    """Results drawn from the forecast itself must score near zero."""
    probabilities = spread()
    actual = outcomes(probabilities)

    assert calibration.expected_calibration_error(probabilities, actual) < 0.01
    assert abs(calibration.confidence_gap(probabilities, actual)) < 0.02


def test_over_confidence_is_detected_and_signed_positive():
    """Sharpen an honest forecast; results stay as the honest one implied."""
    honest = spread()
    actual = outcomes(honest)

    sharpened = honest**2
    sharpened /= sharpened.sum(axis=1, keepdims=True)

    assert calibration.confidence_gap(sharpened, actual) > 0.05
    assert calibration.expected_calibration_error(
        sharpened, actual
    ) > calibration.expected_calibration_error(honest, actual)


def test_under_confidence_is_signed_negative():
    honest = spread()
    actual = outcomes(honest)

    flattened = honest**0.4
    flattened /= flattened.sum(axis=1, keepdims=True)

    assert calibration.confidence_gap(flattened, actual) < -0.03


def test_reliability_bins_track_the_truth():
    probabilities = spread()
    actual = outcomes(probabilities)

    table = calibration.reliability(probabilities, actual, bins=10)

    assert (table.n > 0).all()
    # Every bin's observed frequency should sit near its stated one.
    populated = table[table.n > 200]
    assert (populated.gap.abs() < 0.03).all()


def test_reliability_counts_every_forecast_outcome_pair():
    probabilities = spread(n=500)
    actual = outcomes(probabilities)

    table = calibration.reliability(probabilities, actual, bins=10)

    assert table.n.sum() == 500 * 3


def test_reliability_can_isolate_one_outcome():
    probabilities = spread(n=500)
    actual = outcomes(probabilities)

    table = calibration.reliability(probabilities, actual, bins=10, outcome="D")

    assert table.n.sum() == 500


def test_certainty_lands_in_the_top_bin():
    """An exact 1.0 must not fall off the end of the bin edges."""
    certain = np.array([[1.0, 0.0, 0.0]])

    table = calibration.reliability(certain, ["H"], bins=10)

    assert table.iloc[-1]["bin"] == "90%-100%"
    assert table.iloc[-1]["observed"] == pytest.approx(1.0)


def test_sharpness_separates_a_shrug_from_a_claim():
    shrug = np.tile([1 / 3, 1 / 3, 1 / 3], (100, 1))
    claim = np.tile([0.8, 0.1, 0.1], (100, 1))

    assert calibration.sharpness(shrug) == pytest.approx(1 / 3)
    assert calibration.sharpness(claim) == pytest.approx(0.8)


def test_a_shrug_is_calibrated_but_useless():
    """Why sharpness has to be read next to calibration, not instead."""
    shrug = np.tile([1 / 3, 1 / 3, 1 / 3], (3000, 1))
    actual = outcomes(shrug)

    assert calibration.expected_calibration_error(shrug, actual) < 0.02
    assert calibration.sharpness(shrug) == pytest.approx(1 / 3)
