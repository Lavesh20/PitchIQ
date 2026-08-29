"""Scoring rules for probabilistic match forecasts.

Accuracy is close to useless here: a model that always says "home win"
scores about 45% and has learned nothing. What matters is whether the
probabilities are honest, which is what these measure.
"""

from __future__ import annotations

import numpy as np

OUTCOMES = ["H", "D", "A"]


def _as_index(actual) -> np.ndarray:
    lookup = {name: i for i, name in enumerate(OUTCOMES)}

    return np.array([lookup[a] for a in actual])


def log_loss(probabilities: np.ndarray, actual) -> float:
    """Mean negative log probability of what actually happened.

    Punishes confident mistakes hard, which is the point.
    """
    index = _as_index(actual)
    picked = probabilities[np.arange(len(index)), index]

    return float(-np.mean(np.log(np.clip(picked, 1e-15, 1.0))))


def ranked_probability_score(probabilities: np.ndarray, actual) -> float:
    """RPS over the ordered outcomes home / draw / away.

    The standard measure for football forecasts. Unlike log loss it
    respects order: predicting a home win when the away side won is
    penalised more than predicting a draw. Lower is better; a perfect
    forecast scores 0.
    """
    index = _as_index(actual)

    observed = np.zeros_like(probabilities)
    observed[np.arange(len(index)), index] = 1.0

    forecast = np.cumsum(probabilities, axis=1)[:, :-1]
    truth = np.cumsum(observed, axis=1)[:, :-1]

    return float(np.mean(np.sum((forecast - truth) ** 2, axis=1) / (len(OUTCOMES) - 1)))


def brier_score(probabilities: np.ndarray, actual) -> float:
    """Mean squared error of the probability vector.

    Where RPS respects the order of the outcomes, Brier does not: it
    treats home, draw and away as three unrelated labels. That makes it
    the blunter measure of the two for football, and the reason to
    report it anyway is that it decomposes cleanly into calibration and
    resolution, which RPS does not. Lower is better.

    Scaled so a perfect forecast is 0 and always predicting the wrong
    outcome with certainty is 1, matching the two-class convention.
    """
    index = _as_index(actual)

    observed = np.zeros_like(probabilities)
    observed[np.arange(len(index)), index] = 1.0

    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)) / 2)


def accuracy(probabilities: np.ndarray, actual) -> float:
    return float(np.mean(np.argmax(probabilities, axis=1) == _as_index(actual)))


def summary(probabilities: np.ndarray, actual) -> dict[str, float]:
    return {
        "n": len(actual),
        "rps": ranked_probability_score(probabilities, actual),
        "log_loss": log_loss(probabilities, actual),
        "brier": brier_score(probabilities, actual),
        "accuracy": accuracy(probabilities, actual),
    }


def implied_probabilities(home, draw, away) -> np.ndarray:
    """Convert decimal odds to probabilities, removing the bookmaker margin.

    Normalising the three reciprocals is the simplest de-vigging method.
    It slightly overstates long shots relative to Shin's method, but it
    needs no fitting and is the standard baseline.
    """
    raw = np.column_stack([1.0 / np.asarray(home, dtype=float),
                           1.0 / np.asarray(draw, dtype=float),
                           1.0 / np.asarray(away, dtype=float)])

    return raw / raw.sum(axis=1, keepdims=True)
