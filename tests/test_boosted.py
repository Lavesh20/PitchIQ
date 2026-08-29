"""Tests for the boosted outcome model.

The dangerous failure here is silent: XGBoost returns probability
columns ordered by class label, so a model that has learned perfectly
can still report draws as home wins if the reordering is wrong. Nothing
raises. The score just gets worse. Most of what follows pins that down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchiq.eval.metrics import OUTCOMES
from pitchiq.models import boosted

FAST = boosted.BoostedConfig(n_estimators=60, early_stopping_rounds=10, max_depth=3)


def synthetic(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """A frame where one feature decides the result, with some noise.

    Strong home sides win, weak ones lose, level ones draw. Any correct
    implementation should find that; the point of the fixture is that we
    know which outcome *should* be predicted for a given row.
    """
    generator = np.random.default_rng(seed)
    strength = generator.normal(0, 1, n)

    target = np.where(strength > 0.5, 0, np.where(strength < -0.5, 2, 1))
    flip = generator.random(n) < 0.1
    target = np.where(flip, generator.integers(0, 3, n), target)

    return pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=n, freq="D"),
            "strength": strength,
            "noise": generator.normal(0, 1, n),
            "target": target,
            "ftr": [OUTCOMES[t] for t in target],
        }
    )


COLUMNS = ["strength", "noise"]


def split(frame: pd.DataFrame):
    cut = int(len(frame) * 0.7)
    edge = int(len(frame) * 0.85)

    return frame.iloc[:cut], frame.iloc[cut:edge], frame.iloc[edge:]


def test_probabilities_are_a_distribution():
    train, validation, test = split(synthetic())
    model = boosted.fit(train, validation, COLUMNS, FAST)

    predicted = model.predict(test)

    assert predicted.shape == (len(test), 3)
    assert np.allclose(predicted.sum(axis=1), 1.0)
    assert (predicted >= 0).all()


def test_columns_are_ordered_home_draw_away():
    """The reordering bug this class exists to prevent."""
    train, validation, _ = split(synthetic())
    model = boosted.fit(train, validation, COLUMNS, FAST)

    probe = pd.DataFrame(
        {"strength": [3.0, 0.0, -3.0], "noise": [0.0, 0.0, 0.0]}
    )
    predicted = model.predict(probe)

    # A strong home side must draw the most probability onto column 0,
    # a weak one onto column 2.
    assert predicted[0].argmax() == OUTCOMES.index("H")
    assert predicted[2].argmax() == OUTCOMES.index("A")
    assert predicted[0][0] > predicted[2][0]
    assert predicted[2][2] > predicted[0][2]


def test_validation_must_follow_training():
    """Overlapping windows are a leak, not a warning."""
    frame = synthetic()
    train = frame.iloc[:600]
    overlapping = frame.iloc[500:700]

    with pytest.raises(ValueError, match="overlaps training"):
        boosted.fit(train, overlapping, COLUMNS, FAST)


def test_empty_validation_is_refused():
    train, _, _ = split(synthetic())

    with pytest.raises(ValueError, match="non-empty validation"):
        boosted.fit(train, train.iloc[:0], COLUMNS, FAST)


def test_early_stopping_reports_a_usable_round_count():
    train, validation, _ = split(synthetic())
    model = boosted.fit(train, validation, COLUMNS, FAST)

    assert 1 <= model.rounds <= FAST.n_estimators


def test_refit_uses_the_round_count_it_is_given():
    train, validation, test = split(synthetic())
    searched = boosted.fit(train, validation, COLUMNS, FAST)

    history = pd.concat([train, validation])
    rebuilt = boosted.refit(history, COLUMNS, searched.rounds, FAST)

    assert rebuilt.rounds == searched.rounds
    assert rebuilt.trained_on["matches"] == len(history)
    # Same ordering contract as the early-stopped model.
    assert np.allclose(rebuilt.predict(test).sum(axis=1), 1.0)


def test_same_seed_gives_the_same_model():
    train, validation, test = split(synthetic())

    first = boosted.fit(train, validation, COLUMNS, FAST).predict(test)
    second = boosted.fit(train, validation, COLUMNS, FAST).predict(test)

    assert np.allclose(first, second)


def test_importance_covers_every_column_and_ranks_the_real_one_first():
    train, validation, _ = split(synthetic())
    model = boosted.fit(train, validation, COLUMNS, FAST)

    table = model.importance()

    assert set(table.feature) == set(COLUMNS)
    assert table.iloc[0].feature == "strength"


def test_model_beats_the_base_rate():
    """A sanity floor: if this fails, nothing downstream is worth reading."""
    from pitchiq.eval import metrics

    train, validation, test = split(synthetic())
    model = boosted.fit(train, validation, COLUMNS, FAST)

    rate = (
        train.ftr.value_counts(normalize=True)
        .reindex(OUTCOMES)
        .to_numpy()
    )
    base = np.tile(rate, (len(test), 1))

    assert metrics.ranked_probability_score(
        model.predict(test), test.ftr
    ) < metrics.ranked_probability_score(base, test.ftr)


def test_missing_values_are_accepted():
    """Shots are absent for well over half the record; NaN must not raise."""
    frame = synthetic()
    frame.loc[frame.index[::3], "noise"] = np.nan
    train, validation, test = split(frame)

    model = boosted.fit(train, validation, COLUMNS, FAST)

    assert np.isfinite(model.predict(test)).all()
