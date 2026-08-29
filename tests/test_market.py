"""Tests for the market benchmark.

Two things have to hold. The prices must convert to probabilities
correctly — a de-vigging error would move the benchmark the model is
judged against, in either direction, silently. And the odds must stay
out of everything the model can see, because a model shown the price
can copy it, and then the one question this benchmark exists to answer
cannot be answered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchiq.eval import market


@pytest.fixture
def priced(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "match_id": ["a", "b", "c", "d"],
            "AvgCH": [2.00, 1.50, np.nan, 0.90],
            "AvgCD": [4.00, 4.00, 3.50, 3.00],
            "AvgCA": [4.00, 7.00, 2.10, 4.00],
        }
    )
    path = tmp_path / "domestic.parquet"
    frame.to_parquet(path, index=False)
    monkeypatch.setattr(market, "DOMESTIC", path)

    return frame


def test_probabilities_sum_to_one(priced):
    out = market.load()

    assert np.allclose(market.probabilities(out).sum(axis=1), 1.0)


def test_margin_is_removed(priced):
    """2.00 / 4.00 / 4.00 is a 1.0 book, so it should pass through."""
    out = market.load()
    row = out[out.match_id == "a"]

    assert row.market_home.item() == pytest.approx(0.5)
    assert row.market_draw.item() == pytest.approx(0.25)
    assert row.market_away.item() == pytest.approx(0.25)


def test_shorter_odds_mean_higher_probability(priced):
    out = market.load().set_index("match_id")

    assert out.loc["b", "market_home"] > out.loc["a", "market_home"]


def test_rows_without_a_price_are_dropped(priced):
    assert "c" not in set(market.load().match_id)


def test_impossible_odds_are_dropped(priced):
    """A decimal price at or below 1.0 is a bad row, not a market view."""
    assert "d" not in set(market.load().match_id)


def test_attach_keeps_only_priced_matches(priced):
    fixtures = pd.DataFrame({"match_id": ["a", "b", "c"], "ftr": ["H", "D", "A"]})

    attached = market.attach(fixtures)

    assert list(attached.match_id) == ["a", "b"]
    assert "market_home" in attached.columns


def test_unknown_source_is_refused(priced):
    with pytest.raises(ValueError, match="unknown odds source"):
        market.load("tomorrows_prices")


def test_overround_is_reported_above_one(priced):
    # 1/1.5 + 1/4 + 1/7 is about 1.06, and the NaN row is dropped.
    assert market.margin() > 1.0


def test_odds_are_absent_from_the_feature_table():
    """The guard that keeps this experiment meaningful.

    If a price ever reaches the feature layer, the model learns to copy
    it and every comparison against the market becomes circular.
    """
    from pitchiq import matches
    from pitchiq.features import build

    priced_columns = {c for names in market.SOURCES.values() for c in names}

    assert not priced_columns & set(matches.COLUMNS)
    assert not priced_columns & set(matches.STAT_COLUMNS)
    assert not priced_columns & set(build.IDENTITY)
