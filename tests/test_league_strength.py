"""Per-league corrections to Elo ratings."""

import numpy as np
import pandas as pd
import pytest

from pitchiq.models import elo, league_strength


def _strength(offsets, scales, means):
    return league_strength.LeagueStrength(
        offsets=offsets, scales=scales, means=means,
        counts={c: 100 for c in offsets}, home_advantage=65.0,
    )


def test_no_correction_leaves_ratings_alone():
    s = _strength({"ENG": 0.0}, {"ENG": 1.0}, {"ENG": 1500.0})

    assert s.adjust(np.array([1800.0]), ["ENG"])[0] == pytest.approx(1800.0)


def test_offset_shifts_a_whole_league():
    s = _strength({"SCO": -100.0}, {"SCO": 1.0}, {"SCO": 1500.0})

    adjusted = s.adjust(np.array([1800.0, 1400.0]), ["SCO", "SCO"])

    assert adjusted[0] == pytest.approx(1700.0)
    assert adjusted[1] == pytest.approx(1300.0)
    # A shift preserves the gap, which is why offset alone cannot fix a
    # club whose lead over its own league is the problem.
    assert adjusted[0] - adjusted[1] == pytest.approx(400.0)


def test_scale_below_one_compresses_toward_the_league_mean():
    s = _strength({"SCO": 0.0}, {"SCO": 0.5}, {"SCO": 1500.0})

    adjusted = s.adjust(np.array([1900.0, 1100.0]), ["SCO", "SCO"])

    assert adjusted[0] == pytest.approx(1700.0)
    assert adjusted[1] == pytest.approx(1300.0)
    assert adjusted[0] - adjusted[1] == pytest.approx(400.0 * 0.5)


def test_unknown_country_is_left_uncorrected():
    s = _strength({"ENG": 50.0}, {"ENG": 1.1}, {"ENG": 1500.0})

    assert s.adjust(np.array([1700.0]), ["XXX"])[0] == pytest.approx(1700.0)


def _european(rows):
    return pd.DataFrame(rows)


def test_fit_detects_an_overrated_league():
    """A league whose clubs keep losing in Europe should be marked down."""
    rows = []
    for i in range(200):
        # Ratings say these sides are level; results say otherwise.
        rows.append({
            "elo_home": 1600.0, "elo_away": 1600.0,
            "home_cc": "WEAK", "away_cc": "STRONG",
            "ftr": "A",
        })
    strength = league_strength.fit(
        _european(rows), means={"WEAK": 1500.0, "STRONG": 1500.0}
    )

    assert strength.offsets["WEAK"] < strength.offsets["STRONG"]


def test_fit_centres_the_offsets():
    """Only differences are identified, so the weighted mean is anchored."""
    rows = [
        {"elo_home": 1600.0, "elo_away": 1500.0,
         "home_cc": "A", "away_cc": "B", "ftr": "H"}
        for _ in range(50)
    ] + [
        {"elo_home": 1500.0, "elo_away": 1600.0,
         "home_cc": "B", "away_cc": "A", "ftr": "A"}
        for _ in range(50)
    ]

    strength = league_strength.fit(
        _european(rows), means={"A": 1500.0, "B": 1500.0}
    )
    weights = [strength.counts[c] for c in strength.offsets]
    values = list(strength.offsets.values())

    assert np.average(values, weights=weights) == pytest.approx(0.0, abs=1e-6)


def test_same_country_matches_are_ignored():
    """Only cross-country ties carry information about league strength."""
    rows = [
        {"elo_home": 1600.0, "elo_away": 1500.0,
         "home_cc": "A", "away_cc": "A", "ftr": "H"}
        for _ in range(20)
    ] + [
        {"elo_home": 1500.0, "elo_away": 1500.0,
         "home_cc": "A", "away_cc": "B", "ftr": "H"}
        for _ in range(20)
    ]

    strength = league_strength.fit(_european(rows), means={})

    assert strength.counts["A"] == 20
    assert strength.counts["B"] == 20


def test_league_means_use_domestic_matches_only():
    history = pd.DataFrame([
        {"kind": "domestic", "tier": 1, "home_cc": "SCO", "away_cc": "SCO",
         "elo_home": 1800.0, "elo_away": 1200.0},
        {"kind": "uefa", "tier": 1, "home_cc": "SCO", "away_cc": "ESP",
         "elo_home": 1800.0, "elo_away": 1900.0},
    ])

    means = league_strength.league_means(history)

    # 1800 and 1200 from the domestic match; the European one excluded.
    assert means["SCO"] == pytest.approx(1500.0)
    assert "ESP" not in means
