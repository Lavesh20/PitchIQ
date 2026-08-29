"""Elo update rules and the properties they must hold."""

import numpy as np
import pandas as pd
import pytest

from pitchiq.eval import metrics
from pitchiq.models import elo


def _match(home, away, fthg, ftag, date="2020-01-01", country="England"):
    return {
        "match_id": f"{home}-{away}-{date}", "date": pd.Timestamp(date),
        "kind": "domestic", "competition": "E0", "tier": 1, "season": "2019/20",
        "home_key": home, "away_key": away, "home_team": home, "away_team": away,
        "home_country": country, "away_country": country,
        "fthg": fthg, "ftag": ftag,
        "ftr": "H" if fthg > ftag else ("A" if fthg < ftag else "D"),
    }


def test_expected_score_is_even_for_equal_ratings_without_home_advantage():
    assert elo.expected_score(1500, 1500, 0.0) == pytest.approx(0.5)


def test_home_advantage_favours_the_host():
    assert elo.expected_score(1500, 1500, 65.0) > 0.5


def test_expected_scores_are_complementary():
    home = elo.expected_score(1700, 1500, 65.0)
    away = elo.expected_score(1500, 1700, -65.0)

    assert home + away == pytest.approx(1.0)


@pytest.mark.parametrize(
    "margin, expected",
    [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (-2, 1.5), (3, 1.75), (5, 2.0)],
)
def test_goal_multiplier(margin, expected):
    assert elo.goal_multiplier(margin) == pytest.approx(expected)


def test_ratings_are_zero_sum():
    """Points come from the loser; the system total never changes."""
    frame = pd.DataFrame([
        _match("a", "b", 3, 0),
        _match("b", "c", 1, 1, date="2020-01-08"),
        _match("c", "a", 0, 2, date="2020-01-15"),
    ])

    result = elo.fit(frame, elo.EloConfig(newcomer_penalty=0.0))

    assert sum(result.ratings.values()) == pytest.approx(3 * 1500.0)


def test_winning_raises_and_losing_lowers():
    frame = pd.DataFrame([_match("a", "b", 2, 0)])
    result = elo.fit(frame, elo.EloConfig(newcomer_penalty=0.0))

    assert result.ratings["a"] > 1500 > result.ratings["b"]


def test_bigger_win_moves_ratings_further():
    narrow = elo.fit(pd.DataFrame([_match("a", "b", 1, 0)]),
                     elo.EloConfig(newcomer_penalty=0.0))
    rout = elo.fit(pd.DataFrame([_match("a", "b", 5, 0)]),
                   elo.EloConfig(newcomer_penalty=0.0))

    assert rout.ratings["a"] > narrow.ratings["a"]


def test_history_records_pre_match_ratings():
    """Backtesting depends on the stored rating predating the result."""
    frame = pd.DataFrame([
        _match("a", "b", 3, 0),
        _match("a", "b", 3, 0, date="2020-02-01"),
    ])

    result = elo.fit(frame, elo.EloConfig(newcomer_penalty=0.0))

    assert result.history.elo_home.iloc[0] == pytest.approx(1500.0)
    assert result.history.elo_home.iloc[1] > 1500.0


def test_newcomer_from_unseen_country_enters_by_competition_tier():
    frame = pd.DataFrame([{
        **_match("minnow", "giant", 0, 3, country="AZE"),
        "kind": "uefa", "competition": "UECL_Q", "tier": 4,
    }])

    result = elo.fit(frame)

    # Entered at the tier-4 default of 1250, then lost.
    assert result.history.elo_home.iloc[0] == pytest.approx(1250.0)


class TestMetrics:
    def test_perfect_forecast_scores_zero(self):
        probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        assert metrics.ranked_probability_score(probabilities, ["H", "A"]) == 0.0
        assert metrics.log_loss(probabilities, ["H", "A"]) == pytest.approx(0.0)

    def test_rps_respects_outcome_order(self):
        """Calling a home win when the away side won is the worse miss."""
        near = np.array([[0.0, 1.0, 0.0]])   # said draw
        far = np.array([[1.0, 0.0, 0.0]])    # said home win

        assert metrics.ranked_probability_score(near, ["A"]) < \
               metrics.ranked_probability_score(far, ["A"])

    def test_implied_probabilities_remove_the_margin(self):
        probabilities = metrics.implied_probabilities([2.0], [3.5], [4.0])

        assert probabilities.sum() == pytest.approx(1.0)
        assert probabilities[0, 0] > probabilities[0, 2]
