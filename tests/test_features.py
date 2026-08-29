"""Feature tests, weighted heavily toward one question: does a feature
row for a match contain anything that was not knowable before it kicked
off? Every other property of a feature layer can be wrong and cost us
accuracy. That one can be wrong and cost us the whole result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pitchiq.features import build as feature_build
from pitchiq.features import rolling


def fixtures(rows) -> pd.DataFrame:
    """A small match frame in the shape the feature pass expects."""
    frame = pd.DataFrame(
        rows,
        columns=["date", "home_key", "away_key", "fthg", "ftag"],
    )
    frame["date"] = pd.to_datetime(frame["date"])

    return frame.sort_values("date", kind="mergesort").reset_index(drop=True)


LEAGUE = fixtures(
    [
        ("2020-01-01", "a", "b", 2, 0),
        ("2020-01-08", "c", "a", 1, 1),
        ("2020-01-15", "b", "c", 0, 3),
        ("2020-01-22", "a", "c", 4, 1),
        ("2020-01-29", "b", "a", 1, 2),
        ("2020-02-05", "c", "b", 2, 2),
        ("2020-02-12", "a", "b", 0, 1),
    ]
)


def test_first_match_has_no_history():
    out = rolling.build(LEAGUE)

    row = out.iloc[0]

    assert np.isnan(row["home_form_pts_5"])
    assert np.isnan(row["away_form_pts_5"])
    assert np.isnan(row["home_rest_days"])
    assert row["home_played"] == 0
    assert row["h2h_n"] == 0


def test_features_use_only_earlier_matches():
    """Change a result, and no feature at or before that match may move.

    This is the leakage test. It does not check that the windows are the
    sizes we intended — it checks the property that matters: a row is a
    function of the past alone.
    """
    out = rolling.build(LEAGUE)

    for i in range(len(LEAGUE)):
        altered = LEAGUE.copy()
        altered.loc[i, "fthg"] = 9
        altered.loc[i, "ftag"] = 0

        after = rolling.build(altered)

        pd.testing.assert_frame_equal(
            out.iloc[: i + 1],
            after.iloc[: i + 1],
            check_exact=False,
            obj=f"rows up to and including match {i}",
        )


def test_altering_a_result_moves_the_next_row():
    """The mirror of the leakage test: state must actually be flowing.

    A pass that returned constant columns would sail through the test
    above, so something has to fail when the past does change.
    """
    altered = LEAGUE.copy()
    altered.loc[0, "fthg"] = 0
    altered.loc[0, "ftag"] = 5

    before = rolling.build(LEAGUE)
    after = rolling.build(altered)

    assert not before.iloc[1].equals(after.iloc[1])


def test_form_and_venue_windows_are_correct():
    out = rolling.build(LEAGUE)

    # Club "a" going into the last match (2020-02-12) has played four:
    # W 2-0, D 1-1, W 4-1, W 2-1 -> 3, 1, 3, 3 points.
    last = out.iloc[6]

    assert last["home_form_pts_5"] == pytest.approx((3 + 1 + 3 + 3) / 4)
    assert last["home_form_pts_3"] == pytest.approx((1 + 3 + 3) / 3)
    assert last["home_gf_5"] == pytest.approx((2 + 1 + 4 + 2) / 4)
    assert last["home_ga_5"] == pytest.approx((0 + 1 + 1 + 1) / 4)

    # Its home record alone is the 2-0 and the 4-1.
    assert last["home_venue_pts"] == pytest.approx(3.0)
    assert last["home_venue_gf"] == pytest.approx(3.0)


def test_rest_days_and_congestion():
    out = rolling.build(LEAGUE)

    # "a" last played on 2020-01-29, the match is 2020-02-12.
    assert out.iloc[6]["home_rest_days"] == pytest.approx(14.0)
    # Only the 2020-01-29 match falls inside the 14-day window, and the
    # window is strict, so the boundary match itself is excluded.
    assert out.iloc[6]["home_matches_recent"] == pytest.approx(0.0)


def test_head_to_head_is_oriented_to_the_home_side():
    """A 2-0 for "a" at home must read as a 0-2 when "b" hosts."""
    out = rolling.build(LEAGUE)

    # Match 4 is b v a, after a beat b 2-0 and nothing else between them.
    row = out.iloc[4]

    assert row["h2h_n"] == 1
    assert row["h2h_pts"] == pytest.approx(0.0)
    assert row["h2h_gf"] == pytest.approx(0.0)
    assert row["h2h_ga"] == pytest.approx(2.0)

    # Match 6 is a v b again: a has 2-0 W and 2-1 W over b.
    later = out.iloc[6]

    assert later["h2h_n"] == 2
    assert later["h2h_pts"] == pytest.approx(3.0)
    assert later["h2h_gf"] == pytest.approx(2.0)
    assert later["h2h_ga"] == pytest.approx(0.5)


def test_head_to_head_draw_is_a_point_for_both_sides():
    """The flip that a naive ``3 - points`` gets wrong."""
    drawn = fixtures(
        [
            ("2020-01-01", "a", "b", 1, 1),
            ("2020-01-08", "b", "a", 0, 0),
        ]
    )

    out = rolling.build(drawn)

    assert out.iloc[1]["h2h_pts"] == pytest.approx(1.0)


def test_missing_detail_stays_missing():
    """No shots recorded must not read as no shots taken."""
    frame = LEAGUE.copy()
    frame["home_shots"] = np.nan
    frame["away_shots"] = np.nan
    frame["home_sot"] = [np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan]
    frame["away_sot"] = [np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan]

    out = rolling.build(frame)

    assert out["home_shots_for"].isna().all()
    # "c" was away in that match and faced 4 on target; the average must
    # ignore the matches with nothing recorded rather than counting them
    # as zeroes.
    assert out.iloc[5]["home_sot_against"] == pytest.approx(4.0)
    assert out.iloc[5]["home_sot_for"] == pytest.approx(2.0)


def test_venue_deque_survives_a_long_run_away():
    """Last five home matches must not fall out of a shared window."""
    rows = [("2020-01-01", "a", "z", 3, 0)]
    rows += [
        (f"2020-02-{day:02d}", "other", "a", 0, 0) for day in range(1, 9)
    ]
    rows.append(("2020-03-01", "a", "z", 0, 0))

    out = rolling.build(fixtures(rows))

    # The 3-0 is eight matches back but is still "a"'s last home match.
    assert out.iloc[9]["home_venue_gf"] == pytest.approx(3.0)


def test_unsorted_input_is_refused():
    out_of_order = LEAGUE.iloc[::-1]

    with pytest.raises(ValueError, match="sorted by date"):
        rolling.build(out_of_order)


def test_diff_columns_are_the_subtraction():
    out = rolling.build(LEAGUE)
    row = out.iloc[6]

    assert row["form_pts_5_diff"] == pytest.approx(
        row["home_form_pts_5"] - row["away_form_pts_5"]
    )
    assert row["rest_days_diff"] == pytest.approx(
        row["home_rest_days"] - row["away_rest_days"]
    )


def test_feature_columns_exclude_identity_and_target():
    frame = pd.DataFrame(
        {c: [0] for c in feature_build.IDENTITY}
        | {"target": [0], "elo_diff": [1.0]}
    )

    assert feature_build.feature_columns(frame) == ["elo_diff"]


def test_target_encoding_matches_outcome_order():
    assert feature_build.OUTCOMES == ["H", "D", "A"]
