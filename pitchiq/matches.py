"""One match stream over every source.

Ratings are learned from domestic leagues and carried onto European
fixtures, so both have to arrive as one chronological sequence with the
same columns and the same club keys. That is all this module does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import clubs
from .config import PROCESSED

DOMESTIC = PROCESSED / "domestic_matches.parquet"
UEFA = PROCESSED / "uefa_matches.parquet"

COLUMNS = [
    "match_id", "date", "kind", "competition", "tier", "season",
    "home_key", "away_key", "home_team", "away_team",
    "home_country", "away_country", "home_cc", "away_cc",
    "stage", "fthg", "ftag", "ftr",
]

# In-match detail, carried only when asked for. Football-data.uk records
# it for the bigger leagues from the mid-2010s on and not at all before,
# so these columns are sparse by nature: roughly 54% of matches since
# 2015/16, nothing for UEFA ties, which arrive from openfootball with the
# scoreline alone. Callers get NaN where a source is silent rather than a
# zero, because "no shots recorded" and "no shots taken" are not the same
# statement and a model must be able to tell them apart.
STAT_SOURCE = {
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_sot", "AST": "away_sot",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellows", "AY": "away_yellows",
    "HR": "home_reds", "AR": "away_reds",
}

STAT_COLUMNS = ["hthg", "htag"] + list(STAT_SOURCE.values())

# Continental competitions ranked by the standard of the field. Feeds
# the initial rating for a club that first appears in Europe, and lets
# the Elo fit weight competitions differently if we want it to.
UEFA_TIERS = {
    "UCL": 1, "UEL": 2, "UECL": 3,
    "UCL_Q": 2, "UEL_Q": 3, "UECL_Q": 4,
}


def _domestic(stats: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(DOMESTIC)

    out = pd.DataFrame(
        {
            "match_id": df["match_id"],
            "date": df["date"],
            "kind": "domestic",
            "competition": df["div"],
            "tier": df["tier"],
            "season": df["season"],
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "home_country": df["country"],
            "away_country": df["country"],
            # A domestic league has no rounds to speak of, so every row
            # gets the same label rather than a missing one; that keeps
            # the column a clean categorical downstream.
            "stage": "LEAGUE",
            "fthg": df["fthg"],
            "ftag": df["ftag"],
            "ftr": df["ftr"],
        }
    )

    out["home_key"] = [
        clubs.resolve(n, c) for n, c in zip(df["home_team"], df["country"])
    ]
    out["away_key"] = [
        clubs.resolve(n, c) for n, c in zip(df["away_team"], df["country"])
    ]

    if stats:
        out["hthg"] = pd.to_numeric(df["hthg"], errors="coerce")
        out["htag"] = pd.to_numeric(df["htag"], errors="coerce")
        for source, name in STAT_SOURCE.items():
            out[name] = pd.to_numeric(df[source], errors="coerce")

    return out


def _uefa(stats: bool = False) -> pd.DataFrame:
    df = pd.read_parquet(UEFA)

    out = pd.DataFrame(
        {
            "date": df["date"],
            "kind": "uefa",
            "competition": df["competition"],
            "tier": df["competition"].map(UEFA_TIERS),
            "season": df["season"],
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "home_country": df["home_country"],
            "away_country": df["away_country"],
            "stage": df["stage"],
            "fthg": df["fthg"],
            "ftag": df["ftag"],
            "ftr": df["ftr"],
        }
    )

    out["home_key"] = [
        clubs.resolve(n, c) for n, c in zip(df["home_team"], df["home_country"])
    ]
    out["away_key"] = [
        clubs.resolve(n, c) for n, c in zip(df["away_team"], df["away_country"])
    ]

    out["match_id"] = (
        df["competition"] + "_"
        + df["date"].dt.strftime("%Y%m%d") + "_"
        + out["home_key"].str.replace(" ", "", regex=False) + "_"
        + out["away_key"].str.replace(" ", "", regex=False)
    )

    if stats:
        for name in STAT_COLUMNS:
            out[name] = np.nan

    return out


def load(
    since: str | None = None,
    domestic: bool = True,
    uefa: bool = True,
    stats: bool = False,
) -> pd.DataFrame:
    """Every played match, in date order, with resolved club keys.

    Unplayed and unresolvable rows are dropped: a match with no score
    teaches a rating model nothing, and a match with an unresolved club
    would attach its result to a phantom.
    """
    frames = []

    if domestic:
        frames.append(_domestic(stats))

    if uefa:
        frames.append(_uefa(stats))

    df = pd.concat(frames, ignore_index=True)

    df = df[df["fthg"].notna() & df["ftag"].notna()]
    df = df[df["home_key"].notna() & df["away_key"].notna()]
    df = df[df["home_key"] != df["away_key"]]

    df["fthg"] = df["fthg"].astype(int)
    df["ftag"] = df["ftag"].astype(int)

    # Domestic rows name the country ("England"), UEFA rows code it
    # ("ENG"). League-level work needs one vocabulary.
    df["home_cc"] = [clubs.country_code(c) for c in df["home_country"]]
    df["away_cc"] = [clubs.country_code(c) for c in df["away_country"]]

    if since:
        df = df[df["date"] >= since]

    # Sorting by date alone leaves same-day matches in file order, which
    # differs between runs; the tiebreak keeps the sequence reproducible
    # because Elo updates are order-dependent.
    df = df.sort_values(["date", "competition", "home_key"], kind="mergesort")

    columns = COLUMNS + STAT_COLUMNS if stats else COLUMNS

    return df[columns].reset_index(drop=True)
