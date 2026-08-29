"""Bookmaker closing odds, as probabilities, for benchmarking only.

Odds are deliberately kept out of :mod:`pitchiq.matches` and out of the
feature table. The question this project has to answer is whether the
model carries information the market does not — and a model that has
been shown the odds cannot answer it. Once a price is a feature, the
trees learn to copy it, the backtest looks superb, and the result is
worthless because on an unpriced fixture there is nothing to copy.

So they live here, in the evaluation package, read straight from the
raw file and joined only at scoring time.

The closing line is the right benchmark rather than the opening one: it
is the price after all the money and all the news, and it is the
hardest thing in football forecasting to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PROCESSED
from .metrics import implied_probabilities

DOMESTIC = PROCESSED / "domestic_matches.parquet"

# Ordered by preference. Average-closing covers essentially every
# domestic match since 2023; Pinnacle is the sharper price but is
# missing for about a fifth of them, and a benchmark that quietly drops
# a fifth of the fixtures is a benchmark on a different sample.
SOURCES = {
    "average_closing": ("AvgCH", "AvgCD", "AvgCA"),
    "pinnacle_closing": ("PSCH", "PSCD", "PSCA"),
    "bet365_closing": ("B365CH", "B365CD", "B365CA"),
    "max_closing": ("MaxCH", "MaxCD", "MaxCA"),
}

# UEFA matches come from openfootball, which carries no prices. Any
# comparison against the market is therefore a domestic comparison, and
# saying so matters: the competition this project forecasts is one we
# cannot benchmark this way.
UNPRICED = "uefa"


def load(source: str = "average_closing") -> pd.DataFrame:
    """Match id and de-vigged H/D/A probabilities, where a price exists."""
    if source not in SOURCES:
        raise ValueError(f"unknown odds source {source!r}; have {sorted(SOURCES)}")

    home, draw, away = SOURCES[source]
    frame = pd.read_parquet(DOMESTIC, columns=["match_id", home, draw, away])

    priced = frame.dropna(subset=[home, draw, away])
    # A quoted price below evens on every outcome, or a non-positive
    # one, is a bad row rather than a market view.
    priced = priced[(priced[[home, draw, away]] > 1.0).all(axis=1)]

    probabilities = implied_probabilities(
        priced[home], priced[draw], priced[away]
    )

    return pd.DataFrame(
        {
            "match_id": priced["match_id"].to_numpy(),
            "market_home": probabilities[:, 0],
            "market_draw": probabilities[:, 1],
            "market_away": probabilities[:, 2],
        }
    )


def attach(frame: pd.DataFrame, source: str = "average_closing") -> pd.DataFrame:
    """Inner-join a match frame to the market, keeping only priced rows."""
    return frame.merge(load(source), on="match_id", how="inner")


def probabilities(frame: pd.DataFrame) -> np.ndarray:
    """The market columns of an attached frame, in H, D, A order."""
    return frame[["market_home", "market_draw", "market_away"]].to_numpy()


def margin(source: str = "average_closing") -> float:
    """Mean overround, as a sanity check that de-vigging did something.

    A typical three-way football book runs 1.03 to 1.08. A figure at or
    below 1.0 means the columns were misread.
    """
    home, draw, away = SOURCES[source]
    frame = pd.read_parquet(DOMESTIC, columns=[home, draw, away]).dropna()

    return float((1 / frame).sum(axis=1).mean())
