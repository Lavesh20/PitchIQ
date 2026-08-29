"""Run every model once over the same walk-forward folds, and keep it.

Three scripts had grown their own copy of this loop, which is how two
of them ended up measuring slightly different things. It is also slow:
a full pass refits Dixon-Coles and the boosted model once per season.

So the pass lives here, and its output is cached. Anything that wants
to ask a question of the backtest — how well calibrated is it, does
sampling parameters help, does a blend beat its parts — reads the same
predictions rather than generating its own and hoping they match.

The discipline the folds enforce:

* Each fold trains on everything before its season and predicts only
  that season, so no model is staler than any other. Comparing a model
  refit yearly against one frozen for three years measures the refit
  schedule, which this project has done twice by accident.
* Boosting's round count comes from a validation year ending where the
  fold's test season begins, then the model is rebuilt on the fold's
  full history so nothing is lost to holding that year back.
* Bookmaker prices are joined at the end, never fitted on. They are the
  benchmark, and a model shown the price would only learn to copy it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import matches as match_stream
from ..config import PROCESSED
from ..features import build as feature_build
from ..models import boosted
from ..models import dixon_coles as dc
from ..models.outcome import OutcomeModel
from . import market
from .metrics import OUTCOMES

PREDICTIONS = PROCESSED / "backtest_predictions.parquet"

FOLDS = [pd.Timestamp(d) for d in ("2023-07-01", "2024-07-01", "2025-07-01")]
END = pd.Timestamp("2026-07-01")
VALIDATION_YEARS = 1
# Form over a club's first few matches is mostly the absence of it.
MIN_HISTORY = 10

MODELS = ["base_rate", "elo", "dixon_coles", "boosting"]


def _columns(name: str) -> list[str]:
    return [f"{name}_{o.lower()}" for o in OUTCOMES]


def probabilities(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Pull one model's H/D/A columns back out as an array."""
    return frame[_columns(name)].to_numpy()


def run(verbose: bool = True) -> pd.DataFrame:
    """One row per held-out match, with every model's forecast on it."""
    features = feature_build.load()
    features = features[
        (features.home_played >= MIN_HISTORY) & (features.away_played >= MIN_HISTORY)
    ]
    columns = feature_build.feature_columns(features)

    stream = match_stream.load()
    stream = stream[stream.match_id.isin(features.match_id)]

    priced = market.load()
    parts = []

    for fold_start, fold_end in zip(FOLDS, FOLDS[1:] + [END]):
        history = features[features.date < fold_start]
        season = features[
            (features.date >= fold_start) & (features.date < fold_end)
        ]

        if season.empty:
            continue

        validation_start = fold_start - pd.DateOffset(years=VALIDATION_YEARS)
        train = history[history.date < validation_start]
        validation = history[history.date >= validation_start]

        out = season[
            ["match_id", "date", "kind", "competition", "comp_tier",
             "home_key", "away_key", "fthg", "ftag", "ftr"]
        ].copy()
        out["fold"] = str(fold_start.date())

        rate = (
            history.ftr.value_counts(normalize=True)
            .reindex(OUTCOMES)
            .to_numpy()
        )
        out[_columns("base_rate")] = np.tile(rate, (len(season), 1))

        domestic = history[history.kind == "domestic"]
        mapping = OutcomeModel.fit(domestic.elo_diff, domestic.ftr)
        out[_columns("elo")] = mapping.predict(season.elo_diff)

        goals = dc.fit(
            stream[stream.date < fold_start],
            dc.DixonColesConfig(xi=0.0010, ridge=0.5),
            reference=history.date.max(),
        )
        out[_columns("dixon_coles")] = np.array(
            [
                [goals.predict(h, a)[o] for o in OUTCOMES]
                for h, a in zip(season.home_key, season.away_key)
            ]
        )

        searched = boosted.fit(train, validation, columns)
        model = boosted.refit(history, columns, searched.rounds)
        out[_columns("boosting")] = model.predict(season)

        parts.append(out)

        if verbose:
            print(
                f"  {fold_start.date()} to {fold_end.date()}: "
                f"{len(season):,} matches, boosting {searched.rounds} rounds"
            )

    frame = pd.concat(parts, ignore_index=True)

    # Prices last, and only where they exist: UEFA ties are unpriced, so
    # market columns are NaN there rather than the row being dropped.
    frame = frame.merge(priced, on="match_id", how="left")
    frame["priced"] = frame["market_home"].notna()

    return frame


def save(frame: pd.DataFrame, path=PREDICTIONS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def load(path=PREDICTIONS) -> pd.DataFrame:
    return pd.read_parquet(path)
