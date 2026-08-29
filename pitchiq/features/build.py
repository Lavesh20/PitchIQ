"""Assemble the model-facing feature table.

What belongs here is decided by one question: can this number be
produced for a match using only matches that finished before it?

* Rolling form, venue splits, rest, shots, head-to-head — yes, by
  construction; see :mod:`pitchiq.features.rolling`.
* Elo — yes. The rating pass is sequential and records each club's
  rating *as it stood* before the match, so the whole stream can be
  rated in one go and the ratings stay honest.
* Dixon-Coles and league strength — no. Both are batch fits over a
  window, so running them across all of history and pasting the output
  onto every row would show a 2012 match what happened in 2024. They
  are added inside the walk-forward harness instead, fitted on the
  training window and applied forward.

That distinction is the whole reason this module exists rather than one
big ``fit_everything``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import matches as match_stream
from ..config import PROCESSED
from ..models import elo as elo_model
from . import rolling

FEATURES = PROCESSED / "features.parquet"

# Carried through so a feature row can be traced back to a fixture, and
# so the harness can split, group and score without a second join.
IDENTITY = [
    "match_id", "date", "kind", "competition", "tier", "season",
    "home_key", "away_key", "home_cc", "away_cc",
    "stage", "fthg", "ftag", "ftr",
]

# Stages where a tie is decided rather than a table is filled. Sides
# play them differently — a draw is worth less, and away legs are
# managed — so it is worth telling the model which it is looking at.
KNOCKOUT = {
    "LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL",
    "PLAYOFFS", "QUAL_ROUND_1", "QUAL_ROUND_2", "QUAL_ROUND_3",
}

OUTCOMES = ["H", "D", "A"]


def context(frame: pd.DataFrame) -> pd.DataFrame:
    """Facts about the fixture itself, as numbers a tree can split on."""
    out = pd.DataFrame(index=frame.index)

    # ``tier`` is also carried as identity, so the model-facing copy is
    # named apart from it rather than shadowing it.
    out["comp_tier"] = pd.to_numeric(frame["tier"], errors="coerce")
    out["is_uefa"] = (frame["kind"] == "uefa").astype(float)
    # European ties between clubs from the same country only happen deep
    # in a competition, and domestic rows are all same-country, so this
    # is really "is this a late-stage European tie".
    out["same_country"] = (frame["home_cc"] == frame["away_cc"]).astype(float)
    # Winter fixture congestion and summer restarts are real; the month
    # is the cheapest handle on both.
    out["month"] = frame["date"].dt.month.astype(float)
    out["is_knockout"] = frame["stage"].isin(KNOCKOUT).astype(float)

    return out


def build(
    since: str | None = None,
    config: rolling.RollingConfig | None = None,
    elo_config: elo_model.EloConfig | None = None,
    frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per match: identity, target, and pre-match features.

    ``since`` trims the *output*, not the history the features are built
    from. Rolling form and Elo both need a run-up, so the pass always
    starts from the beginning of the record and the frame is cut at the
    end — asking for 2015 onward and getting features that only know
    about 2015 onward would be a quiet and expensive mistake.
    """
    if frame is None:
        frame = match_stream.load(stats=True)

    rated = elo_model.fit(frame, elo_config).history

    parts = [
        rated[IDENTITY].reset_index(drop=True),
        rated[["elo_home", "elo_away", "elo_diff", "elo_expected"]].reset_index(drop=True),
        context(rated).reset_index(drop=True),
        rolling.build(rated, config).reset_index(drop=True),
    ]

    out = pd.concat(parts, axis=1)
    out["target"] = out["ftr"].map({o: i for i, o in enumerate(OUTCOMES)})

    # Shots are absent for well over half the record, and absent in a
    # patterned way: the leagues and seasons that have them are the
    # bigger and more recent ones. A model given only NaNs can learn
    # that pattern implicitly and call it football. Stating it as its
    # own column at least makes what it learned visible to us.
    out["has_detail"] = (
        out["home_sot_for"].notna() & out["away_sot_for"].notna()
    ).astype(float)

    if since:
        out = out[out["date"] >= since]

    return out.reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every column a model may read — identity and target excluded."""
    blocked = set(IDENTITY) | {"target"}

    return [c for c in frame.columns if c not in blocked]


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of rows where each feature is present.

    Worth looking at before training. Missingness here is not random —
    shots are recorded for the bigger leagues and the later seasons — so
    a feature that is 40% present is also a feature that half-encodes
    "big league, recent season", and a model can learn that instead of
    the football.
    """
    columns = feature_columns(frame)

    return pd.DataFrame(
        {
            "feature": columns,
            "present": [float(frame[c].notna().mean()) for c in columns],
        }
    ).sort_values("present").reset_index(drop=True)


def save(frame: pd.DataFrame, path=FEATURES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def load(path=FEATURES) -> pd.DataFrame:
    return pd.read_parquet(path)
