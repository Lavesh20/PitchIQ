"""Gradient boosting against the structural models, refit in step.

The plan asks for four models on one split. Elo, logistic regression
and Dixon-Coles exist; this adds the fourth and scores them together.

The comparison is walk-forward by season, and that detail is the whole
point. A first attempt fitted Dixon-Coles once at the start of a
three-year test window while the boosted model's Elo features carried
on updating match by match. Boosting won by 0.008 RPS, and the number
was meaningless: it measured which model had seen the more recent
football, not which model was better. This project has made that exact
mistake once before and it is recorded in the backlog.

So here every model is rebuilt at each season boundary on everything
that came before it, and predicts only that season. Nobody is stale and
nobody is fresh.

Nothing is tuned on the test rows. The boosting round count comes from
a validation year that ends where each fold's test season begins, and
the model is then rebuilt on the fold's full history so that holding
validation back does not cost it training data.
"""

import time

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.eval import metrics
from pitchiq.features import build as features
from pitchiq.models import boosted
from pitchiq.models import dixon_coles as dc
from pitchiq.models.outcome import OutcomeModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Each boundary opens a European season, so a fold is one campaign.
FOLDS = [pd.Timestamp(d) for d in ("2023-07-01", "2024-07-01", "2025-07-01")]
END = pd.Timestamp("2026-07-01")
VALIDATION_YEARS = 1
MIN_HISTORY = 10
BOOTSTRAP = 2000


def rps_per_match(probabilities: np.ndarray, actual) -> np.ndarray:
    index = np.array([metrics.OUTCOMES.index(a) for a in actual])
    observed = np.eye(3)[index]
    gap = np.cumsum(probabilities, axis=1) - np.cumsum(observed, axis=1)

    return (gap[:, :2] ** 2).sum(axis=1) / 2


def interval(difference: np.ndarray, draws: int = BOOTSTRAP):
    generator = np.random.default_rng(0)
    resampled = [
        difference[generator.integers(0, len(difference), len(difference))].mean()
        for _ in range(draws)
    ]

    return np.percentile(resampled, [2.5, 97.5])


frame = features.load()
frame = frame[(frame.home_played >= MIN_HISTORY) & (frame.away_played >= MIN_HISTORY)]
columns = features.feature_columns(frame)

# Dixon-Coles reads the match stream rather than the feature table; the
# rows are the same fixtures, filtered the same way.
stream = matches.load()
stream = stream[stream.match_id.isin(frame.match_id)]

collected: dict[str, list[np.ndarray]] = {}
truth: list[np.ndarray] = []
european: list[np.ndarray] = []


def keep(name: str, probabilities: np.ndarray) -> None:
    collected.setdefault(name, []).append(probabilities)


for fold_start, fold_end in zip(FOLDS, FOLDS[1:] + [END]):
    history = frame[frame.date < fold_start]
    season = frame[(frame.date >= fold_start) & (frame.date < fold_end)]

    if season.empty:
        continue

    validation_start = fold_start - pd.DateOffset(years=VALIDATION_YEARS)
    train = history[history.date < validation_start]
    validation = history[history.date >= validation_start]

    truth.append(season.ftr.to_numpy())
    european.append((season.is_uefa == 1).to_numpy())

    print(
        f"\n--- {fold_start.date()} to {fold_end.date()}: "
        f"{len(history):,} known, {len(season):,} to predict "
        f"({int(season.is_uefa.sum()):,} European) ---"
    )

    rate = history.ftr.value_counts(normalize=True).reindex(metrics.OUTCOMES).to_numpy()
    keep("base rate", np.tile(rate, (len(season), 1)))

    # --- Elo, mapped to probabilities on this fold's history ---------
    domestic = history[history.kind == "domestic"]
    mapping = OutcomeModel.fit(domestic.elo_diff, domestic.ftr)
    keep("elo", mapping.predict(season.elo_diff))

    # --- Dixon-Coles, refit on everything before the season ----------
    start = time.time()
    goals = dc.fit(
        stream[stream.date < fold_start],
        dc.DixonColesConfig(xi=0.0010, ridge=0.5),
        reference=history.date.max(),
    )
    keep(
        "dixon-coles",
        np.array(
            [
                [goals.predict(h, a)[o] for o in metrics.OUTCOMES]
                for h, a in zip(season.home_key, season.away_key)
            ]
        ),
    )
    print(f"    dixon-coles refit in {time.time() - start:.1f}s")

    # --- the same features through a straight line -------------------
    # Boosting is only worth its complexity if it beats this. A linear
    # model on identical columns isolates what the trees add: the
    # interactions and the non-linear shapes, and nothing else.
    linear = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000),
    )
    linear.fit(history[columns], history.ftr)
    classes = list(linear.classes_)
    keep(
        "logistic",
        linear.predict_proba(season[columns])[
            :, [classes.index(o) for o in metrics.OUTCOMES]
        ],
    )

    # --- gradient boosting, rounds chosen on the validation year -----
    start = time.time()
    searched = boosted.fit(train, validation, columns)
    model = boosted.refit(history, columns, searched.rounds)
    keep("gradient boosting", model.predict(season))
    print(
        f"    boosting {searched.rounds} rounds, refit in {time.time() - start:.1f}s"
    )

actual = np.concatenate(truth)
is_european = np.concatenate(european)
scored = {name: np.vstack(parts) for name, parts in collected.items()}

for label, rows in (
    ("all held-out matches", slice(None)),
    ("European only", is_european),
):
    print(f"\n=== {label} ({len(actual[rows]):,}) ===")

    for name, probabilities in scored.items():
        summary = metrics.summary(probabilities[rows], actual[rows])
        print(
            f"  {name:<20} RPS {summary['rps']:.4f}"
            f"   log-loss {summary['log_loss']:.4f}"
            f"   acc {summary['accuracy']:.1%}"
        )

print("\n=== paired differences, positive means the first model is better ===")
pairs = [
    ("gradient boosting", "logistic"),
    ("gradient boosting", "dixon-coles"),
    ("gradient boosting", "elo"),
    ("logistic", "elo"),
    ("dixon-coles", "elo"),
]

for label, rows in (("all", slice(None)), ("European", is_european)):
    print(f"  {label}:")

    for better, worse in pairs:
        difference = (
            rps_per_match(scored[worse][rows], actual[rows])
            - rps_per_match(scored[better][rows], actual[rows])
        )
        low, high = interval(difference)
        verdict = "significant" if low > 0 or high < 0 else "NOT significant"
        print(
            f"    {better} vs {worse:<18} {difference.mean():+.5f}"
            f"   95% CI [{low:+.5f}, {high:+.5f}]   {verdict}"
        )

print("\n=== what the trees split on (final fold) ===")
print(model.importance(15).to_string(index=False))
