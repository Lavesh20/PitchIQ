"""Does the feature layer add anything over the rating it is built on?

The comparison holds the model class fixed — the same logistic
regression, the same split, the same rows — and varies only the columns
it is shown. Anything it gains is therefore attributable to the
features and not to a change of estimator, which is the mistake that
made the first ensemble result unreliable.

The difference is reported with a paired bootstrap interval. A raw
"0.2101 against 0.2091" says nothing on its own about whether the gap
would survive a different 39,000 matches.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pitchiq.eval import metrics
from pitchiq.features import build as features

TEST_START = pd.Timestamp("2023-07-01")
# Form over a club's first handful of matches is mostly the absence of
# information. Requiring some history keeps the comparison about the
# features rather than about how each model copes with NaN.
MIN_HISTORY = 10
BOOTSTRAP = 2000

ORDER = ["H", "D", "A"]


def rps_per_match(probabilities: np.ndarray, actual) -> np.ndarray:
    """The score for each match, needed for a *paired* bootstrap."""
    index = np.array([ORDER.index(a) for a in actual])
    observed = np.eye(3)[index]
    gap = np.cumsum(probabilities, axis=1) - np.cumsum(observed, axis=1)

    return (gap[:, :2] ** 2).sum(axis=1) / 2


def fit_predict(train, test, columns):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000),
    )
    model.fit(train[columns], train.ftr)

    predicted = model.predict_proba(test[columns])
    classes = list(model.classes_)

    return predicted[:, [classes.index(o) for o in ORDER]]


def report(name, probabilities, test):
    uefa = (test.is_uefa == 1).to_numpy()
    actual = test.ftr.to_numpy()

    print(
        f"{name:<26} RPS {metrics.ranked_probability_score(probabilities, actual):.4f}"
        f"   log-loss {metrics.log_loss(probabilities, actual):.4f}"
        f"   RPS on UEFA {metrics.ranked_probability_score(probabilities[uefa], actual[uefa]):.4f}"
    )


frame = features.load()
frame = frame[
    (frame.home_played >= MIN_HISTORY) & (frame.away_played >= MIN_HISTORY)
]

train = frame[frame.date < TEST_START]
test = frame[frame.date >= TEST_START]
actual = test.ftr.to_numpy()

print(
    f"train {len(train):,} matches to {TEST_START.date()}, "
    f"test {len(test):,} of which {int(test.is_uefa.sum()):,} European\n"
)

columns = features.feature_columns(frame)
rating_only = ["elo_diff"]
without_rating = [c for c in columns if not c.startswith("elo")]

rate = train.ftr.value_counts(normalize=True)
report("base rate", np.tile([rate[o] for o in ORDER], (len(test), 1)), test)

rating = fit_predict(train, test, rating_only)
everything = fit_predict(train, test, columns)

report("elo only", rating, test)
report("elo + features", everything, test)
report("features without elo", fit_predict(train, test, without_rating), test)

# --- is the gain real, or is it this particular test window? ---------
difference = rps_per_match(rating, actual) - rps_per_match(everything, actual)
generator = np.random.default_rng(0)
resampled = np.array(
    [
        difference[generator.integers(0, len(difference), len(difference))].mean()
        for _ in range(BOOTSTRAP)
    ]
)

low, high = np.percentile(resampled, [2.5, 97.5])
print(
    f"\nfeatures gain {difference.mean():+.5f} RPS over elo alone"
    f"   95% CI [{low:+.5f}, {high:+.5f}]"
)
print("interval excludes zero" if low > 0 else "interval includes zero: not significant")
