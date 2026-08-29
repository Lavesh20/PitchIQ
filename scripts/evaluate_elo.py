"""Backtest Elo: is it better than nothing, and how far off the market?"""

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.eval import metrics
from pitchiq.models import elo
from pitchiq.models.outcome import OutcomeModel

SPLIT = "2023-07-01"

df = matches.load()
result = elo.fit(df)
history = result.history

# Fit the rating-to-probability mapping on domestic matches before the
# split, so nothing in the test window informs it.
train = history[(history.date < SPLIT) & (history.kind == "domestic")]
model = OutcomeModel.fit(train.elo_diff, train.ftr)

print(f"train: {len(train):,} domestic matches before {SPLIT}\n")

# --- held-out European matches ---------------------------------------
uefa = history[(history.date >= SPLIT) & (history.kind == "uefa")]
predictions = model.predict(uefa.elo_diff)

base = np.tile(
    train.ftr.value_counts(normalize=True).reindex(metrics.OUTCOMES).to_numpy(),
    (len(uefa), 1),
)

print("=== held-out UEFA matches ===")
print(f"  elo        {metrics.summary(predictions, uefa.ftr)}")
print(f"  base rate  {metrics.summary(base, uefa.ftr)}")

# --- held-out domestic matches, against the closing line -------------
dom = pd.read_parquet("data/processed/domestic_matches.parquet")
dom = dom[dom.AvgCH.notna() & dom.AvgCD.notna() & dom.AvgCA.notna()]

key = ["date", "home_team", "away_team"]
odds = dom[key + ["AvgCH", "AvgCD", "AvgCA"]]

test = history[(history.date >= SPLIT) & (history.kind == "domestic")]
test = test.merge(odds, on=key, how="inner")

market = metrics.implied_probabilities(test.AvgCH, test.AvgCD, test.AvgCA)
ours = model.predict(test.elo_diff)

print("\n=== held-out domestic matches with closing odds ===")
print(f"  elo        {metrics.summary(ours, test.ftr)}")
print(f"  market     {metrics.summary(market, test.ftr)}")

gap = metrics.ranked_probability_score(ours, test.ftr) - \
      metrics.ranked_probability_score(market, test.ftr)
print(f"\n  RPS gap to the closing line: {gap:+.4f}")
