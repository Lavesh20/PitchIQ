"""Final backtest: base rate, Elo, Dixon-Coles, and the blend.

The blend weight is chosen on a validation window that ends before the
test window begins, so no number below was tuned on what it scores.
"""

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.eval import metrics
from pitchiq.models import ensemble

VALIDATION_START = pd.Timestamp("2022-07-01")
TEST_START = pd.Timestamp("2023-07-01")

frame = matches.load()


def probabilities(predictor, df):
    from_elo = np.array(
        [predictor._elo_probabilities(h, a) for h, a in zip(df.home_key, df.away_key)]
    )
    from_goals = np.array(
        [
            [predictor.goals.predict(h, a)[o] for o in metrics.OUTCOMES]
            for h, a in zip(df.home_key, df.away_key)
        ]
    )
    return from_elo, from_goals


# --- choose the weight on the validation window ----------------------
validation = frame[
    (frame.date >= VALIDATION_START)
    & (frame.date < TEST_START)
    & (frame.kind == "uefa")
]
tuner = ensemble.build(frame, VALIDATION_START, weight=0.5)
elo_v, goals_v = probabilities(tuner, validation)

weight = ensemble.choose_weight(elo_v, goals_v, validation.ftr)
print(f"validation {VALIDATION_START.date()} to {TEST_START.date()}, "
      f"{len(validation)} matches -> w_elo = {weight:.2f}")

# --- refit up to the split and score the test window -----------------
predictor = ensemble.build(frame, TEST_START, weight=weight)
test = frame[(frame.date >= TEST_START) & (frame.kind == "uefa")]
elo_t, goals_t = probabilities(predictor, test)

train = frame[(frame.date < TEST_START) & (frame.kind == "domestic")]
rate = train.ftr.value_counts(normalize=True).reindex(metrics.OUTCOMES).to_numpy()

print(f"\n=== held-out European matches ({len(test):,}) ===")
for name, p in [
    ("base rate", np.tile(rate, (len(test), 1))),
    ("elo + league strength", elo_t),
    ("dixon-coles", goals_t),
    (f"ensemble (w={weight:.2f})", weight * elo_t + (1 - weight) * goals_t),
]:
    s = metrics.summary(p, test.ftr)
    print(f"  {name:26s} RPS {s['rps']:.4f}  log-loss {s['log_loss']:.4f}  "
          f"acc {s['accuracy']:.1%}")

print("\n=== sample forecasts for the 2026/27 draw ===")
squad = pd.read_csv("data/external/ucl_2026_27_clubs.csv")
pairs = pd.read_csv("data/external/ucl_2026_27_pairings.csv")

from pitchiq import clubs
key = {c: clubs.resolve(c, cc) for c, cc in zip(squad.club, squad.country)}

live = ensemble.build(frame, frame.date.max() + pd.Timedelta(days=1), weight=weight)

for home, away in list(zip(pairs.home, pairs.away))[:8]:
    p = live.outcome_probabilities(key[home], key[away])
    xg_home, xg_away = live.expected_goals(key[home], key[away])
    print(f"  {home:22s} v {away:22s}  "
          f"H {p['H']:.0%}  D {p['D']:.0%}  A {p['A']:.0%}   "
          f"xG {xg_home:.2f}-{xg_away:.2f}")
