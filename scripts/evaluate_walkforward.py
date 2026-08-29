"""Walk-forward comparison of Elo and Dixon-Coles.

Elo updates after every match, so freezing Dixon-Coles at the split and
comparing the two measures the refit schedule as much as the model. Here
Dixon-Coles is refitted at the start of each quarter on everything known
by then, and only predicts matches after that point -- the same
information Elo has.
"""

import sys
import time

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.eval import metrics
from pitchiq.models import dixon_coles as dc
from pitchiq.models import elo, league_strength
from pitchiq.models.outcome import OutcomeModel

SPLIT = pd.Timestamp("2023-07-01")

frame = matches.load()
test = frame[(frame.date >= SPLIT) & (frame.kind == "uefa")].copy()

# --- Elo, already walk-forward by construction ------------------------
history = elo.fit(frame).history
before = history[history.date < SPLIT]
domestic_before = before[before.kind == "domestic"]

strength = league_strength.fit(
    before[before.kind == "uefa"], league_strength.league_means(before)
)
adjusted = league_strength.apply(history, strength)
train_adj = adjusted[(adjusted.date < SPLIT) & (adjusted.kind == "domestic")]

plain = OutcomeModel.fit(domestic_before.elo_diff, domestic_before.ftr)
corrected = OutcomeModel.fit(train_adj.elo_diff_adj, train_adj.ftr)

elo_probs = plain.predict(
    history[(history.date >= SPLIT) & (history.kind == "uefa")].elo_diff
)
elo_ls_probs = corrected.predict(
    adjusted[(adjusted.date >= SPLIT) & (adjusted.kind == "uefa")].elo_diff_adj
)

rate = domestic_before.ftr.value_counts(normalize=True).reindex(
    metrics.OUTCOMES
).to_numpy()

results = [
    ("base rate", np.tile(rate, (len(test), 1))),
    ("elo", elo_probs),
    ("elo + league strength", elo_ls_probs),
]

# --- Dixon-Coles, refitted quarterly ---------------------------------
cuts = pd.date_range(SPLIT, frame.date.max() + pd.Timedelta(days=92), freq="QS")

configs = [
    ("xi=0.0010 ridge=0.5", dc.DixonColesConfig(xi=0.0010, ridge=0.5)),
    ("xi=0.0010 ridge=0.1", dc.DixonColesConfig(xi=0.0010, ridge=0.1)),
    ("xi=0.0018 ridge=0.1", dc.DixonColesConfig(xi=0.0018, ridge=0.1)),
]

for label, config in configs:
    started = time.time()
    probabilities = np.full((len(test), 3), np.nan)
    positions = {m: i for i, m in enumerate(test.match_id)}
    refits = 0

    for start, end in zip(cuts[:-1], cuts[1:]):
        window = test[(test.date >= start) & (test.date < end)]

        if window.empty:
            continue

        model = dc.fit(
            frame[frame.date < start], config, reference=start
        )
        refits += 1

        for match_id, home, away in zip(window.match_id, window.home_key, window.away_key):
            prediction = model.predict(home, away)
            probabilities[positions[match_id]] = [
                prediction[o] for o in metrics.OUTCOMES
            ]

    covered = ~np.isnan(probabilities[:, 0])
    print(f"  {label}: {refits} refits, {covered.sum()}/{len(test)} matches "
          f"covered, {time.time()-started:.0f}s")

    results.append((f"dixon-coles {label}", probabilities))

print(f"\n=== held-out European matches ({len(test):,}) ===")
for name, probabilities in results:
    mask = ~np.isnan(np.asarray(probabilities)[:, 0])
    s = metrics.summary(np.asarray(probabilities)[mask], test.ftr[mask])
    print(f"  {name:34s} RPS {s['rps']:.4f}  log-loss {s['log_loss']:.4f}  "
          f"acc {s['accuracy']:.1%}  n={s['n']}")
