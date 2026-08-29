"""Backtest Dixon-Coles against Elo on held-out European matches."""

import time

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.eval import metrics
from pitchiq.models import dixon_coles as dc
from pitchiq.models import elo, league_strength
from pitchiq.models.outcome import OutcomeModel

SPLIT = "2023-07-01"

frame = matches.load()
train = frame[frame.date < SPLIT]
test = frame[(frame.date >= SPLIT) & (frame.kind == "uefa")]

# --- Elo baselines, as already established ---------------------------
history = elo.fit(frame).history
before = history[history.date < SPLIT]

strength = league_strength.fit(
    before[before.kind == "uefa"], league_strength.league_means(before)
)
adjusted = league_strength.apply(history, strength)

plain = OutcomeModel.fit(
    before[before.kind == "domestic"].elo_diff,
    before[before.kind == "domestic"].ftr,
)
train_adj = adjusted[(adjusted.date < SPLIT) & (adjusted.kind == "domestic")]
corrected = OutcomeModel.fit(train_adj.elo_diff_adj, train_adj.ftr)

test_history = history[(history.date >= SPLIT) & (history.kind == "uefa")]
test_adj = adjusted[(adjusted.date >= SPLIT) & (adjusted.kind == "uefa")]

rate = (
    before[before.kind == "domestic"].ftr.value_counts(normalize=True)
    .reindex(metrics.OUTCOMES).to_numpy()
)

results = [
    ("base rate", np.tile(rate, (len(test), 1))),
    ("elo", plain.predict(test_history.elo_diff)),
    ("elo + league strength", corrected.predict(test_adj.elo_diff_adj)),
]

# --- Dixon-Coles, swept over the decay rate --------------------------
reference = train.date.max()

for xi in [0.0, 0.0010, 0.0018, 0.0030, 0.0050]:
    start = time.time()
    model = dc.fit(train, dc.DixonColesConfig(xi=xi), reference=reference)
    elapsed = time.time() - start

    probabilities = np.array(
        [
            [model.predict(h, a)[o] for o in metrics.OUTCOMES]
            for h, a in zip(test.home_key, test.away_key)
        ]
    )

    half_life = np.log(2) / xi / 365.25 if xi else float("inf")
    label = f"dixon-coles xi={xi:.4f}"
    results.append((label, probabilities))

    print(f"  fitted xi={xi:.4f} in {elapsed:5.1f}s  "
          f"half-life {half_life:5.2f}y  converged={model.converged}  "
          f"home_adv={model.home_advantage:.3f}  rho={model.rho:+.3f}")

print(f"\n=== held-out European matches ({len(test):,}) ===")
for name, probabilities in results:
    s = metrics.summary(probabilities, test.ftr)
    print(f"  {name:24s} RPS {s['rps']:.4f}  log-loss {s['log_loss']:.4f}  "
          f"acc {s['accuracy']:.1%}")
