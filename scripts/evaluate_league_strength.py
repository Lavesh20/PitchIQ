"""Does correcting for league strength actually predict better?

Everything is fitted before the split and scored after it.
"""

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.eval import metrics
from pitchiq.models import elo, league_strength
from pitchiq.models.outcome import OutcomeModel

SPLIT = "2023-07-01"

frame = matches.load()
result = elo.fit(frame)
history = result.history

before = history[history.date < SPLIT]
train_domestic = before[before.kind == "domestic"]
train_uefa = before[before.kind == "uefa"]
test_uefa = history[(history.date >= SPLIT) & (history.kind == "uefa")]

means = league_strength.league_means(before)
strength = league_strength.fit(train_uefa, means)
adjusted = league_strength.apply(history, strength)

train_adj = adjusted[(adjusted.date < SPLIT) & (adjusted.kind == "domestic")]
test_adj = adjusted[(adjusted.date >= SPLIT) & (adjusted.kind == "uefa")]

plain = OutcomeModel.fit(train_domestic.elo_diff, train_domestic.ftr)
corrected = OutcomeModel.fit(train_adj.elo_diff_adj, train_adj.ftr)

rate = train_domestic.ftr.value_counts(normalize=True).reindex(metrics.OUTCOMES).to_numpy()

print(f"offsets and scales fitted on {len(train_uefa):,} European matches "
      f"before {SPLIT}\n")
print(f"=== held-out European matches ({len(test_uefa):,}) ===")
for name, probabilities in [
    ("base rate", np.tile(rate, (len(test_uefa), 1))),
    ("elo", plain.predict(test_uefa.elo_diff)),
    ("elo + league strength", corrected.predict(test_adj.elo_diff_adj)),
]:
    s = metrics.summary(probabilities, test_uefa.ftr)
    print(f"  {name:22s} RPS {s['rps']:.4f}  log-loss {s['log_loss']:.4f}  "
          f"acc {s['accuracy']:.1%}")

table = strength.table()
solid = table[table.european_matches >= 40]

print("\n=== league corrections (40+ European matches) ===")
print(solid.to_string(index=False))

# What happens to the clubs that looked wrong.
country_of = {}
for col, ccol in [("home_key", "home_cc"), ("away_key", "away_cc")]:
    country_of.update(dict(zip(frame[col], frame[ccol])))

final = pd.DataFrame(
    sorted(result.ratings.items(), key=lambda kv: -kv[1]), columns=["club", "elo"]
)
final["cc"] = final.club.map(country_of)
final["adjusted"] = strength.adjust(final.elo.to_numpy(), list(final.cc))
final["old_rank"] = final.elo.rank(ascending=False).astype(int)
final["new_rank"] = final.adjusted.rank(ascending=False).astype(int)

print("\n=== clubs that looked wrong ===")
watch = ["SCO:celtic", "ENG:bournemouth", "Zenit", "ENG:brentford",
         "Ath Madrid", "NOR:bodo glimt", "ITA:roma", "Sp Lisbon"]
print(final[final.club.isin(watch)][
    ["club", "cc", "elo", "adjusted", "old_rank", "new_rank"]
].to_string(index=False))

print("\n=== top 20 after correction ===")
print(final.sort_values("adjusted", ascending=False)
      .head(20)[["club", "cc", "elo", "adjusted", "old_rank"]]
      .to_string(index=False))
