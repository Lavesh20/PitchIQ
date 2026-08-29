"""Simulate the 2026/27 Champions League and save the results."""

import time

import pandas as pd

from pitchiq import clubs
from pitchiq.config import DATA
from pitchiq.models import store
from pitchiq.sim import tournament

RUNS = 10000

model = store.load()

squad = pd.read_csv(DATA / "external" / "ucl_2026_27_clubs.csv")
pairs = pd.read_csv(DATA / "external" / "ucl_2026_27_pairings.csv")

squad["key"] = [clubs.resolve(c, cc) for c, cc in zip(squad.club, squad.country)]
keys = list(squad.key)
slot = {club: i for i, club in enumerate(squad.club)}

fixtures = pd.DataFrame(
    {
        "home": [slot[h] for h in pairs.home],
        "away": [slot[a] for a in pairs.away],
    }
)

print(f"simulating {RUNS:,} seasons of the 2026/27 Champions League")
print(f"  {len(squad)} clubs, {len(fixtures)} league-phase fixtures")

started = time.time()
simulation = tournament.run(model, keys, fixtures, runs=RUNS, seed=1)
print(f"  done in {time.time() - started:.1f}s\n")

table = simulation.summary()
table.insert(0, "club", table.pop("club").map(dict(zip(squad.key, squad.club))))
table = table.merge(squad[["club", "country", "pot"]], on="club", how="left")

order = ["club", "country", "pot", "avg_points", "avg_position", "top_8",
         "top_24", "last_16", "quarter_finals", "semi_finals", "final", "wins_it"]
table = table[order]

out = DATA / "predictions"
out.mkdir(parents=True, exist_ok=True)
path = out / "ucl_2026_27_simulation.csv"
table.to_csv(path, index=False)

shown = table.copy()
for column in ["top_8", "top_24", "last_16", "quarter_finals",
               "semi_finals", "final", "wins_it"]:
    shown[column] = (shown[column] * 100).round(1)
shown["avg_points"] = shown["avg_points"].round(1)
shown["avg_position"] = shown["avg_position"].round(1)

print(shown.to_string(index=False))
print(f"\nsaved -> {path}")
