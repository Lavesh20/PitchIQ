"""Replay 2025/26 with a model that has never seen it.

Every earlier number measured one match at a time. This tests the whole
chain end to end: train, forecast, simulate a season, and compare the
distribution against what actually happened.
"""

import numpy as np
import pandas as pd

from pitchiq import clubs, matches
from pitchiq.config import DATA
from pitchiq.models import dixon_coles as dc
from pitchiq.sim import tournament

CUTOFF = pd.Timestamp("2025-07-01")
RUNS = 10000

frame = matches.load()
uefa = pd.read_parquet(DATA / "processed" / "uefa_matches.parquet")

season = uefa[(uefa.season == "2025/26") & (uefa.competition == "UCL")]
league = season[season.stage == "LEAGUE_PHASE"].copy()

league["home_key"] = [clubs.resolve(n, c) for n, c in zip(league.home_team, league.home_country)]
league["away_key"] = [clubs.resolve(n, c) for n, c in zip(league.away_team, league.away_country)]

keys = sorted(set(league.home_key) | set(league.away_key))
slot = {k: i for i, k in enumerate(keys)}

print(f"2025/26 league phase: {len(league)} fixtures, {len(keys)} clubs")
print(f"training on everything before {CUTOFF.date()}")

model = dc.fit(
    frame[frame.date < CUTOFF],
    dc.DixonColesConfig(xi=0.0010, ridge=0.5, home_advantage_by=None),
    reference=CUTOFF,
)

fixtures = pd.DataFrame({
    "home": [slot[k] for k in league.home_key],
    "away": [slot[k] for k in league.away_key],
})

simulation = tournament.run(model, keys, fixtures, runs=RUNS, seed=2)

# --- what actually happened ------------------------------------------
points, scored, conceded = {}, {}, {}
for k in keys:
    points[k] = scored[k] = conceded[k] = 0

for _, m in league.iterrows():
    h, a, hg, ag = m.home_key, m.away_key, int(m.fthg), int(m.ftag)
    scored[h] += hg; conceded[h] += ag
    scored[a] += ag; conceded[a] += hg
    points[h] += 3 if hg > ag else (1 if hg == ag else 0)
    points[a] += 3 if ag > hg else (1 if hg == ag else 0)

actual = pd.DataFrame({
    "key": keys,
    "actual_points": [points[k] for k in keys],
    "gd": [scored[k] - conceded[k] for k in keys],
    "gf": [scored[k] for k in keys],
}).sort_values(["actual_points", "gd", "gf"], ascending=False).reset_index(drop=True)
actual["actual_position"] = actual.index + 1

predicted = pd.DataFrame({
    "key": keys,
    "pred_points": simulation.points.mean(axis=0),
    "pred_position": simulation.position.mean(axis=0),
    "top_8": (simulation.position <= 8).mean(axis=0),
    "wins_it": simulation.reached["wins_it"].mean(axis=0),
})

joined = actual.merge(predicted, on="key")

print(f"\ncorrelation, predicted vs actual points:   "
      f"{np.corrcoef(joined.pred_points, joined.actual_points)[0,1]:.3f}")
print(f"correlation, predicted vs actual position: "
      f"{np.corrcoef(joined.pred_position, joined.actual_position)[0,1]:.3f}")
print(f"mean absolute position error:              "
      f"{(joined.pred_position - joined.actual_position).abs().mean():.1f} places")

print("\n=== actual top 12, and what we would have said ===")
show = joined.sort_values("actual_position").head(12)
print(show[["key", "actual_position", "actual_points", "pred_position",
            "pred_points", "top_8", "wins_it"]].to_string(index=False))

print("\n=== the champion ===")
psg = joined[joined.key == "Paris SG"]
if len(psg):
    r = psg.iloc[0]
    print(f"  PSG actually finished {int(r.actual_position)}th in the league phase "
          f"and won the trophy.")
    print(f"  we gave them {r.wins_it:.1%} to win it, "
          f"and predicted {r.pred_position:.1f} as their league position.")

# How surprising was the real table under our distribution?
inside = 0
for i, k in enumerate(keys):
    lo, hi = np.percentile(simulation.position[:, i], [5, 95])
    truth = int(joined.loc[joined.key == k, "actual_position"].iloc[0])
    inside += lo <= truth <= hi
print(f"\nactual positions inside our 90% range: {inside}/{len(keys)} "
      f"({inside/len(keys):.0%})  -- around 90% means well calibrated")
