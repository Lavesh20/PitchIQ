"""Are the simulator's ranges honest, over thousands of seasons?

The 2025/26 backtest put 33 of 36 clubs inside their 90% interval. That
reads as 92% and it settles nothing: 36 observations cannot separate a
well-calibrated forecast from a noticeably over-confident one, and the
whole question about the tournament forecast — whether it puts too much
probability on the favourites — turns on exactly that distinction.

UEFA's format change makes the obvious fix impossible. Only 2024/25 and
2025/26 used the 36-club league phase, so repeating the European
backtest gets us from 36 observations to 72.

Domestic leagues are the same shape of problem and the record holds
hundreds of complete seasons. Each one is simulated from a model that
knew nothing beyond its first fixture, and we count how often the truth
lands inside the range we stated.

What that does *not* establish: a domestic league is the easy case,
played between clubs who meet twice and whose ratings are settled. Good
coverage here is strong evidence, not proof, that European ranges are
honest too.
"""

import time

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.models import dixon_coles as dc
from pitchiq.sim import league

SINCE = "2015-07-01"
RUNS = 2000
LEVELS = (0.5, 0.8, 0.9)
MIN_CLUBS = 10

frame = matches.load()
domestic = frame[frame.kind == "domestic"]

# A complete double round-robin: every club hosts every other once. A
# partial season would give the simulator fixtures the real table never
# played, and the comparison would be against a different competition.
grouped = domestic.groupby(["competition", "season"])
sizes = grouped.agg(
    played=("match_id", "size"),
    clubs=("home_key", "nunique"),
    start=("date", "min"),
)
complete = sizes[
    (sizes.played == sizes.clubs * (sizes.clubs - 1))
    & (sizes.clubs >= MIN_CLUBS)
    & (sizes.start >= SINCE)
]

print(f"{len(complete):,} complete league-seasons, {complete.clubs.sum():,} club-seasons\n")

# One Dixon-Coles fit per quarter rather than per season: a fit costs
# about nine seconds and most leagues start within weeks of each other.
# The cutoff is floored, never rounded, so every fit stays strictly
# behind the season it is used for.
fits: dict[pd.Timestamp, object] = {}
rows = []
started = time.time()

for (competition, season), meta in complete.iterrows():
    season_rows = domestic[
        (domestic.competition == competition) & (domestic.season == season)
    ].sort_values("date")

    cutoff = meta.start.to_period("Q").start_time

    if cutoff not in fits:
        history = frame[frame.date < cutoff]

        if len(history) < 20000:
            continue

        fits[cutoff] = dc.fit(
            history,
            dc.DixonColesConfig(xi=0.0010, ridge=0.5),
            reference=history.date.max(),
        )
        print(f"  fitted to {cutoff.date()} ({len(fits)} fits, {time.time() - started:.0f}s)")

    model = fits[cutoff]

    keys, placing = league.positions(model, season_rows, runs=RUNS)
    actual = league.table(season_rows)
    result = league.coverage(keys, placing, actual, LEVELS)

    result["competition"] = competition
    result["season"] = season
    result["clubs"] = len(keys)
    # A club the model has never rated is treated as exactly average,
    # which is a real limitation for promoted sides and worth counting
    # separately rather than letting it hide in the average.
    result["rated"] = result.club.isin(model.attack)

    rows.append(result)

out = pd.concat(rows, ignore_index=True)
print(f"\nsimulated {out.season.nunique()} seasons in {time.time() - started:.0f}s\n")

print("=== interval coverage ===")
print(f"  {'stated':>8}{'observed':>10}{'club-seasons':>14}{'verdict':>16}")
for level in LEVELS:
    at = out[out.level == level]
    observed = at.inside.mean()
    # Binomial standard error on the observed share, so a gap can be
    # read against what sampling alone would produce.
    error = np.sqrt(observed * (1 - observed) / len(at))
    gap = observed - level
    verdict = (
        "honest" if abs(gap) < 2 * error
        else ("too narrow" if gap < 0 else "too wide")
    )
    print(f"  {level:>8.0%}{observed:>10.1%}{len(at):>14,}{verdict:>16}")

print("\n=== 90% interval, split by whether the model had rated the club ===")
at = out[out.level == 0.9]
for rated, part in at.groupby("rated"):
    label = "rated" if rated else "never seen before"
    print(f"  {label:<20} {part.inside.mean():>6.1%}  n={len(part):,}")

print("\n=== 90% interval, by league size ===")
for size, part in at.groupby(pd.cut(at.clubs, [9, 13, 17, 21, 30])):
    if part.empty:
        continue
    print(f"  {str(size):<12} {part.inside.mean():>6.1%}  n={len(part):,}")

print("\n=== how far off, in places ===")
at = at.assign(error=(at.actual - at.expected))
print(f"  mean absolute error {at.error.abs().mean():.2f} positions")
print(f"  median              {at.error.abs().median():.2f}")
print(f"  bias                {at.error.mean():+.3f}  (positive = clubs finish lower than predicted)")

print("\n=== the favourites specifically ===")
# If the simulator is over-confident anywhere it is on the clubs it
# expects to win, which is the claim the tournament forecast rests on.
top = at[at.expected <= 3.0]
print(f"  clubs predicted to finish top 3: {len(top):,}")
print(f"  landed inside their 90% interval: {top.inside.mean():.1%}")
print(f"  mean predicted {top.expected.mean():.2f}, mean actual {top.actual.mean():.2f}")

out.to_parquet("data/processed/simulator_coverage.parquet", index=False)
print("\nwritten to data/processed/simulator_coverage.parquet")
