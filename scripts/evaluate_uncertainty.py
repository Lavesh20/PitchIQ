"""Does sampling the ratings make the forecast honest?

The claim being tested is specific and was measured before any of this
was built: across 4,225 club-seasons, where clubs actually finished
regressed on where we predicted they would finish with a slope of
0.828, not 1.0. Predictions spread about a fifth wider than reality
supports — good clubs pushed too far up, weak ones too far down.

The suspected cause is that the simulator treats fitted ratings as
facts. Arsenal's attack is 1.04 in all 10,000 simulated seasons, and
so is the rating of a club we have seen twenty times.

So this is an A/B on exactly that. Same seasons, same fixtures, same
number of simulated seasons, same random seed. The only difference is
whether every season is played with one set of ratings or with ratings
drawn from the bootstrap. If the diagnosis is right the slope moves
toward 1.0; if it does not move, the diagnosis was wrong and the honest
thing is to report that.
"""

import time

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.config import DATA
from pitchiq.models import dixon_coles as dc
from pitchiq.models.uncertainty import ParameterSamples
from pitchiq.sim import league

CUTOFFS = [pd.Timestamp(d) for d in
           ("2020-07-01", "2021-07-01", "2022-07-01", "2024-07-01")]
RUNS = 2000
PARAMETER_DRAWS = 40
LEVELS = (0.5, 0.8, 0.9)
MIN_CLUBS = 10
CONFIG = dc.DixonColesConfig(xi=0.0010, ridge=0.5)


def slope(table: pd.DataFrame) -> tuple[float, float, float]:
    """Regress actual finishing position on predicted, within-league scale.

    Positions are put on a 0-1 scale inside each league so a 20-club
    division and a 12-club one contribute comparably. A calibrated
    forecast gives 1.0. Below 1.0 means the predictions are spread wider
    than the results.
    """
    x = ((table.expected - 1) / (table.clubs - 1)).to_numpy()
    y = ((table.actual - 1) / (table.clubs - 1)).to_numpy()

    fitted = np.polyfit(x, y, 1)[0]

    generator = np.random.default_rng(0)
    resampled = [
        np.polyfit(x[pick], y[pick], 1)[0]
        for pick in (generator.integers(0, len(x), len(x)) for _ in range(400))
    ]

    return fitted, *np.percentile(resampled, [2.5, 97.5])


frame = matches.load()
domestic = frame[frame.kind == "domestic"]

sizes = domestic.groupby(["competition", "season"]).agg(
    played=("match_id", "size"),
    clubs=("home_key", "nunique"),
    start=("date", "min"),
)
complete = sizes[
    (sizes.played == sizes.clubs * (sizes.clubs - 1)) & (sizes.clubs >= MIN_CLUBS)
]

rows = []
started = time.time()

for cutoff in CUTOFFS:
    path = DATA / "models" / f"uncertainty_{cutoff.date()}.npz"

    if not path.exists():
        print(f"  no bootstrap for {cutoff.date()}; run scripts/bootstrap_uncertainty.py")
        continue

    samples = ParameterSamples.load(path, CONFIG)
    quarter = complete[complete.start.dt.to_period("Q").dt.start_time == cutoff]

    print(
        f"\n{cutoff.date()}: {len(quarter)} league-seasons, "
        f"{samples.draws} parameter draws",
        flush=True,
    )

    for (competition, season), meta in quarter.iterrows():
        season_rows = domestic[
            (domestic.competition == competition) & (domestic.season == season)
        ].sort_values("date")

        actual = league.table(season_rows)

        for label, model in (("fixed", samples.point), ("sampled", samples)):
            keys, placing = league.positions(
                model,
                season_rows,
                runs=RUNS,
                seed=0,
                parameter_draws=PARAMETER_DRAWS,
            )
            result = league.coverage(keys, placing, actual, LEVELS)
            result["ratings"] = label
            result["competition"] = competition
            result["season"] = season
            result["clubs"] = len(keys)
            rows.append(result)

out = pd.concat(rows, ignore_index=True)
out.to_parquet("data/processed/uncertainty_coverage.parquet", index=False)

at = out[out.level == 0.9]
club_seasons = len(at) // 2
print(f"\nsimulated {club_seasons:,} club-seasons both ways in {time.time() - started:.0f}s")

print("\n=== interval coverage ===")
print(f"  {'ratings':<10}{'50%':>8}{'80%':>8}{'90%':>8}")
for label, part in out.groupby("ratings"):
    line = f"  {label:<10}"
    for level in LEVELS:
        line += f"{part[part.level == level].inside.mean():>8.1%}"
    print(line)

print("\n=== the headline: actual against predicted ===")
print(f"  {'ratings':<10}{'slope':>8}{'95% CI':>20}")
for label, part in at.groupby("ratings"):
    fitted, low, high = slope(part)
    print(f"  {label:<10}{fitted:>8.3f}   [{low:.3f}, {high:.3f}]")
print("  1.000 is calibrated; below 1.000 means predictions too spread out")

print("\n=== 90% coverage by predicted position ===")
print(f"  {'band':<12}{'fixed':>10}{'sampled':>10}{'n':>8}")
for lo, hi in ((1, 3), (3, 6), (6, 10), (10, 14), (14, 25)):
    band = at[(at.expected >= lo) & (at.expected < hi)]
    if band.empty:
        continue
    fixed = band[band.ratings == "fixed"]
    sampled = band[band.ratings == "sampled"]
    print(
        f"  {f'{lo}-{hi}':<12}{fixed.inside.mean():>10.1%}"
        f"{sampled.inside.mean():>10.1%}{len(fixed):>8,}"
    )

print("\n=== mean interval width, in places ===")
for label, part in at.groupby("ratings"):
    width = (part.high - part.low + 1).mean()
    print(f"  {label:<10}{width:>6.1f}")
