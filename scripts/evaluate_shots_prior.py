"""Do shot-based ratings help the goals model? No, and the reason matters.

Shots on target are a steadier read of team strength than goals: four
times as many events, and a side creating chances it fails to convert
looks weak on goals alone. Fitting the same attack and defence structure
to shots and using it as the prior the goals model shrinks toward is a
standard idea and a reasonable one.

It does not work here, and the script exists to show why rather than to
leave the idea untried on the backlog. The failure is not in the fit --
it is that football-data.co.uk records shots for the big divisions from
the mid-2010s on, which are the clubs whose goal records are already
long. The clubs a prior would actually rescue have no shot data at all.
"""

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.eval import metrics
from pitchiq.models import dixon_coles as dc
from pitchiq.models import shots

CUT = pd.Timestamp("2023-07-01")
THIN = 60
BOOTSTRAP = 2000
CONFIG = dc.DixonColesConfig(xi=0.0010, ridge=0.5)


def rps_per_match(model, frame) -> np.ndarray:
    predicted = np.array(
        [
            [model.predict(h, a)[o] for o in metrics.OUTCOMES]
            for h, a in zip(frame.home_key, frame.away_key)
        ]
    )
    index = np.array([metrics.OUTCOMES.index(o) for o in frame.ftr])
    observed = np.eye(3)[index]

    return ((np.cumsum(predicted, 1) - np.cumsum(observed, 1))[:, :2] ** 2).sum(1) / 2


frame = matches.load(stats=True)
train, test = frame[frame.date < CUT], frame[frame.date >= CUT]

prior = shots.fit(train, CONFIG)

if prior is None:
    raise SystemExit("not enough shot data to fit a prior")

print(
    f"shot ratings fitted on {prior.fitted_on:,} matches\n"
    f"one unit of shot rating is worth {prior.scale:.3f} of goal rating\n"
)

plain = dc.fit(train, CONFIG, reference=train.date.max())
guided = dc.fit(train, CONFIG, reference=train.date.max(), prior=prior.as_result())

counts = pd.concat([train.home_key, train.away_key]).value_counts()
thin = test[
    (test.home_key.map(counts).fillna(0) < THIN)
    | (test.away_key.map(counts).fillna(0) < THIN)
]

generator = np.random.default_rng(0)

print(f"  {'window':<12}{'n':>8}{'plain':>9}{'prior':>9}{'gain':>11}{'95% CI':>24}")
for label, window in (
    ("European", test[test.kind == "uefa"]),
    ("domestic", test[test.kind == "domestic"]),
    ("thin data", thin),
):
    without = rps_per_match(plain, window)
    with_prior = rps_per_match(guided, window)
    difference = without - with_prior

    resampled = [
        difference[generator.integers(0, len(difference), len(difference))].mean()
        for _ in range(BOOTSTRAP)
    ]
    low, high = np.percentile(resampled, [2.5, 97.5])

    print(
        f"  {label:<12}{len(window):>8,}{without.mean():>9.4f}{with_prior.mean():>9.4f}"
        f"{difference.mean():>+11.5f}   [{low:+.5f}, {high:+.5f}]"
    )

# --- why ------------------------------------------------------------
print("\n=== where the shot data actually is ===")
clubs = set(test.home_key) | set(test.away_key)
have = set(prior.attack)

print(f"  {'training matches':<20}{'clubs':>8}{'have a prior':>14}")
for low_, high_ in ((0, 20), (20, 60), (60, 150), (150, 400), (400, 10 ** 6)):
    band = {c for c in clubs if low_ <= counts.get(c, 0) < high_}

    if not band:
        continue

    print(
        f"  {f'{low_}-{high_ if high_ < 10 ** 6 else ''}':<20}"
        f"{len(band):>8}{len(band & have) / len(band):>13.0%}"
    )

european = test[test.kind == "uefa"]
needy = {
    c
    for c in set(european.home_key) | set(european.away_key)
    if counts.get(c, 0) < THIN
}

print(
    f"\n  {len(needy)} clubs in the European field have fewer than {THIN} matches"
    f" behind them.\n  {len(needy & have)} of them have shot data."
)
print(
    "\n  The prior covers the clubs that least need it. That is a property of\n"
    "  the source, not of the method: any better-measured signal that did\n"
    "  reach the thin clubs would go straight into the same argument."
)
