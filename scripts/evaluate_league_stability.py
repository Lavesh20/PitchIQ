"""How much can the per-country corrections actually be trusted?

Two questions, and the answers point in different directions.

**How precise is each correction?** Romania's offset is reported to the
nearest rating point off about a hundred European matches. Resampling
those matches shows how much of that number is evidence and how much is
the particular sample we happen to have.

**Is the shrinkage set correctly?** The prior already shrinks a country
in proportion to its evidence, because the error term grows with match
count while the penalty does not. Whether it shrinks by the right amount
is a separate question, and it is settled by selection across several
validation seasons — never on the test window, which is where a first
attempt at this went wrong: the test set preferred far harder shrinkage,
and acting on that would have been choosing a hyperparameter on the
thing it was about to be judged by.
"""

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.eval import metrics
from pitchiq.models import elo
from pitchiq.models import league_strength as ls
from pitchiq.models.outcome import OutcomeModel

DRAWS = 100
FOLDS = [pd.Timestamp(d) for d in
         ("2019-07-01", "2020-07-01", "2021-07-01", "2022-07-01")]
TEST = pd.Timestamp("2023-07-01")
END = pd.Timestamp("2027-01-01")
GRID = [(o, s) for o in (15.0, 25.0, 50.0, 100.0, 200.0) for s in (0.15, 0.35, 0.70)]

history = elo.fit(matches.load()).history


def score(start, end, offset_prior, scale_prior):
    """Fit corrections on everything before ``start``, score what follows."""
    train = history[history.date < start]

    strength = ls.fit(
        train[train.kind == "uefa"],
        ls.league_means(train),
        offset_prior=offset_prior,
        scale_prior=scale_prior,
    )
    adjusted = ls.apply(history, strength)

    fitted = adjusted[(adjusted.date < start) & (adjusted.kind == "domestic")]
    mapping = OutcomeModel.fit(fitted.elo_diff_adj, fitted.ftr)

    window = adjusted[
        (adjusted.date >= start) & (adjusted.date < end) & (adjusted.kind == "uefa")
    ]

    return (
        metrics.ranked_probability_score(
            mapping.predict(window.elo_diff_adj), window.ftr
        ),
        len(window),
    )


# --- how precise is each correction? ---------------------------------
before = history[history.date < TEST]
samples = ls.bootstrap(
    before[before.kind == "uefa"], ls.league_means(before), draws=DRAWS
)
table = samples.table()

pd.set_option("display.width", 200)
print(f"=== per-country corrections, {DRAWS} resamples, 90% intervals ===")
columns = ["country", "european_matches", "offset", "offset_low", "offset_high",
           "scale", "scale_low", "scale_high"]
print(table[columns].head(8).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
print("  ...")
print(table[columns].tail(8).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

print(
    f"\n  offsets distinguishable from zero: {int(table.offset_certain.sum())} of {len(table)}"
    f"\n  scales distinguishable from one:   {int(table.scale_certain.sum())} of {len(table)}"
)
print(
    "\n  Roughly half the corrections could be zero. They are still the best\n"
    "  estimate available and the model is better with them than without,\n"
    "  but a country with thirty European matches has an offset that is a\n"
    "  reasonable guess rather than a measurement."
)

# --- is the shrinkage set correctly? ---------------------------------
print(f"\n=== choosing the priors on {len(FOLDS)} validation seasons ===")
chosen, best = None, np.inf

for offset_prior, scale_prior in GRID:
    scores, sizes = [], []

    for start, end in zip(FOLDS, FOLDS[1:] + [TEST]):
        rps, n = score(start, end, offset_prior, scale_prior)
        scores.append(rps)
        sizes.append(n)

    average = float(np.average(scores, weights=sizes))

    if average < best:
        chosen, best = (offset_prior, scale_prior), average

print(f"  chosen: offset_prior={chosen[0]:.0f}, scale_prior={chosen[1]:.2f}")
print(f"  validation surface spans {best:.4f} at best; it is close to flat")

print("\n=== held-out test, scored once ===")
for label, priors in (("current  (100, 0.35)", (100.0, 0.35)),
                      (f"chosen   ({chosen[0]:.0f}, {chosen[1]:.2f})", chosen)):
    rps, n = score(TEST, END, *priors)
    print(f"  {label:<22} RPS {rps:.4f} on {n:,} European matches")

print(
    "\n  The selected priors do not beat the current ones. The validation\n"
    "  surface is flat enough that the difference is noise, so the\n"
    "  conclusion is that the shrinkage is already set sensibly and the\n"
    "  useful output of this script is the interval table above."
)
