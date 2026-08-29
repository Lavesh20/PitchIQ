"""Is a stated probability worth what it says?

The plan sets the test in one sentence (§23): *if PitchIQ predicts 70%
for a class repeatedly, the observed frequency of that class should be
close to 70%*. That is calibration, and it is a different property from
accuracy. A model can rank matches perfectly and still be badly
calibrated — say 80% whenever it means 65% — and it will score well on
anything that only cares about ordering while being useless for
anything that acts on the number itself.

Which is exactly our situation. The simulator multiplies probabilities
together across eight matchdays and 10,000 seasons. If each one is a
few points too confident, the compounded error is what puts Arsenal at
20.4% to win a competition the market prices at 15-18%.

Nothing here fixes anything. It measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import OUTCOMES, _as_index


def reliability(
    probabilities: np.ndarray,
    actual,
    bins: int = 10,
    outcome: str | None = None,
) -> pd.DataFrame:
    """Predicted probability against observed frequency, in bins.

    Every forecast/outcome pair is one observation: a three-way forecast
    over N matches gives 3N of them. Pooling the outcomes is deliberate
    — a claim of "70%" should mean the same thing whether it is about a
    home win or a draw, and splitting by outcome first hides a model
    that is confident about favourites and timid about draws.

    Pass ``outcome`` to look at one of them on its own.
    """
    index = _as_index(actual)
    observed = np.zeros_like(probabilities)
    observed[np.arange(len(index)), index] = 1.0

    if outcome is not None:
        column = OUTCOMES.index(outcome)
        stated = probabilities[:, column]
        happened = observed[:, column]
    else:
        stated = probabilities.reshape(-1)
        happened = observed.reshape(-1)

    edges = np.linspace(0.0, 1.0, bins + 1)
    # ``digitize`` puts an exact 1.0 in a bin of its own; fold it back.
    slot = np.clip(np.digitize(stated, edges[1:-1]), 0, bins - 1)

    rows = []
    for b in range(bins):
        inside = slot == b

        if not inside.any():
            continue

        rows.append(
            {
                "bin": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
                "n": int(inside.sum()),
                "predicted": float(stated[inside].mean()),
                "observed": float(happened[inside].mean()),
            }
        )

    table = pd.DataFrame(rows)
    table["gap"] = table["observed"] - table["predicted"]

    return table


def expected_calibration_error(
    probabilities: np.ndarray,
    actual,
    bins: int = 10,
    outcome: str | None = None,
) -> float:
    """One number for how far the stated probabilities are from the truth.

    The bin-count weighted mean absolute gap. Zero is perfect. It is
    worth reading next to the table rather than instead of it, because a
    small average can hide a large error in the confident bins, which
    are the ones the simulator leans on hardest.
    """
    table = reliability(probabilities, actual, bins, outcome)

    if table.empty:
        return float("nan")

    return float(
        np.average(table["gap"].abs(), weights=table["n"])
    )


def sharpness(probabilities: np.ndarray) -> float:
    """How far from a shrug the forecasts are, on average.

    Reported alongside calibration because the two trade off. Always
    predicting the base rate is perfectly calibrated and worthless, so
    "well calibrated" is only a compliment when sharpness holds up.
    """
    return float(np.mean(probabilities.max(axis=1)))


def confidence_gap(probabilities: np.ndarray, actual) -> float:
    """Mean stated confidence in the favourite, minus how often it won.

    Positive means over-confident: the model claims more than it
    delivers. This is the single number that speaks to the Arsenal
    problem, and it is the one to watch after any attempt to fix it.
    """
    index = _as_index(actual)
    picked = probabilities.argmax(axis=1)

    return float(probabilities.max(axis=1).mean() - (picked == index).mean())
