"""Are the stated probabilities worth what they say?

The plan sets the test in one sentence (§23): if PitchIQ predicts 70%
for a class repeatedly, that class should happen about 70% of the time.
It is the last of the plan's eight V1 criteria that has never been
checked, and it is a different property from accuracy — a model can
rank matches perfectly and still overstate every number it prints.

This project has a specific reason to suspect it does. The simulator
compounds probabilities across eight matchdays and 10,000 seasons, so a
few points of over-confidence per match becomes Arsenal at 20.4% to win
a competition the market prices at 15-18%.
"""

import numpy as np
import pandas as pd

from pitchiq.config import DATA
from pitchiq.eval import backtest, calibration, metrics

BINS = 10
FIGURE = DATA / "figures" / "reliability.png"

frame = backtest.load()
actual = frame.ftr.to_numpy()

sources = {name: backtest.probabilities(frame, name) for name in backtest.MODELS}

priced = frame.priced.to_numpy()
market_probabilities = frame.loc[
    priced, ["market_home", "market_draw", "market_away"]
].to_numpy()

print(f"held-out matches {len(frame):,}, of which {int(priced.sum()):,} priced\n")

print(f"  {'model':<14}{'RPS':>8}{'Brier':>8}{'ECE':>8}{'sharp':>8}{'over-conf':>11}")
for name, probabilities in sources.items():
    print(
        f"  {name:<14}"
        f"{metrics.ranked_probability_score(probabilities, actual):>8.4f}"
        f"{metrics.brier_score(probabilities, actual):>8.4f}"
        f"{calibration.expected_calibration_error(probabilities, actual, BINS):>8.4f}"
        f"{calibration.sharpness(probabilities):>8.3f}"
        f"{calibration.confidence_gap(probabilities, actual):>+11.4f}"
    )

market_actual = actual[priced]
print(
    f"  {'market':<14}"
    f"{metrics.ranked_probability_score(market_probabilities, market_actual):>8.4f}"
    f"{metrics.brier_score(market_probabilities, market_actual):>8.4f}"
    f"{calibration.expected_calibration_error(market_probabilities, market_actual, BINS):>8.4f}"
    f"{calibration.sharpness(market_probabilities):>8.3f}"
    f"{calibration.confidence_gap(market_probabilities, market_actual):>+11.4f}"
)
print(
    "\n  ECE: mean gap between what was claimed and what happened, 0 is perfect."
    "\n  sharp: mean confidence in the favourite. Always saying 33% would be"
    "\n         perfectly calibrated and useless, so read the two together."
    "\n  over-conf: claimed confidence minus how often the favourite won."
    "\n             Positive means the model promises more than it delivers."
)

# --- where the error sits, which the single number hides -------------
for name in ("boosting", "dixon_coles"):
    print(f"\n=== {name}: predicted against observed ===")
    table = calibration.reliability(sources[name], actual, BINS)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n=== European matches only, the ones the simulator runs on ===")
european = (frame.kind == "uefa").to_numpy()
print(f"  {'model':<14}{'RPS':>8}{'ECE':>8}{'over-conf':>11}")
for name, probabilities in sources.items():
    print(
        f"  {name:<14}"
        f"{metrics.ranked_probability_score(probabilities[european], actual[european]):>8.4f}"
        f"{calibration.expected_calibration_error(probabilities[european], actual[european], 5):>8.4f}"
        f"{calibration.confidence_gap(probabilities[european], actual[european]):>+11.4f}"
    )

# --- the picture the plan asks for -----------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(6, 6))
    axes.plot([0, 1], [0, 1], color="0.7", linestyle="--", label="perfect")

    for name in ("boosting", "dixon_coles", "elo"):
        table = calibration.reliability(sources[name], actual, BINS)
        axes.plot(table.predicted, table.observed, marker="o", label=name)

    table = calibration.reliability(market_probabilities, market_actual, BINS)
    axes.plot(table.predicted, table.observed, marker="s", color="black", label="market")

    axes.set_xlabel("probability stated")
    axes.set_ylabel("frequency observed")
    axes.set_title(f"Reliability, {len(frame):,} held-out matches")
    axes.legend()
    axes.grid(alpha=0.3)

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=140, bbox_inches="tight")
    print(f"\nreliability diagram written to {FIGURE.relative_to(DATA.parent)}")
except ImportError:
    print("\nmatplotlib not available; skipped the diagram")
