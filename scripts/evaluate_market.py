"""Model against the closing line, and the two together.

The plan asks for three comparisons and only two have ever been run.
The third is the one that matters: a model that merely reproduces the
market is an expensive way to read a price, and the only way to tell
the difference is to blend them and see whether the blend beats the
market on its own.

How the blend weight is chosen is the whole experiment. Sweeping it on
the test rows and reporting the best would guarantee a win and mean
nothing, which is a mistake this project has already made once with an
ensemble weight. So each fold picks its weight on the validation year
that ends where its test season begins, using a model that never saw
that year, and then applies it forward unchanged.

Scope limit, stated because it is severe: football-data.uk prices
domestic leagues only. The Champions League — the thing PitchIQ exists
to forecast — has no odds here, so nothing below says how the model
compares to the market on European football.
"""

import numpy as np
import pandas as pd

from pitchiq import matches
from pitchiq.eval import market, metrics
from pitchiq.features import build as features
from pitchiq.models import boosted
from pitchiq.models import dixon_coles as dc
from pitchiq.models.outcome import OutcomeModel
from sklearn.linear_model import LogisticRegression

FOLDS = [pd.Timestamp(d) for d in ("2023-07-01", "2024-07-01", "2025-07-01")]
END = pd.Timestamp("2026-07-01")
VALIDATION_YEARS = 1
MIN_HISTORY = 10
BOOTSTRAP = 2000
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.05), 2)


def rps_per_match(probabilities: np.ndarray, actual) -> np.ndarray:
    index = np.array([metrics.OUTCOMES.index(a) for a in actual])
    observed = np.eye(3)[index]
    gap = np.cumsum(probabilities, axis=1) - np.cumsum(observed, axis=1)

    return (gap[:, :2] ** 2).sum(axis=1) / 2


def interval(difference: np.ndarray, draws: int = BOOTSTRAP):
    generator = np.random.default_rng(0)
    resampled = [
        difference[generator.integers(0, len(difference), len(difference))].mean()
        for _ in range(draws)
    ]

    return np.percentile(resampled, [2.5, 97.5])


def blend(model: np.ndarray, prices: np.ndarray, weight: float) -> np.ndarray:
    return weight * model + (1.0 - weight) * prices


frame = features.load()
frame = frame[(frame.home_played >= MIN_HISTORY) & (frame.away_played >= MIN_HISTORY)]
columns = features.feature_columns(frame)

priced = market.load()
stream = matches.load()
stream = stream[stream.match_id.isin(frame.match_id)]

print(f"overround on the raw prices: {market.margin():.4f}\n")

collected: dict[str, list[np.ndarray]] = {}
truth: list[np.ndarray] = []
chosen: list[float] = []


def keep(name: str, probabilities: np.ndarray) -> None:
    collected.setdefault(name, []).append(probabilities)


for fold_start, fold_end in zip(FOLDS, FOLDS[1:] + [END]):
    history = frame[frame.date < fold_start]
    season = frame[(frame.date >= fold_start) & (frame.date < fold_end)]

    validation_start = fold_start - pd.DateOffset(years=VALIDATION_YEARS)
    train = history[history.date < validation_start]
    validation = history[history.date >= validation_start]

    # Only rows the market actually priced can take part.
    season = season.merge(priced, on="match_id", how="inner")
    validation_priced = validation.merge(priced, on="match_id", how="inner")

    if season.empty or validation_priced.empty:
        continue

    truth.append(season.ftr.to_numpy())
    prices = market.probabilities(season)
    keep("market closing line", prices)

    rate = history.ftr.value_counts(normalize=True).reindex(metrics.OUTCOMES).to_numpy()
    keep("base rate", np.tile(rate, (len(season), 1)))

    domestic = history[history.kind == "domestic"]
    mapping = OutcomeModel.fit(domestic.elo_diff, domestic.ftr)
    keep("elo", mapping.predict(season.elo_diff))

    goals = dc.fit(
        stream[stream.date < fold_start],
        dc.DixonColesConfig(xi=0.0010, ridge=0.5),
        reference=history.date.max(),
    )
    keep(
        "dixon-coles",
        np.array(
            [
                [goals.predict(h, a)[o] for o in metrics.OUTCOMES]
                for h, a in zip(season.home_key, season.away_key)
            ]
        ),
    )

    # --- the model, and the weight chosen on the validation year -----
    searched = boosted.fit(train, validation, columns)
    model = boosted.refit(history, columns, searched.rounds)

    on_validation = searched.predict(validation_priced)
    validation_prices = market.probabilities(validation_priced)
    validation_actual = validation_priced.ftr.to_numpy()

    scores = [
        metrics.ranked_probability_score(
            blend(on_validation, validation_prices, w), validation_actual
        )
        for w in WEIGHTS
    ]
    weight = float(WEIGHTS[int(np.argmin(scores))])
    chosen.append(weight)

    # How flat is the choice? A weight of zero picked off a curve that
    # barely moves says something different from one picked off a curve
    # that climbs steeply, and only the second is real evidence.
    curve = "  ".join(
        f"w={w:.2f}:{score:.5f}" for w, score in zip(WEIGHTS[:5], scores[:5])
    )
    print(f"    validation curve  {curve}")


    on_season = model.predict(season)
    keep("gradient boosting", on_season)
    keep("model + market", blend(on_season, prices, weight))

    # A single weight can only slide between two sets of probabilities.
    # Stacking in log space can also reshape them — sharpen the market,
    # or lean on the model for one outcome and not the others. If even
    # this gives the model nothing, the answer is not an artefact of how
    # the blend was parameterised.
    stack = LogisticRegression(max_iter=3000)
    stack.fit(
        np.column_stack([np.log(on_validation), np.log(validation_prices)]),
        validation_actual,
    )
    order = [list(stack.classes_).index(o) for o in metrics.OUTCOMES]
    keep(
        "stacked",
        stack.predict_proba(
            np.column_stack([np.log(on_season), np.log(prices)])
        )[:, order],
    )

    model_weight = np.abs(stack.coef_[:, :3]).sum()
    market_weight = np.abs(stack.coef_[:, 3:]).sum()
    print(
        f"    stack leans {model_weight / (model_weight + market_weight):.1%} "
        f"on the model by absolute coefficient"
    )

    print(
        f"{fold_start.date()} to {fold_end.date()}: "
        f"{len(season):,} priced matches, boosting {searched.rounds} rounds, "
        f"blend weight {weight:.2f} on the model"
    )

actual = np.concatenate(truth)
scored = {name: np.vstack(parts) for name, parts in collected.items()}

print(f"\n=== priced domestic matches ({len(actual):,}) ===")
for name, probabilities in scored.items():
    summary = metrics.summary(probabilities, actual)
    print(
        f"  {name:<22} RPS {summary['rps']:.4f}"
        f"   log-loss {summary['log_loss']:.4f}"
        f"   acc {summary['accuracy']:.1%}"
    )

print("\n=== paired differences, positive means the first is better ===")
pairs = [
    ("model + market", "market closing line"),
    ("stacked", "market closing line"),
    ("gradient boosting", "market closing line"),
]

for better, worse in pairs:
    difference = (
        rps_per_match(scored[worse], actual) - rps_per_match(scored[better], actual)
    )
    low, high = interval(difference)
    verdict = "significant" if low > 0 or high < 0 else "NOT significant"
    print(
        f"  {better:<20} vs {worse:<22} {difference.mean():+.5f}"
        f"   95% CI [{low:+.5f}, {high:+.5f}]   {verdict}"
    )

# --- is the market equally sharp everywhere? -------------------------
# If the model has an edge anywhere it is where the money is thinnest,
# so this splits by division rather than reporting one average.
tiers = np.concatenate(
    [
        f[(f.date >= a) & (f.date < b)].merge(priced, on="match_id", how="inner").comp_tier.to_numpy()
        for f, a, b in ((frame, a, b) for a, b in zip(FOLDS, FOLDS[1:] + [END]))
    ]
)

print("\n=== by division tier ===")
print(f"  {'tier':<6}{'matches':>9}{'market':>10}{'boosting':>10}{'gap':>10}")
for tier in sorted(set(tiers[~np.isnan(tiers)])):
    rows = tiers == tier
    if rows.sum() < 500:
        continue
    market_rps = metrics.ranked_probability_score(
        scored["market closing line"][rows], actual[rows]
    )
    model_rps = metrics.ranked_probability_score(
        scored["gradient boosting"][rows], actual[rows]
    )
    print(
        f"  {int(tier):<6}{int(rows.sum()):>9,}{market_rps:>10.4f}"
        f"{model_rps:>10.4f}{model_rps - market_rps:>+10.4f}"
    )

print(f"\nweights chosen per fold: {chosen}")
print(
    "a weight above zero means the validation year preferred to keep some "
    "of the model; whether that helped is the first line above."
)
