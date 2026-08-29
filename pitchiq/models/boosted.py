"""Gradient-boosted trees over the pre-match feature layer.

The structural models each impose a shape on the problem. Elo says
strength is one number. Dixon-Coles says goals are Poisson with an
attack and a defence term. Both shapes are good, which is why they
work, but neither can express "a short rest hurts, and it hurts more
after a European away leg". A tree expresses that as two nested splits.

So this is the model that earns its place only if the interactions are
real. It is measured against Dixon-Coles on the same rows, and the
difference is reported with an interval, because on the ~1,400
European matches we care about, a gap of a thousandth of an RPS is not
distinguishable from luck.

Two things it cannot do, and one guard it needs:

* It predicts H/D/A, never 2-1. The league phase is settled on goal
  difference, so a goals model stays in the pipeline regardless.
* It has no notion of a club. It sees form and ratings, not identity,
  which is the point: club identity is what Elo and Dixon-Coles are
  already for.
* It will memorise noise if allowed to. Every window below ends before
  the one after it begins, early stopping watches a validation slice
  that the test window never touches, and the trees are kept shallow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ..eval.metrics import OUTCOMES


@dataclass(frozen=True)
class BoostedConfig:
    """Deliberately conservative settings.

    Depth 4 rather than the usual 6: with 55 features and a signal this
    faint, deep trees find structure that is not there. The row and
    column sampling serve the same end. ``n_estimators`` is an upper
    bound only — early stopping decides the real number.
    """

    n_estimators: int = 2000
    max_depth: int = 4
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 20.0
    reg_lambda: float = 2.0
    early_stopping_rounds: int = 50
    seed: int = 0


@dataclass
class BoostedResult:
    model: XGBClassifier
    columns: list[str]
    order: list[int]
    rounds: int
    trained_on: dict = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Probabilities in H, D, A order, one row per fixture."""
        raw = self.model.predict_proba(frame[self.columns])

        return raw[:, self.order]

    def importance(self, top: int | None = None) -> pd.DataFrame:
        """Which features the trees actually split on.

        Gain, not split count: a feature used once on a decisive split
        matters more than one used often near the leaves.
        """
        scores = self.model.get_booster().get_score(importance_type="gain")

        table = pd.DataFrame(
            {
                "feature": self.columns,
                "gain": [scores.get(c, 0.0) for c in self.columns],
            }
        ).sort_values("gain", ascending=False, ignore_index=True)

        return table.head(top) if top else table


def fit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
    config: BoostedConfig | None = None,
) -> BoostedResult:
    """Fit on ``train``, stop when ``validation`` stops improving.

    ``validation`` must sit strictly between the training window and
    whatever the model is later scored on. Choosing the number of trees
    on the test set is the same mistake as choosing an ensemble weight
    on it, which this project has already made once.
    """
    config = config or BoostedConfig()

    if validation.empty:
        raise ValueError("early stopping needs a non-empty validation window")

    latest_train = train["date"].max()
    earliest_validation = validation["date"].min()

    if earliest_validation <= latest_train:
        raise ValueError(
            "validation window overlaps training: it starts "
            f"{earliest_validation.date()} but training runs to "
            f"{latest_train.date()}"
        )

    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        early_stopping_rounds=config.early_stopping_rounds,
        objective="multi:softprob",
        num_class=len(OUTCOMES),
        # mlogloss, not accuracy: we are judged on the honesty of the
        # probabilities, not on how often the argmax is right.
        eval_metric="mlogloss",
        random_state=config.seed,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        train[columns],
        train["target"],
        eval_set=[(validation[columns], validation["target"])],
        verbose=False,
    )

    # XGBoost orders probability columns by class label. The targets are
    # already encoded 0=H, 1=D, 2=A, but relying on that silently is how
    # a model ends up quietly predicting draws as home wins.
    classes = list(model.classes_)
    order = [classes.index(i) for i, _ in enumerate(OUTCOMES)]

    return BoostedResult(
        model=model,
        columns=list(columns),
        order=order,
        rounds=int(model.best_iteration) + 1,
        trained_on={
            "matches": int(len(train)),
            "from": str(train["date"].min().date()),
            "to": str(latest_train.date()),
            "validation_matches": int(len(validation)),
            "features": len(columns),
        },
    )


def refit(
    frame: pd.DataFrame,
    columns: list[str],
    rounds: int,
    config: BoostedConfig | None = None,
) -> BoostedResult:
    """Retrain on the full training period for a fixed number of rounds.

    Early stopping has to hold a validation slice back, which leaves the
    boosted model trained on less history than the models it is being
    compared with. That would make any loss ambiguous — fewer trees, or
    fewer matches? So the tree count is chosen on the validation slice,
    and then the model is rebuilt on everything up to the test window
    with that count fixed. Both sides of the comparison then see exactly
    the same matches.
    """
    config = config or BoostedConfig()

    model = XGBClassifier(
        n_estimators=rounds,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        objective="multi:softprob",
        num_class=len(OUTCOMES),
        eval_metric="mlogloss",
        random_state=config.seed,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(frame[columns], frame["target"], verbose=False)

    classes = list(model.classes_)
    order = [classes.index(i) for i, _ in enumerate(OUTCOMES)]

    return BoostedResult(
        model=model,
        columns=list(columns),
        order=order,
        rounds=rounds,
        trained_on={
            "matches": int(len(frame)),
            "from": str(frame["date"].min().date()),
            "to": str(frame["date"].max().date()),
            "features": len(columns),
        },
    )
