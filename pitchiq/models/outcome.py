"""Turn a rating difference into home / draw / away probabilities.

Elo yields an expected score, not three probabilities, and the split
between draw and win is not recoverable from it analytically. Rather
than assume a shape, this fits the mapping on real results: multinomial
logistic regression on the rating difference.

Fitted on domestic matches only, so that continental matches stay
untouched for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..eval.metrics import OUTCOMES


@dataclass
class OutcomeModel:
    """Rating difference -> P(home), P(draw), P(away)."""

    model: LogisticRegression

    @staticmethod
    def _features(elo_difference) -> np.ndarray:
        difference = np.asarray(elo_difference, dtype=float).reshape(-1, 1)

        # The absolute difference lets the draw probability peak when
        # the sides are level and fall away in both directions, which a
        # single linear term cannot express.
        return np.column_stack([difference / 100.0, np.abs(difference) / 100.0])

    @classmethod
    def fit(cls, elo_difference, actual) -> "OutcomeModel":
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(cls._features(elo_difference), np.asarray(actual))

        return cls(model=model)

    def predict(self, elo_difference) -> np.ndarray:
        raw = self.model.predict_proba(self._features(elo_difference))

        # sklearn orders columns by sorted class label; reorder to H D A.
        order = [list(self.model.classes_).index(o) for o in OUTCOMES]

        return raw[:, order]
