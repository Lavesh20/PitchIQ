"""Ratings learned from shots on target, used as a prior for goals.

Goals are a noisy read of how good a side is. A team can create six
clear chances and score none, and the goals model has no way to tell
that apart from a team that created nothing. Shots on target are the
same story told with four times as many events, so the same attack and
defence structure fitted to them is a steadier estimate of the thing we
actually want to know.

They cannot replace goals: the simulator needs scorelines, and nobody
qualifies on shots. What they were meant to do is say where a club sits
when its goal record is thin, which is what the ridge penalty otherwise
answers with "assume average".

**Tested, and it does not work with this data.** Held-out RPS moves by
-0.00002 on European matches and -0.00003 on domestic: nothing. The
reason is a coverage mismatch that no amount of modelling fixes.

    training matches    clubs in the test window    have a shot prior
    0-20                                     246                   0%
    20-60                                     35                  20%
    150-400                                  262                  21%
    400+                                     374                  91%

Football-data.co.uk records shots for the bigger divisions from the
mid-2010s on, about 127,000 matches against 300,000. Those are precisely
the clubs whose goal record is already long enough that the ridge barely
touches them, so changing what they are shrunk *toward* changes nothing.
Of the 173 clubs in the European field with fewer than sixty matches
behind them — the ones a prior exists to help — **not one has shot data.**

The module is kept because the finding is worth keeping, and because the
``prior`` argument it drove into :func:`dixon_coles.fit` is a general
capability: any better-measured signal can shrink the goals model toward
it. Lineup and injury data would be the obvious candidate, and unlike
shots it would reach the clubs that need it.

Everything here degrades to goals-only where the columns are absent
rather than failing: :func:`fit` returns ``None`` when there is too
little shot data to learn from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import dixon_coles as dc


@dataclass
class ShotPrior:
    """Shot-based ratings, rescaled onto the goals model's scale."""

    attack: dict[str, float]
    defence: dict[str, float]
    scale: float
    fitted_on: int
    raw: dc.DixonColesResult

    def as_result(self) -> dc.DixonColesResult:
        """The prior in the shape :func:`dixon_coles.fit` expects."""
        return dc.DixonColesResult(
            attack=dict(self.attack),
            defence=dict(self.defence),
            home_advantage=self.raw.home_advantage,
            home_advantages=dict(self.raw.home_advantages),
            rho=0.0,
            config=self.raw.config,
            converged=self.raw.converged,
            log_likelihood=float("nan"),
        )

    def table(self, top: int | None = None) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "club": list(self.attack),
                "attack": list(self.attack.values()),
                "defence": [self.defence[c] for c in self.attack],
            }
        ).sort_values("attack", ascending=False, ignore_index=True)

        return frame.head(top) if top else frame


def with_shots(matches: pd.DataFrame) -> pd.DataFrame:
    """The subset where both sides' shots on target were recorded."""
    if "home_sot" not in matches.columns:
        return matches.iloc[0:0]

    return matches[matches.home_sot.notna() & matches.away_sot.notna()]


def _rescale(goals: dc.DixonColesResult, shots: dc.DixonColesResult) -> float:
    """How far a unit of shot rating moves a club's goal rating.

    Both fits anchor mean attack at zero, so their ratings are centred
    and directly comparable in shape — but not in spread. A side taking
    40% more shots on target than average does not score 40% more goals,
    it scores rather more, because the better sides also convert at a
    higher rate. The slope of that relationship is what makes the shot
    ratings usable as a prior on the goals scale, and it is measured
    rather than assumed.

    Only clubs present in both fits contribute, and the line is forced
    through the origin because both scales are already centred.
    """
    shared = [c for c in goals.attack if c in shots.attack]

    if len(shared) < 20:
        return 1.0

    x = np.array([shots.attack[c] for c in shared])
    y = np.array([goals.attack[c] for c in shared])

    denominator = float(np.sum(x * x))

    return float(np.sum(x * y) / denominator) if denominator else 1.0


def fit(
    matches: pd.DataFrame,
    config: dc.DixonColesConfig | None = None,
    reference: pd.Timestamp | None = None,
) -> ShotPrior | None:
    """Fit attack and defence on shots on target, rescaled for goals.

    Returns ``None`` when there is not enough shot data to learn from,
    so callers can fall back to the ordinary goals-only fit rather than
    being handed a prior built on nothing.
    """
    config = config or dc.DixonColesConfig()
    recorded = with_shots(matches)

    if len(recorded) < 5000:
        return None

    # The low-score correction exists because real football produces more
    # 0-0s and 1-1s than independent Poisson allows. Shot counts average
    # around five a side, so those cells are nearly empty and the
    # parameter is describing noise rather than football -- it fits to
    # about +0.13 here against -0.04 on goals. It is left free because
    # nothing downstream reads it: only the attack and defence ratings
    # become the prior, and ``as_result`` sets rho to zero.
    as_shots = recorded.assign(
        fthg=recorded.home_sot.astype(float),
        ftag=recorded.away_sot.astype(float),
    )

    shot_ratings = dc.fit(as_shots, config, reference)
    goal_ratings = dc.fit(recorded, config, reference)

    scale = _rescale(goal_ratings, shot_ratings)

    return ShotPrior(
        attack={c: v * scale for c, v in shot_ratings.attack.items()},
        defence={c: v * scale for c, v in shot_ratings.defence.items()},
        scale=scale,
        fitted_on=len(recorded),
        raw=shot_ratings,
    )
