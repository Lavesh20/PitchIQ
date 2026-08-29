"""How firmly is each rating actually pinned down?

Dixon-Coles returns Arsenal's attack as 1.04. The simulator then plays
10,000 seasons in which Arsenal's attack is 1.04 every single time, so
the only thing that varies across those seasons is luck inside the
matches. The possibility that we have simply misjudged the club never
enters.

That is not a small omission. 1.04 rests on 238 matches and is worth
trusting. A club we have seen 22 times gets a number written to the
same two decimal places and deserves nothing like the same confidence.
Treating both as facts is what makes the forecast too sure of itself,
and it is measurable: across 4,225 club-seasons, finishing positions
regress on predicted positions with a slope of 0.83 rather than 1.0,
which is to say the predictions are spread about a fifth wider than
reality supports.

The fix is to stop pretending. Refit the model many times on resampled
versions of the record, keep every set of parameters, and let the
simulator draw a different one for each season it plays. A club whose
rating barely moves between refits stays firm; one that swings around
brings that swing into the forecast, where it belongs.

The resampling is the Bayesian bootstrap: every match keeps a positive
weight drawn from a Dirichlet, rather than the classical version where
matches are drawn with replacement. With the classical form a club with
twenty matches can be left with none at all in some resamples and its
rating collapses to the ridge prior, which reads as enormous
uncertainty when it is really an artefact of the resampling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import dixon_coles as dc


@dataclass
class ParameterSamples:
    """A set of plausible parameter vectors, one per resample."""

    clubs: list[str]
    attack: np.ndarray            # (draws, clubs)
    defence: np.ndarray           # (draws, clubs)
    home_advantages: dict[str, np.ndarray]
    rho: np.ndarray               # (draws,)
    point: dc.DixonColesResult
    trained_on: dict = field(default_factory=dict)

    @property
    def draws(self) -> int:
        return len(self.rho)

    def draw(self, i: int) -> dc.DixonColesResult:
        """One resample, as a model the simulator can use unchanged."""
        return dc.DixonColesResult(
            attack={club: float(self.attack[i, j]) for j, club in enumerate(self.clubs)},
            defence={club: float(self.defence[i, j]) for j, club in enumerate(self.clubs)},
            home_advantage=float(np.mean([v[i] for v in self.home_advantages.values()])),
            home_advantages={k: float(v[i]) for k, v in self.home_advantages.items()},
            rho=float(self.rho[i]),
            config=self.point.config,
            converged=True,
            log_likelihood=float("nan"),
        )

    def spread(self, top: int | None = None) -> pd.DataFrame:
        """Standard deviation of each club's rating across the resamples.

        The column worth reading is ``attack_sd``: it should be small
        for clubs with a long record and large for clubs without one. If
        it is flat across both, the bootstrap has not worked.
        """
        table = pd.DataFrame(
            {
                "club": self.clubs,
                "attack": self.attack.mean(axis=0),
                "attack_sd": self.attack.std(axis=0, ddof=1),
                "defence": self.defence.mean(axis=0),
                "defence_sd": self.defence.std(axis=0, ddof=1),
            }
        ).sort_values("attack_sd", ascending=False, ignore_index=True)

        return table.head(top) if top else table

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            clubs=np.array(self.clubs, dtype=object),
            attack=self.attack,
            defence=self.defence,
            rho=self.rho,
            groups=np.array(list(self.home_advantages), dtype=object),
            gammas=np.array(list(self.home_advantages.values())),
            point_attack=np.array([self.point.attack[c] for c in self.clubs]),
            point_defence=np.array([self.point.defence[c] for c in self.clubs]),
            point_rho=self.point.rho,
            point_gammas=np.array(
                [self.point.home_advantages[g] for g in self.home_advantages]
            ),
            trained_on=np.array([str(self.trained_on)], dtype=object),
        )

    @classmethod
    def load(cls, path, config: dc.DixonColesConfig | None = None) -> "ParameterSamples":
        data = np.load(path, allow_pickle=True)
        clubs = list(data["clubs"])
        groups = list(data["groups"])

        point = dc.DixonColesResult(
            attack=dict(zip(clubs, data["point_attack"])),
            defence=dict(zip(clubs, data["point_defence"])),
            home_advantage=float(np.mean(data["point_gammas"])),
            home_advantages=dict(zip(groups, data["point_gammas"].tolist())),
            rho=float(data["point_rho"]),
            config=config or dc.DixonColesConfig(),
            converged=True,
            log_likelihood=float("nan"),
        )

        return cls(
            clubs=clubs,
            attack=data["attack"],
            defence=data["defence"],
            home_advantages=dict(zip(groups, data["gammas"])),
            rho=data["rho"],
            point=point,
            trained_on=eval(str(data["trained_on"][0])),  # noqa: S307 - our own dict
        )


def bootstrap(
    matches: pd.DataFrame,
    config: dc.DixonColesConfig | None = None,
    reference: pd.Timestamp | None = None,
    draws: int = 200,
    seed: int = 0,
    point: dc.DixonColesResult | None = None,
    verbose: bool = True,
) -> ParameterSamples:
    """Refit the model ``draws`` times on reweighted copies of the record.

    Each refit is warm-started from the point estimate, so it only has
    to travel as far as its resample moved things. The iteration cap is
    deliberately left at the default: capping it lower does cut the time
    per fit, but the under-converged fits drift in a consistent
    direction rather than a random one, which would show up as
    uncertainty that is really just an unfinished optimisation.
    """
    config = config or dc.DixonColesConfig()
    point = point or dc.fit(matches, config, reference)

    generator = np.random.default_rng(seed)
    n = len(matches)

    clubs = sorted(point.attack)
    groups = list(point.home_advantages)

    attack = np.empty((draws, len(clubs)))
    defence = np.empty((draws, len(clubs)))
    gammas = {g: np.empty(draws) for g in groups}
    rho = np.empty(draws)

    started = time.time()

    for d in range(draws):
        # Dirichlet weights averaging one, so the effective sample size
        # matches the real one and the ridge penalty keeps its meaning.
        weights = generator.dirichlet(np.ones(n)) * n

        fitted = dc.fit(
            matches,
            config,
            reference,
            sample_weights=weights,
            start_from=point,
        )

        attack[d] = [fitted.attack.get(c, 0.0) for c in clubs]
        defence[d] = [fitted.defence.get(c, 0.0) for c in clubs]
        rho[d] = fitted.rho

        for g in groups:
            gammas[g][d] = fitted.home_advantages.get(g, fitted.home_advantage)

        if verbose and (d + 1) % 10 == 0:
            elapsed = time.time() - started
            print(
                f"  {d + 1}/{draws} resamples, {elapsed:.0f}s elapsed, "
                f"{elapsed / (d + 1) * (draws - d - 1):.0f}s left",
                flush=True,
            )

    return ParameterSamples(
        clubs=clubs,
        attack=attack,
        defence=defence,
        home_advantages=gammas,
        rho=rho,
        point=point,
        trained_on={
            "matches": int(n),
            "draws": int(draws),
            "from": str(matches["date"].min().date()),
            "to": str(matches["date"].max().date()),
            "seed": int(seed),
        },
    )
