"""Dixon-Coles: a goals model, not a winner model.

Elo answers "who wins". The league phase needs more than that -- places
eight and nine are separated by goal difference, so the table cannot be
simulated without scorelines. This model produces them.

Each club gets an attack and a defence rating. For a match between i at
home and j away, goals are Poisson with

    log lambda = attack[i] + defence[j] + home_advantage
    log mu     = attack[j] + defence[i]

Two departures from plain Poisson, both from Dixon and Coles (1997).

Independent Poisson gets low scores wrong: real matches produce more
0-0 and 1-1 and fewer 1-0 and 0-1 than independence predicts, because
sides shut up shop. A single parameter ``rho`` corrects those four
scorelines and leaves the rest alone.

And a match from 2015 should not count like one from April, so every
match is weighted by ``exp(-xi * days_ago)`` in the likelihood.

Everything is fitted jointly by maximum likelihood over all matches at
once, domestic and European together. That matters for league strength:
a joint fit propagates evidence between leagues through shared
opponents, where sequential Elo only passes it along one match at a
time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass
class DixonColesConfig:
    """Parameters of the fit."""

    # Time decay per day. 0.0018 gives a half-life of about a year;
    # larger forgets faster. Zero weights every match equally.
    xi: float = 0.0018

    # Ridge penalty on attack and defence. Pulls clubs with little data
    # toward average rather than letting ten matches define them.
    ridge: float = 0.5

    # Largest scoreline considered when building a probability grid.
    max_goals: int = 10

    # Drop matches whose time weight falls below this; they cannot
    # influence the fit but still cost time.
    weight_floor: float = 1e-4

    # Column whose levels each get their own home advantage. The fit is
    # dominated by 300k domestic matches, so a single shared value
    # describes domestic football and misdescribes European ties, which
    # are more lopsided and produce fewer draws. Set to None for one
    # shared value.
    home_advantage_by: str | None = "kind"

    # Optimiser iteration cap. The default is generous enough for a
    # cold fit; a bootstrap refit warm-started from the point estimate
    # only has to travel as far as the resample moved it, and can be
    # capped far lower without materially changing where it lands.
    max_iterations: int = 500


def _tau(goals_home, goals_away, lam, mu, rho):
    """Dixon-Coles correction for the four lowest scorelines."""
    out = np.ones_like(lam)

    both_nil = (goals_home == 0) & (goals_away == 0)
    nil_one = (goals_home == 0) & (goals_away == 1)
    one_nil = (goals_home == 1) & (goals_away == 0)
    one_one = (goals_home == 1) & (goals_away == 1)

    out[both_nil] = 1.0 - lam[both_nil] * mu[both_nil] * rho
    out[nil_one] = 1.0 + lam[nil_one] * rho
    out[one_nil] = 1.0 + mu[one_nil] * rho
    out[one_one] = 1.0 - rho

    return out, (both_nil, nil_one, one_nil, one_one)


@dataclass
class DixonColesResult:
    """Fitted attack and defence ratings."""

    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    home_advantages: dict[str, float]
    rho: float
    config: DixonColesConfig
    converged: bool
    log_likelihood: float

    def rates(self, home: str, away: str, group: str | None = None) -> tuple[float, float]:
        """Expected goals for each side."""
        attack_home = self.attack.get(home, 0.0)
        attack_away = self.attack.get(away, 0.0)
        defence_home = self.defence.get(home, 0.0)
        defence_away = self.defence.get(away, 0.0)

        gamma = self.home_advantages.get(group, self.home_advantage)

        return (
            float(np.exp(attack_home + defence_away + gamma)),
            float(np.exp(attack_away + defence_home)),
        )

    def score_matrix(self, home: str, away: str, group: str | None = None) -> np.ndarray:
        """Probability of every scoreline up to ``max_goals``.

        Rows are home goals, columns away goals.
        """
        lam, mu = self.rates(home, away, group)
        size = self.config.max_goals + 1

        goals = np.arange(size)
        grid = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))

        # The same low-score correction, applied to the grid.
        grid[0, 0] *= 1.0 - lam * mu * self.rho
        grid[0, 1] *= 1.0 + lam * self.rho
        grid[1, 0] *= 1.0 + mu * self.rho
        grid[1, 1] *= 1.0 - self.rho

        return grid / grid.sum()

    def predict(self, home: str, away: str, group: str | None = None) -> dict[str, float]:
        """Home / draw / away probabilities from the scoreline grid."""
        grid = self.score_matrix(home, away, group)

        return {
            "H": float(np.tril(grid, -1).sum()),
            "D": float(np.trace(grid)),
            "A": float(np.triu(grid, 1).sum()),
        }

    def table(self, top: int | None = None) -> pd.DataFrame:
        """Clubs ranked by overall strength: attack less defence."""
        rows = [
            {
                "club": club,
                "attack": value,
                "defence": self.defence.get(club, 0.0),
                "strength": value - self.defence.get(club, 0.0),
            }
            for club, value in self.attack.items()
        ]

        df = pd.DataFrame(rows).sort_values("strength", ascending=False)
        df.insert(0, "rank", range(1, len(df) + 1))

        return df.head(top) if top else df.reset_index(drop=True)


def fit(
    matches: pd.DataFrame,
    config: DixonColesConfig | None = None,
    reference: pd.Timestamp | None = None,
    sample_weights: np.ndarray | None = None,
    start_from: "DixonColesResult | None" = None,
) -> DixonColesResult:
    """Maximise the weighted Dixon-Coles likelihood.

    Gradients are analytic; a numerical gradient over three thousand
    parameters would need three thousand likelihood evaluations per
    step and is not practical.

    ``sample_weights`` multiplies the time-decay weight of each match,
    which is how :mod:`pitchiq.models.uncertainty` resamples the record
    to find out how firmly each rating is pinned down. It is applied
    after the decay floor, so which matches take part is decided by
    their age alone and does not shift from one resample to the next.

    ``start_from`` seeds the optimiser with an already-fitted result.
    Two hundred bootstrap refits from a standing start is half an hour;
    from the point estimate, each one only has to travel as far as the
    resample moved it.
    """
    config = config or DixonColesConfig()

    reference = reference or matches["date"].max()
    days = (reference - matches["date"]).dt.days.to_numpy().astype(float)
    weights = np.exp(-config.xi * days)

    keep = weights >= config.weight_floor
    matches = matches[keep]
    weights = weights[keep]

    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=float)

        if len(sample_weights) != len(keep):
            raise ValueError(
                f"sample_weights has {len(sample_weights)} entries for "
                f"{len(keep)} matches"
            )

        weights = weights * sample_weights[keep]

    clubs = sorted(set(matches["home_key"]) | set(matches["away_key"]))
    index = {club: i for i, club in enumerate(clubs)}
    n = len(clubs)

    home = matches["home_key"].map(index).to_numpy()
    away = matches["away_key"].map(index).to_numpy()
    goals_home = matches["fthg"].to_numpy().astype(float)
    goals_away = matches["ftag"].to_numpy().astype(float)

    if config.home_advantage_by and config.home_advantage_by in matches:
        levels = sorted(matches[config.home_advantage_by].dropna().unique())
        group = matches[config.home_advantage_by].map(
            {level: i for i, level in enumerate(levels)}
        ).to_numpy()
    else:
        levels = ["all"]
        group = np.zeros(len(matches), dtype=int)

    g = len(levels)

    def unpack(parameters):
        attack = parameters[:n]
        defence = parameters[n : 2 * n]
        gammas = parameters[2 * n : 2 * n + g]

        return attack, defence, gammas, parameters[2 * n + g]

    def negative_log_likelihood(parameters):
        attack, defence, gammas, rho = unpack(parameters)

        log_lam = attack[home] + defence[away] + gammas[group]
        log_mu = attack[away] + defence[home]

        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        tau, masks = _tau(goals_home, goals_away, lam, mu, rho)
        tau = np.clip(tau, 1e-10, None)

        loglik = weights * (
            np.log(tau)
            - lam + goals_home * log_lam
            - mu + goals_away * log_mu
        )

        penalty = config.ridge * (np.sum(attack**2) + np.sum(defence**2))

        # Only the split between attack and defence is unidentified;
        # anchoring the mean attack fixes it.
        anchor = 1000.0 * np.mean(attack) ** 2

        value = -np.sum(loglik) + penalty + anchor

        # --- gradient ---
        both_nil, nil_one, one_nil, one_one = masks

        d_tau_d_lam = np.zeros_like(lam)
        d_tau_d_mu = np.zeros_like(mu)
        d_tau_d_rho = np.zeros_like(lam)

        d_tau_d_lam[both_nil] = -mu[both_nil] * rho / tau[both_nil]
        d_tau_d_mu[both_nil] = -lam[both_nil] * rho / tau[both_nil]
        d_tau_d_rho[both_nil] = -lam[both_nil] * mu[both_nil] / tau[both_nil]

        d_tau_d_lam[nil_one] = rho / tau[nil_one]
        d_tau_d_rho[nil_one] = lam[nil_one] / tau[nil_one]

        d_tau_d_mu[one_nil] = rho / tau[one_nil]
        d_tau_d_rho[one_nil] = mu[one_nil] / tau[one_nil]

        d_tau_d_rho[one_one] = -1.0 / tau[one_one]

        # d(loglik)/d(log lambda) and d(loglik)/d(log mu)
        d_lam = weights * (goals_home - lam + d_tau_d_lam * lam)
        d_mu = weights * (goals_away - mu + d_tau_d_mu * mu)

        grad_attack = np.zeros(n)
        grad_defence = np.zeros(n)

        np.add.at(grad_attack, home, d_lam)
        np.add.at(grad_attack, away, d_mu)
        np.add.at(grad_defence, away, d_lam)
        np.add.at(grad_defence, home, d_mu)

        grad_gamma = np.zeros(g)
        np.add.at(grad_gamma, group, d_lam)

        gradient = np.empty(2 * n + g + 1)
        gradient[:n] = -grad_attack + 2.0 * config.ridge * attack + \
            2000.0 * np.mean(attack) / n
        gradient[n : 2 * n] = -grad_defence + 2.0 * config.ridge * defence
        gradient[2 * n : 2 * n + g] = -grad_gamma
        gradient[2 * n + g] = -np.sum(weights * d_tau_d_rho)

        return value, gradient

    start = np.zeros(2 * n + g + 1)
    start[2 * n : 2 * n + g] = 0.25   # home advantage in log-goal space
    start[2 * n + g] = -0.05          # rho is small and negative in practice

    if start_from is not None:
        # A club absent from the seed keeps the neutral zero rather than
        # borrowing another club's rating.
        for club, i in index.items():
            start[i] = start_from.attack.get(club, 0.0)
            start[n + i] = start_from.defence.get(club, 0.0)

        for i, level in enumerate(levels):
            start[2 * n + i] = start_from.home_advantages.get(
                str(level), start_from.home_advantage
            )

        start[2 * n + g] = np.clip(start_from.rho, -0.4, 0.4)

    bounds = [(None, None)] * (2 * n + g) + [(-0.4, 0.4)]

    fitted = minimize(
        negative_log_likelihood,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": config.max_iterations},
    )

    attack, defence, gammas, rho = unpack(fitted.x)

    by_group = {str(level): float(gammas[i]) for i, level in enumerate(levels)}

    return DixonColesResult(
        attack={club: float(attack[i]) for club, i in index.items()},
        defence={club: float(defence[i]) for club, i in index.items()},
        home_advantage=float(np.mean(gammas)),
        home_advantages=by_group,
        rho=float(rho),
        config=config,
        converged=bool(fitted.success),
        log_likelihood=float(-fitted.fun),
    )
