"""Correct Elo ratings for the league they were earned in.

Elo is zero-sum, so a league cannot inflate as a bloc without winning in
Europe. What it *can* do is let a dominant club hoard its own league's
points indefinitely. Celtic take points off the same Scottish sides every
season; Bournemouth trade points inside an inflated English pool and
never face anyone outside it. Only continental matches correct either,
and there are few of them.

The correction therefore has two parts per country, not one:

    adjusted = mean[c] + scale[c] * (rating - mean[c]) + offset[c]

``offset`` moves a whole league up or down. That alone cannot fix
Celtic, because shifting every Scottish club equally preserves Celtic's
lead over Rangers. ``scale`` is what does the work: a value below one
pulls a league's clubs toward their own average, squeezing a lead that
was built on domestic dominance rather than European evidence.

Both are fitted on the only evidence that speaks to them -- matches
between clubs from different countries -- and both are shrunk toward
"no correction", so a country with forty European matches cannot swing
far on noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCORES = {"H": 1.0, "D": 0.5, "A": 0.0}


@dataclass
class LeagueStrength:
    """Per-country location and dispersion corrections."""

    offsets: dict[str, float]
    scales: dict[str, float]
    means: dict[str, float]
    counts: dict[str, int]
    home_advantage: float
    priors: tuple[float, float] = (100.0, 0.35)

    def adjust(self, rating, country):
        """Map a raw rating onto the common European scale."""
        rating = np.asarray(rating, dtype=float)

        if isinstance(country, str) or country is None:
            country = [country] * rating.size

        mean = np.array([self.means.get(c, 1500.0) for c in country])
        scale = np.array([self.scales.get(c, 1.0) for c in country])
        offset = np.array([self.offsets.get(c, 0.0) for c in country])

        return mean + scale * (rating - mean) + offset

    def table(self) -> pd.DataFrame:
        rows = [
            {
                "country": country,
                "offset": self.offsets[country],
                "scale": self.scales[country],
                "league_mean": self.means.get(country, float("nan")),
                "european_matches": self.counts.get(country, 0),
            }
            for country in self.offsets
        ]

        return (
            pd.DataFrame(rows)
            .sort_values("offset", ascending=False)
            .reset_index(drop=True)
        )


def league_means(history: pd.DataFrame, tier: int = 1) -> dict[str, float]:
    """Average rating of a country's top-division clubs.

    Taken over domestic matches so the mean reflects the whole league,
    not just the sides good enough to reach Europe. Shrinking Celtic
    toward a mean computed only from European participants would shrink
    it toward itself.
    """
    domestic = history[(history.kind == "domestic") & (history.tier == tier)]

    stacked = pd.concat(
        [
            domestic[["home_cc", "elo_home"]].rename(
                columns={"home_cc": "cc", "elo_home": "elo"}
            ),
            domestic[["away_cc", "elo_away"]].rename(
                columns={"away_cc": "cc", "elo_away": "elo"}
            ),
        ]
    )

    return stacked.groupby("cc").elo.mean().to_dict()


def fit(
    european: pd.DataFrame,
    means: dict[str, float],
    home_advantage: float = 65.0,
    offset_prior: float = 100.0,
    scale_prior: float = 0.35,
) -> LeagueStrength:
    """Fit per-country offset and scale on cross-country matches.

    ``offset_prior`` is the standard deviation, in rating points, of how
    far a league is expected to sit from its Elo-implied level.
    ``scale_prior`` is the same for the log of the dispersion scale;
    0.35 lets a league's spread roughly halve or double before the
    penalty bites.
    """
    frame = european[
        (european.home_cc != european.away_cc)
        & european.home_cc.notna()
        & european.away_cc.notna()
    ]

    countries = sorted(set(frame.home_cc) | set(frame.away_cc))
    index = {country: i for i, country in enumerate(countries)}
    n = len(countries)

    # No cross-country matches means no evidence about how leagues
    # compare, so the honest answer is to leave every rating alone.
    if frame.empty or n == 0:
        return LeagueStrength(
            offsets={}, scales={}, means=dict(means), counts={},
            home_advantage=home_advantage,
            priors=(offset_prior, scale_prior),
        )

    home_index = frame.home_cc.map(index).to_numpy().astype(int)
    away_index = frame.away_cc.map(index).to_numpy().astype(int)

    home_rating = frame.elo_home.to_numpy()
    away_rating = frame.elo_away.to_numpy()

    home_mean = np.array([means.get(c, 1500.0) for c in frame.home_cc])
    away_mean = np.array([means.get(c, 1500.0) for c in frame.away_cc])

    actual = frame.ftr.map(SCORES).to_numpy()
    counts = pd.concat([frame.home_cc, frame.away_cc]).value_counts().to_dict()

    def unpack(parameters):
        return parameters[:n], parameters[n:]

    def objective(parameters: np.ndarray) -> float:
        offsets, log_scales = unpack(parameters)

        scale_home = np.exp(log_scales[home_index])
        scale_away = np.exp(log_scales[away_index])

        adjusted_home = (
            home_mean + scale_home * (home_rating - home_mean) + offsets[home_index]
        )
        adjusted_away = (
            away_mean + scale_away * (away_rating - away_mean) + offsets[away_index]
        )

        difference = adjusted_home + home_advantage - adjusted_away
        expected = 1.0 / (1.0 + 10.0 ** (-difference / 400.0))

        error = np.sum((actual - expected) ** 2)

        penalty = (
            np.sum(offsets**2) / (2.0 * offset_prior**2)
            + np.sum(log_scales**2) / (2.0 * scale_prior**2)
        )

        return error + penalty

    start = np.zeros(2 * n)
    fitted = minimize(objective, start, method="L-BFGS-B").x

    offsets, log_scales = unpack(fitted)

    # Only differences between countries are identified, so anchor the
    # level by weighting each country by how much European football it
    # actually plays.
    weights = np.array([counts.get(c, 0) for c in countries], dtype=float)
    offsets = offsets - np.average(offsets, weights=weights)

    return LeagueStrength(
        offsets={c: float(offsets[i]) for c, i in index.items()},
        scales={c: float(np.exp(log_scales[i])) for c, i in index.items()},
        means={c: float(means.get(c, 1500.0)) for c in countries},
        counts={c: int(counts.get(c, 0)) for c in countries},
        home_advantage=home_advantage,
        priors=(offset_prior, scale_prior),
    )


@dataclass
class StrengthSamples:
    """Per-country corrections across resamples of the European record."""

    countries: list[str]
    offsets: np.ndarray          # (draws, countries)
    scales: np.ndarray           # (draws, countries)
    counts: dict[str, int]
    point: LeagueStrength

    def table(self, level: float = 0.9) -> pd.DataFrame:
        """Each country's correction with an interval and its evidence.

        Read the interval against the point estimate. Romania's offset
        rests on a few dozen European matches, and an interval that
        comfortably contains zero is the honest way to say so — the
        correction is the best guess available, not a measurement.
        """
        tail = (1.0 - level) / 2.0

        if not self.countries:
            # No cross-country evidence at all. An empty frame still has
            # to carry the columns, or every caller has to special-case
            # it before it can be filtered or joined.
            return pd.DataFrame(
                columns=[
                    "country", "european_matches",
                    "offset", "offset_low", "offset_high",
                    "scale", "scale_low", "scale_high",
                    "offset_certain", "scale_certain",
                ]
            )

        rows = []
        for i, country in enumerate(self.countries):
            offset = self.offsets[:, i]
            scale = self.scales[:, i]

            rows.append(
                {
                    "country": country,
                    "european_matches": self.counts.get(country, 0),
                    "offset": self.point.offsets.get(country, 0.0),
                    "offset_low": float(np.quantile(offset, tail)),
                    "offset_high": float(np.quantile(offset, 1 - tail)),
                    "scale": self.point.scales.get(country, 1.0),
                    "scale_low": float(np.quantile(scale, tail)),
                    "scale_high": float(np.quantile(scale, 1 - tail)),
                }
            )

        table = pd.DataFrame(rows)
        # A correction whose interval spans zero (or one, for a scale) is
        # not evidence of anything; flagging it is cheaper than expecting
        # every reader to check.
        table["offset_certain"] = (table.offset_low > 0) | (table.offset_high < 0)
        table["scale_certain"] = (table.scale_low > 1) | (table.scale_high < 1)

        return table.sort_values("offset", ascending=False).reset_index(drop=True)


def bootstrap(
    european: pd.DataFrame,
    means: dict[str, float],
    home_advantage: float = 65.0,
    offset_prior: float = 100.0,
    scale_prior: float = 0.35,
    draws: int = 200,
    seed: int = 0,
) -> StrengthSamples:
    """Refit the corrections on resampled European matches.

    The prior already shrinks a country in proportion to how little
    evidence it has, because the error term grows with match count while
    the penalty does not. What the prior cannot do is say how uncertain
    the surviving correction is, and a −117 point offset reported to
    three figures invites more confidence than forty matches can carry.

    Ordinary resampling with replacement is used rather than the
    Dirichlet weighting the Dixon-Coles bootstrap needs: here a country
    dropping out of a resample entirely is informative rather than an
    artefact, since it means the country barely plays in Europe.
    """
    point = fit(european, means, home_advantage, offset_prior, scale_prior)
    countries = sorted(point.offsets)

    if not countries:
        return StrengthSamples(
            countries=[],
            offsets=np.empty((0, 0)),
            scales=np.empty((0, 0)),
            counts={},
            point=point,
        )

    generator = np.random.default_rng(seed)
    frame = european[
        (european.home_cc != european.away_cc)
        & european.home_cc.notna()
        & european.away_cc.notna()
    ]

    offsets = np.zeros((draws, len(countries)))
    scales = np.ones((draws, len(countries)))

    for d in range(draws):
        pick = generator.integers(0, len(frame), len(frame))
        resampled = fit(
            frame.iloc[pick], means, home_advantage, offset_prior, scale_prior
        )

        for i, country in enumerate(countries):
            offsets[d, i] = resampled.offsets.get(country, 0.0)
            scales[d, i] = resampled.scales.get(country, 1.0)

    return StrengthSamples(
        countries=countries,
        offsets=offsets,
        scales=scales,
        counts=dict(point.counts),
        point=point,
    )


def apply(history: pd.DataFrame, strength: LeagueStrength) -> pd.DataFrame:
    """Add league-adjusted ratings and rating difference to a history frame."""
    out = history.copy()

    out["elo_home_adj"] = strength.adjust(out.elo_home.to_numpy(), list(out.home_cc))
    out["elo_away_adj"] = strength.adjust(out.elo_away.to_numpy(), list(out.away_cc))
    out["elo_diff_adj"] = (
        out.elo_home_adj + strength.home_advantage - out.elo_away_adj
    )

    return out
