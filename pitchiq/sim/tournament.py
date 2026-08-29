"""Monte Carlo simulation of the Champions League.

The goals model gives one match at a time. Questions worth asking are
about whole seasons -- who finishes top eight, who lifts the trophy --
and those depend on all 144 league-phase results together plus every
knockout round after. There are far too many combinations to enumerate,
so we play the season out at random many thousands of times and count.

The competition is simulated as UEFA actually runs it since 2024/25:

    36 clubs, eight league-phase matches each, three points a win.
    1-8    straight to the round of 16
    9-24   a two-legged knockout play-off first
    25-36  eliminated outright

Two details do real work. Positions are separated by UEFA's tiebreakers,
not just points -- last season four clubs finished on sixteen, and the
gap between eighth and ninth is two extra matches against a good side.
And the bracket is fixed rather than redrawn each round, so a league
position matters twice: once for skipping the play-off, again for which
half of the draw you land in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .rules import DEFAULT, Format, for_season  # noqa: F401

# The bracket used to live here as module constants. It now lives in
# ``rules.py`` keyed by season, because UEFA changes the format and a
# simulator with the shape baked into its code loses the old behaviour
# every time that happens. These aliases keep the previous names working.
PLAYOFF_BANDS = list(DEFAULT.playoff_bands)
ROUND_OF_16 = list(DEFAULT.round_of_16)
EXTRA_TIME_SHARE = DEFAULT.extra_time_share


@dataclass
class Grids:
    """Cumulative scoreline distributions for every possible pairing.

    Pairings change from run to run once the knockout bracket forms, so
    every ordered pair is precomputed once and sampled by lookup.
    """

    cumulative: np.ndarray          # (n, n, cells)
    home_rate: np.ndarray           # (n, n)
    away_rate: np.ndarray           # (n, n)
    width: int

    def sample(self, home: np.ndarray, away: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
        """Draw a scoreline for each of many fixtures at once."""
        draws = rng.random(len(home))
        cells = self.cumulative[home, away]

        picked = (cells < draws[:, None]).sum(axis=1)
        picked = np.clip(picked, 0, cells.shape[1] - 1)

        return picked // self.width, picked % self.width


def build_grids(model, keys: list[str], neutral: bool = False) -> Grids:
    """Precompute every ordered pairing's scoreline distribution."""
    n = len(keys)
    first = model.score_matrix(keys[0], keys[1])
    width = first.shape[1]
    cells = first.size

    cumulative = np.zeros((n, n, cells))
    home_rate = np.zeros((n, n))
    away_rate = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            grid = model.score_matrix(keys[i], keys[j])

            if neutral:
                # A final is played at a neutral venue, so average the
                # two orientations to strip the home advantage out.
                grid = 0.5 * (grid + model.score_matrix(keys[j], keys[i]).T)
                grid = grid / grid.sum()

            if not np.isfinite(grid).all() or (grid < 0).any():
                raise ValueError(
                    f"score matrix for {keys[i]} v {keys[j]} is not a "
                    "distribution; the model produced a negative or "
                    "non-finite goal rate"
                )

            cumulative[i, j] = np.cumsum(grid.ravel())

            goals = np.arange(grid.shape[0])
            home_rate[i, j] = (grid.sum(axis=1) * goals).sum()
            away_rate[i, j] = (grid.sum(axis=0) * goals).sum()

    return Grids(cumulative, home_rate, away_rate, width)


def league_phase(grids: Grids, home: np.ndarray, away: np.ndarray, runs: int, rng):
    """Play every league-phase fixture ``runs`` times.

    Returns the per-club totals needed for UEFA's tiebreakers.
    """
    n_clubs = int(max(home.max(), away.max())) + 1
    fixtures = len(home)

    totals = {
        name: np.zeros((runs, n_clubs))
        for name in ("points", "scored", "conceded", "away_scored", "wins", "away_wins")
    }

    for f in range(fixtures):
        h = np.full(runs, home[f])
        a = np.full(runs, away[f])

        goals_home, goals_away = grids.sample(h, a, rng)

        home_won = goals_home > goals_away
        away_won = goals_away > goals_home
        drawn = ~home_won & ~away_won

        totals["points"][:, home[f]] += np.where(home_won, 3, np.where(drawn, 1, 0))
        totals["points"][:, away[f]] += np.where(away_won, 3, np.where(drawn, 1, 0))

        totals["scored"][:, home[f]] += goals_home
        totals["scored"][:, away[f]] += goals_away
        totals["conceded"][:, home[f]] += goals_away
        totals["conceded"][:, away[f]] += goals_home

        totals["away_scored"][:, away[f]] += goals_away
        totals["wins"][:, home[f]] += home_won
        totals["wins"][:, away[f]] += away_won
        totals["away_wins"][:, away[f]] += away_won

    return totals


def rank(totals: dict[str, np.ndarray], rng=None) -> np.ndarray:
    """Order clubs by UEFA's league-phase tiebreakers.

    Points, then goal difference, then goals scored, then away goals,
    then wins, then away wins.

    UEFA has two further criteria — disciplinary points, then club
    coefficient — and neither is modelled, because neither can be. The
    UEFA feed carries no cards, and simulating them from domestic card
    rates would be inventing the evidence rather than using it. Club
    coefficients are not in the draw file either.

    So a tie can survive all six. Measured over 20,000 simulated
    seasons, that happens in **2.2% of seasons**, and in only **0.16%**
    — about one in 645 — does the tie straddle the 8/9 or 24/25 line
    and therefore change who qualifies for what.

    Passing ``rng`` breaks those survivors at random, which is what the
    default of ordering by club index gets wrong: the index is fixed, so
    the same club wins every tie it is ever in. The bias is small but it
    is a bias, and randomising costs nothing. ``table`` in
    :mod:`pitchiq.sim.league` leaves it out deliberately — a real
    season's table should compile the same way every time it is read.
    """
    difference = totals["scored"] - totals["conceded"]

    score = (
        totals["points"] * 1e10
        + (difference + 100.0) * 1e7
        + totals["scored"] * 1e5
        + totals["away_scored"] * 1e3
        + totals["wins"] * 1e1
        + totals["away_wins"]
    )

    if rng is not None:
        # Strictly below 0.5, so it can only separate clubs already
        # level on every criterion above: one away win is worth 1.0.
        score = score + rng.random(score.shape) * 0.499

    return np.argsort(-score, axis=1, kind="stable")


def _two_legged(grids: Grids, first: np.ndarray, second: np.ndarray, rng,
                rules: Format = DEFAULT):
    """Play a two-legged tie. ``second`` hosts the return leg.

    The away-goals rule was abolished in 2021, so a level aggregate goes
    to extra time and then penalties. ``rules.away_goals`` is honoured so
    an older format can be simulated without changing this function.
    """
    leg1_home, leg1_away = grids.sample(first, second, rng)
    leg2_home, leg2_away = grids.sample(second, first, rng)

    aggregate_first = leg1_home + leg2_away
    aggregate_second = leg1_away + leg2_home

    level = aggregate_first == aggregate_second

    if rules.away_goals:
        # Under the old rule a level aggregate was settled by goals
        # scored away from home before extra time was reached.
        away_first, away_second = leg2_away, leg1_away
        decided = level & (away_first != away_second)
        level = level & ~decided

    if level.any():
        # Extra time, at the second leg's venue and a third of the rate.
        share = rules.extra_time_share
        rate_second = grids.home_rate[second[level], first[level]] * share
        rate_first = grids.away_rate[second[level], first[level]] * share

        aggregate_second[level] += rng.poisson(rate_second)
        aggregate_first[level] += rng.poisson(rate_first)

    still_level = aggregate_first == aggregate_second
    shootout = rng.random(len(first)) < 0.5

    first_through = aggregate_first > aggregate_second
    first_through[still_level] = shootout[still_level]

    if rules.away_goals:
        first_through = np.where(decided, away_first > away_second, first_through)

    return np.where(first_through, first, second)


@dataclass
class Simulation:
    """What happened across every simulated season."""

    clubs: list[str]
    runs: int
    points: np.ndarray
    position: np.ndarray
    reached: dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = []

        for i, club in enumerate(self.clubs):
            row = {
                "club": club,
                "avg_points": self.points[:, i].mean(),
                "avg_position": self.position[:, i].mean(),
                "top_8": (self.position[:, i] <= 8).mean(),
                "top_24": (self.position[:, i] <= 24).mean(),
            }
            for stage, reached in self.reached.items():
                row[stage] = reached[:, i].mean()

            rows.append(row)

        return pd.DataFrame(rows).sort_values("wins_it", ascending=False).reset_index(drop=True)


def run(
    model,
    keys: list[str],
    fixtures: pd.DataFrame,
    runs: int = 10000,
    seed: int = 0,
    parameter_draws: int | None = None,
    rules: Format = DEFAULT,
) -> Simulation:
    """Simulate the whole competition ``runs`` times.

    ``keys`` are the 36 club keys in a fixed order; ``fixtures`` needs
    ``home`` and ``away`` columns holding indices into that list.

    Pass a set of bootstrap draws rather than a single fitted model and
    the seasons are split across them, so the output carries our
    uncertainty about how good each club is alongside the luck inside
    the matches. With one model the ratings are treated as exact, which
    is the older behaviour and remains the right comparison to measure
    the change against.
    """
    from . import draws as parameter_batches

    if parameter_batches.is_sampled(model):
        parts = [
            _run_once(parameters, keys, fixtures, count, seed + i, rules)
            for i, (parameters, count) in enumerate(
                parameter_batches.batches(model, runs, parameter_draws)
            )
        ]

        return Simulation(
            clubs=keys,
            runs=runs,
            points=np.vstack([p.points for p in parts]),
            position=np.vstack([p.position for p in parts]),
            reached={
                stage: np.vstack([p.reached[stage] for p in parts])
                for stage in parts[0].reached
            },
        )

    return _run_once(model, keys, fixtures, runs, seed, rules)


def _run_once(
    model,
    keys: list[str],
    fixtures: pd.DataFrame,
    runs: int,
    seed: int,
    rules: Format = DEFAULT,
) -> Simulation:
    """One block of seasons, all played with the same ratings."""
    if len(keys) != rules.clubs:
        raise ValueError(
            f"{rules.name} expects {rules.clubs} clubs, got {len(keys)}"
        )

    rng = np.random.default_rng(seed)

    grids = build_grids(model, keys)
    neutral = build_grids(model, keys, neutral=True)

    home = fixtures["home"].to_numpy()
    away = fixtures["away"].to_numpy()

    totals = league_phase(grids, home, away, runs, rng)
    order = rank(totals, rng)

    n_clubs = len(keys)
    position = np.empty((runs, n_clubs), dtype=int)
    position[np.arange(runs)[:, None], order] = np.arange(1, n_clubs + 1)

    stages = ["playoffs", "last_16", "quarter_finals", "semi_finals", "final", "wins_it"]
    reached = {stage: np.zeros((runs, n_clubs), dtype=bool) for stage in stages}

    rows = np.arange(runs)

    def seat(place: int) -> np.ndarray:
        """Which club finished in a given position, per run."""
        return order[:, place]

    for place in range(rules.direct_qualifiers, rules.eliminated_from):
        reached["playoffs"][rows, seat(place)] = True

    # --- knockout play-offs ------------------------------------------
    band_winners: list[list[np.ndarray]] = []

    for seeded, unseeded in rules.playoff_bands:
        top = np.stack([seat(p) for p in seeded], axis=1)
        bottom = np.stack([seat(p) for p in unseeded], axis=1)

        # The pairing inside a band is drawn.
        swap = rng.random(runs) < 0.5
        bottom = np.where(swap[:, None], bottom[:, ::-1], bottom)

        winners = [
            _two_legged(grids, bottom[:, k], top[:, k], rng, rules)
            for k in range(2)
        ]
        band_winners.append(winners)

    # --- round of 16 --------------------------------------------------
    r16_winners: list[np.ndarray] = []

    for places, band in rules.round_of_16:
        hosts = np.stack([seat(p) for p in places], axis=1)
        visitors = band_winners[band]

        swap = rng.random(runs) < 0.5

        for k in range(2):
            challenger = np.where(swap, visitors[1 - k], visitors[k])
            host = hosts[:, k]

            reached["last_16"][rows, host] = True
            reached["last_16"][rows, challenger] = True

            r16_winners.append(_two_legged(grids, challenger, host, rng, rules))

    for winner in r16_winners:
        reached["quarter_finals"][rows, winner] = True

    # --- quarter-finals, semi-finals, final ---------------------------
    quarter_winners = [
        _two_legged(grids, r16_winners[k + 1], r16_winners[k], rng, rules)
        for k in range(0, len(r16_winners), 2)
    ]
    for winner in quarter_winners:
        reached["semi_finals"][rows, winner] = True

    semi_winners = [
        _two_legged(grids, quarter_winners[k + 1], quarter_winners[k], rng, rules)
        for k in range(0, len(quarter_winners), 2)
    ]
    for winner in semi_winners:
        reached["final"][rows, winner] = True

    goals_home, goals_away = neutral.sample(semi_winners[0], semi_winners[1], rng)

    first_wins = goals_home > goals_away
    level = goals_home == goals_away

    if level.any():
        rate_first = neutral.home_rate[semi_winners[0][level], semi_winners[1][level]]
        rate_second = neutral.away_rate[semi_winners[0][level], semi_winners[1][level]]

        extra_first = rng.poisson(rate_first * rules.extra_time_share)
        extra_second = rng.poisson(rate_second * rules.extra_time_share)

        decided = np.zeros(runs, dtype=bool)
        decided[level] = extra_first > extra_second

        shootout = rng.random(runs) < 0.5
        still = np.zeros(runs, dtype=bool)
        still[level] = extra_first == extra_second

        first_wins = np.where(level, np.where(still, shootout, decided), first_wins)

    champion = np.where(first_wins, semi_winners[0], semi_winners[1])
    reached["wins_it"][rows, champion] = True

    return Simulation(
        clubs=keys,
        runs=runs,
        points=totals["points"],
        position=position,
        reached=reached,
    )
