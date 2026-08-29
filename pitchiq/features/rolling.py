"""Pre-match features, built in one causal pass.

Every number a model is shown for a match has to have been knowable
before kick-off. The safe way to guarantee that is not to compute
rolling windows in pandas and trust that the ``shift`` is right — that
is the single easiest way to build a model that looks superb and is
worthless. Instead this module walks the fixture list in date order
holding per-club state, and for each match does exactly two things in
this order:

1. read the state and write the feature row,
2. fold the result into the state.

A match therefore cannot see itself, and cannot see any later match,
because neither has reached the state when its row is written. The
property is structural, not a convention to be maintained.

Form windows span competitions. A club that played 120 minutes in a
European tie on Wednesday is tired on Saturday, and its league form is
part of the same story, so the deques hold every match a club plays.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

POINTS_WIN = 3.0
POINTS_DRAW = 1.0


def points_for(goals_for: float, goals_against: float) -> float:
    if goals_for > goals_against:
        return POINTS_WIN

    if goals_for == goals_against:
        return POINTS_DRAW

    return 0.0


@dataclass(frozen=True)
class RollingConfig:
    """Window sizes, in matches or days.

    ``form`` windows are short/medium/long reads of the same thing: 3 is
    a streak, 10 is closer to a season's standing. ``goals`` uses the
    longer two only, because goals over three matches is mostly noise.
    ``detail`` is deliberately the longest window: shots are recorded
    for barely half the matches we hold, so a short window would return
    NaN far too often to be usable.
    """

    form: tuple[int, ...] = (3, 5, 10)
    goals: tuple[int, ...] = (5, 10)
    detail: int = 10
    venue: int = 5
    head_to_head: int = 5
    congestion_days: int = 14


@dataclass(frozen=True, slots=True)
class _Played:
    """One completed match, from one club's point of view."""

    date: pd.Timestamp
    points: float
    goals_for: float
    goals_against: float
    shots_for: float
    shots_against: float
    sot_for: float
    sot_against: float


@dataclass
class _Club:
    """What we remember about a club going into its next match."""

    config: RollingConfig
    recent: deque = field(init=False)
    at_home: deque = field(init=False)
    away: deque = field(init=False)
    played: int = 0

    def __post_init__(self) -> None:
        # ``recent`` has to be long enough for both the longest form
        # window and the congestion count; no side plays more than a
        # dozen matches in a fortnight.
        span = max(max(self.config.form), max(self.config.goals), self.config.detail)
        self.recent = deque(maxlen=max(span, 12))
        # Venue records get their own deques rather than being filtered
        # out of ``recent``: a club can play six straight away ties, and
        # its last five home matches would fall off the end.
        self.at_home = deque(maxlen=self.config.venue)
        self.away = deque(maxlen=self.config.venue)

    def add(self, record: _Played, home: bool) -> None:
        self.recent.append(record)
        (self.at_home if home else self.away).append(record)
        self.played += 1


def _mean(values: list[float]) -> float:
    """Average of the values that exist, NaN when none do.

    Sparse columns are the norm here, so an all-missing window has to
    come back as NaN. A zero would tell a model that a side managed no
    shots, when the truth is that nobody wrote them down.
    """
    present = [v for v in values if v == v]

    return float(np.mean(present)) if present else np.nan


def _window(records: deque, size: int) -> list[_Played]:
    if size >= len(records):
        return list(records)

    return list(records)[-size:]


def _side(club: _Club, venue: deque, kickoff, config: RollingConfig) -> dict:
    """The feature block for one side of one match."""
    out: dict[str, float] = {}

    for n in config.form:
        window = _window(club.recent, n)
        out[f"form_pts_{n}"] = _mean([r.points for r in window])

    for n in config.goals:
        window = _window(club.recent, n)
        out[f"gf_{n}"] = _mean([r.goals_for for r in window])
        out[f"ga_{n}"] = _mean([r.goals_against for r in window])

    detail = _window(club.recent, config.detail)
    out["shots_for"] = _mean([r.shots_for for r in detail])
    out["shots_against"] = _mean([r.shots_against for r in detail])
    out["sot_for"] = _mean([r.sot_for for r in detail])
    out["sot_against"] = _mean([r.sot_against for r in detail])

    out["venue_pts"] = _mean([r.points for r in venue])
    out["venue_gf"] = _mean([r.goals_for for r in venue])
    out["venue_ga"] = _mean([r.goals_against for r in venue])

    if club.recent:
        last = club.recent[-1].date
        out["rest_days"] = float((kickoff - last).days)
        cutoff = kickoff - pd.Timedelta(days=config.congestion_days)
        out["matches_recent"] = float(sum(r.date > cutoff for r in club.recent))
    else:
        # No history at all. NaN rather than a large rest figure: the
        # honest statement is that we do not know, and a tree can split
        # on missingness.
        out["rest_days"] = np.nan
        out["matches_recent"] = np.nan

    # How much history the rest of the row rests on. Form over three
    # matches means one thing for a club with 200 behind it and another
    # for a promoted side we have barely seen.
    out["played"] = float(club.played)

    return out


# Features that describe the same quantity for both sides, and so are
# worth handing over as a difference too. A tree splits one feature at a
# time, so it approximates ``home - away`` only through many splits;
# giving it the subtraction directly is cheap and it is usually the
# comparison that actually carries the signal.
DIFFED = (
    "form_pts_5", "form_pts_10", "gf_10", "ga_10",
    "sot_for", "sot_against", "rest_days",
)


class _HeadToHead:
    """Recent meetings between a pair, stored once per unordered pair."""

    def __init__(self, config: RollingConfig):
        self.config = config
        self.pairs: dict[tuple[str, str], deque] = {}

    @staticmethod
    def _key(home: str, away: str) -> tuple[tuple[str, str], bool]:
        # One deque per pair, recorded from the point of view of whichever
        # club sorts first, plus a flag saying whether today's home side
        # is that club. Without the flag a 3-0 win would read as a 0-3.
        if home <= away:
            return (home, away), True

        return (away, home), False

    def read(self, home: str, away: str) -> dict:
        key, home_is_first = self._key(home, away)
        meetings = self.pairs.get(key)

        if not meetings:
            return {"h2h_n": 0.0, "h2h_pts": np.nan, "h2h_gf": np.nan, "h2h_ga": np.nan}

        # Stored goals belong to the first club of the pair. Orient them
        # to today's home side, then derive points; deriving is safer
        # than storing points and flipping them, because a draw is one
        # point from both sides and no arithmetic flip preserves that.
        if home_is_first:
            oriented = [(gf, ga) for gf, ga in meetings]
        else:
            oriented = [(ga, gf) for gf, ga in meetings]

        return {
            "h2h_n": float(len(oriented)),
            "h2h_pts": _mean([points_for(gf, ga) for gf, ga in oriented]),
            "h2h_gf": _mean([gf for gf, _ in oriented]),
            "h2h_ga": _mean([ga for _, ga in oriented]),
        }

    def add(self, home: str, away: str, goals_home: int, goals_away: int) -> None:
        key, home_is_first = self._key(home, away)
        meetings = self.pairs.setdefault(
            key, deque(maxlen=self.config.head_to_head)
        )

        if home_is_first:
            meetings.append((float(goals_home), float(goals_away)))
        else:
            meetings.append((float(goals_away), float(goals_home)))


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """A stat column as floats, or all-NaN when the source omitted it."""
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)

    return np.full(len(frame), np.nan)


def build(matches: pd.DataFrame, config: RollingConfig | None = None) -> pd.DataFrame:
    """Pre-match rolling features for every row of a date-ordered frame.

    ``matches`` must be sorted by date — the one pass assumes it, and an
    out-of-order frame would let a club's later result inform an earlier
    match. :func:`pitchiq.matches.load` returns the frame sorted.
    """
    config = config or RollingConfig()

    if not matches["date"].is_monotonic_increasing:
        raise ValueError(
            "matches must be sorted by date: the causal pass reads state "
            "in row order, so an unsorted frame would leak later results"
        )

    clubs: dict[str, _Club] = {}
    head_to_head = _HeadToHead(config)
    rows: list[dict] = []

    columns = zip(
        matches["home_key"].to_numpy(),
        matches["away_key"].to_numpy(),
        matches["date"].to_numpy(),
        matches["fthg"].to_numpy(),
        matches["ftag"].to_numpy(),
        _column(matches, "home_shots"),
        _column(matches, "away_shots"),
        _column(matches, "home_sot"),
        _column(matches, "away_sot"),
    )

    for (home, away, date, goals_home, goals_away,
         shots_home, shots_away, sot_home, sot_away) in columns:

        kickoff = pd.Timestamp(date)

        for key in (home, away):
            if key not in clubs:
                clubs[key] = _Club(config)

        home_club = clubs[home]
        away_club = clubs[away]

        row: dict[str, float] = {}

        home_side = _side(home_club, home_club.at_home, kickoff, config)
        away_side = _side(away_club, away_club.away, kickoff, config)

        for name, value in home_side.items():
            row[f"home_{name}"] = value

        for name, value in away_side.items():
            row[f"away_{name}"] = value

        for name in DIFFED:
            row[f"{name}_diff"] = home_side[name] - away_side[name]

        row.update(head_to_head.read(home, away))

        rows.append(row)

        # --- state update: nothing above this line may see the result ---

        home_points = points_for(goals_home, goals_away)
        away_points = points_for(goals_away, goals_home)

        home_club.add(
            _Played(
                date=kickoff,
                points=home_points,
                goals_for=float(goals_home),
                goals_against=float(goals_away),
                shots_for=shots_home,
                shots_against=shots_away,
                sot_for=sot_home,
                sot_against=sot_away,
            ),
            home=True,
        )
        away_club.add(
            _Played(
                date=kickoff,
                points=away_points,
                goals_for=float(goals_away),
                goals_against=float(goals_home),
                shots_for=shots_away,
                shots_against=shots_home,
                sot_for=sot_away,
                sot_against=sot_home,
            ),
            home=False,
        )

        head_to_head.add(home, away, goals_home, goals_away)

    return pd.DataFrame(rows, index=matches.index)
