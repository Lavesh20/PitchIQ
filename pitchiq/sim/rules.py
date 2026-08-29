"""Competition format as versioned data, not as constants in the simulator.

UEFA rewrote the Champions League format for 2024/25 — 36 clubs in one
league phase instead of 32 in eight groups — and adjusts details between
seasons. Every time that happens, a simulator with the bracket baked
into its code has to be edited, and the old behaviour is lost rather
than kept alongside the new one.

So the format lives here, keyed by season, and the simulator reads it.
Adding next year's rules means adding an entry, not rewriting a bracket.

What is deliberately *not* here: any difference between 2024/25 and
2026/27. Both are registered against the same format because no
difference between them was verified. Inventing one to make the config
look richer would be worse than a single shared entry, since the
simulator would then be reproducing a rule that may not exist.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    """Everything about a competition's shape that the simulator needs."""

    name: str
    clubs: int
    matches_per_club: int

    # Positions 1..direct_qualifiers go straight through; the next
    # playoff_places contest a two-legged tie; the rest are out.
    direct_qualifiers: int
    playoff_places: int

    # Play-off pairings as zero-based finishing positions: each entry is
    # (seeded pair, unseeded pair). The tie inside a band is drawn, which
    # the simulation randomises.
    playoff_bands: tuple[tuple[tuple[int, int], tuple[int, int]], ...]

    # The round of 16 in bracket order: each entry pairs a band of direct
    # qualifiers with the index of the play-off band they meet. Ties 0
    # and 1 feed one quarter-final, 2 and 3 the next, so first and eighth
    # can only meet in the last eight.
    round_of_16: tuple[tuple[tuple[int, int], int], ...]

    # Abolished in 2021. Kept as a field so an older format can be added
    # without changing the simulator.
    away_goals: bool

    # Extra time is thirty minutes against ninety.
    extra_time_share: float = 30.0 / 90.0

    @property
    def eliminated_from(self) -> int:
        """First position, zero-based, that goes out at the league phase."""
        return self.direct_qualifiers + self.playoff_places

    def validate(self) -> None:
        """Check the bracket actually consumes the clubs it claims to."""
        if self.eliminated_from > self.clubs:
            raise ValueError(
                f"{self.name}: {self.direct_qualifiers} direct and "
                f"{self.playoff_places} play-off places exceed {self.clubs} clubs"
            )

        seeded = [p for band, _ in self.playoff_bands for p in band]
        unseeded = [p for _, band in self.playoff_bands for p in band]
        places = sorted(seeded + unseeded)

        if places != list(range(self.direct_qualifiers, self.eliminated_from)):
            raise ValueError(
                f"{self.name}: play-off bands do not cover positions "
                f"{self.direct_qualifiers}..{self.eliminated_from - 1} exactly"
            )

        hosts = sorted(p for band, _ in self.round_of_16 for p in band)

        if hosts != list(range(self.direct_qualifiers)):
            raise ValueError(
                f"{self.name}: round of 16 hosts do not cover the "
                f"{self.direct_qualifiers} direct qualifiers exactly"
            )

        met = sorted(band for _, band in self.round_of_16)

        if met != list(range(len(self.playoff_bands))):
            raise ValueError(
                f"{self.name}: each play-off band must feed the round of 16 once"
            )


LEAGUE_PHASE_36 = Format(
    name="league phase, 36 clubs",
    clubs=36,
    matches_per_club=8,
    direct_qualifiers=8,
    playoff_places=16,
    playoff_bands=(
        ((8, 9), (22, 23)),    # positions 9/10 v 23/24
        ((10, 11), (20, 21)),  # 11/12 v 21/22
        ((12, 13), (18, 19)),  # 13/14 v 19/20
        ((14, 15), (16, 17)),  # 15/16 v 17/18
    ),
    round_of_16=(
        ((0, 1), 3),   # 1/2 v winners from the 15/16-17/18 band
        ((6, 7), 0),   # 7/8 v winners from the 9/10-23/24 band
        ((2, 3), 2),   # 3/4 v winners from the 13/14-19/20 band
        ((4, 5), 1),   # 5/6 v winners from the 11/12-21/22 band
    ),
    away_goals=False,
)

BY_SEASON = {
    "2024/25": LEAGUE_PHASE_36,
    "2025/26": LEAGUE_PHASE_36,
    "2026/27": LEAGUE_PHASE_36,
}

DEFAULT = LEAGUE_PHASE_36


def for_season(season: str) -> Format:
    """The format in force for a season.

    Raises rather than falling back to the newest rules. A season we
    have no entry for is a season whose rules nobody has checked, and
    silently simulating it under this year's format would produce a
    confident answer to a question we cannot answer.
    """
    if season not in BY_SEASON:
        raise KeyError(
            f"no format registered for {season!r}; known seasons are "
            f"{sorted(BY_SEASON)}. Add an entry rather than assuming the "
            "current rules still apply."
        )

    return BY_SEASON[season]


for _format in set(BY_SEASON.values()):
    _format.validate()
