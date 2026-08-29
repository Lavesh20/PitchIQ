"""Spread simulated seasons across parameter draws.

A simulator that fixes the ratings answers "how do results vary given
that this is exactly how good every club is". That is only half the
question. The other half is that we do not know how good every club is,
and for clubs we have barely seen we scarcely know at all.

So a run of 10,000 seasons is split into batches, each played with a
different set of ratings drawn from the bootstrap. Both sources of
variation end up in the same output: luck within the matches, and our
own uncertainty about the teams playing them.

Passing a single fitted model still works and still means "take these
ratings as given" — useful as the comparison, and the only option where
no bootstrap has been run.
"""

from __future__ import annotations

from typing import Iterator


def is_sampled(model) -> bool:
    """Whether this is a set of draws rather than one fitted model."""
    return hasattr(model, "draw") and hasattr(model, "draws")


def batches(model, runs: int, draws: int | None = None) -> Iterator[tuple[object, int]]:
    """Yield ``(model, runs)`` pairs covering ``runs`` seasons in total.

    Each batch costs a fresh set of scoreline grids, which is where the
    time goes, so the number of draws is worth choosing deliberately:
    enough to represent the spread of plausible ratings, not so many
    that most of the work is rebuilding grids for a handful of seasons
    each.
    """
    if not is_sampled(model):
        yield model, runs
        return

    n = min(draws or model.draws, model.draws, runs)

    # Spread the remainder rather than giving the last draw a short
    # batch: an uneven split would weight some parameter sets more than
    # others for no reason.
    base, extra = divmod(runs, n)

    for i in range(n):
        count = base + (1 if i < extra else 0)

        if count:
            yield model.draw(i), count
