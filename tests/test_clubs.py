"""Club-name resolution.

The cases here are the ones that actually broke: pairs a fuzzy matcher
merges, clubs football-data.co.uk files under an obsolete name, and
codes for countries with no division in the archive.
"""

import pandas as pd
import pytest

from pitchiq import clubs


@pytest.mark.parametrize(
    "name, country, expected",
    [
        # Same club, three vocabularies.
        ("FC Bayern München", "GER", "Bayern Munich"),
        ("Bayern Munich", "Germany", "Bayern Munich"),
        ("Bayern München", "GER", "Bayern Munich"),
        # The pair any similarity threshold merges.
        ("Sporting Clube de Portugal", "POR", "Sp Lisbon"),
        ("Sporting CP", "POR", "Sp Lisbon"),
        ("Sp Lisbon", "Portugal", "Sp Lisbon"),
        ("Sporting Clube de Braga", "POR", "Sp Braga"),
        ("Sp Braga", "Portugal", "Sp Braga"),
        # Renamed clubs, still filed under the old name upstream.
        ("İstanbul Başakşehir", "TUR", "Buyuksehyr"),
        ("FC Steaua Bucureşti", "ROU", "FCSB"),
        # Monaco has no league of its own and plays in Ligue 1.
        ("AS Monaco FC", "MCO", "FRA:monaco"),
        ("Monaco", "France", "FRA:monaco"),
        # Countries with no domestic division still need stable keys.
        ("SK Slavia Praha", "CZE", "Slavia Prague"),
        ("Slavia Prague", "CZE", "Slavia Prague"),
        ("ŠK Slovan Bratislava", "SVK", "Slovan Bratislava"),
        # No alias needed: agreement after normalisation is enough.
        ("Galatasaray SK", "TUR", "TUR:galatasaray"),
        ("Galatasaray", "Turkey", "TUR:galatasaray"),
        ("FK Bodø/Glimt", "NOR", "NOR:bodo glimt"),
        ("Bodo/Glimt", "Norway", "NOR:bodo glimt"),
    ],
)
def test_resolves(name, country, expected):
    assert clubs.resolve(name, country) == expected


def test_distinct_clubs_stay_distinct():
    """Sporting Lisbon and Sporting Braga must never collapse."""
    assert clubs.resolve("Sporting Clube de Portugal", "POR") != clubs.resolve(
        "Sporting Clube de Braga", "POR"
    )


def test_reserve_teams_are_separate():
    """'Ath Madrid B' is a different side from 'Ath Madrid'."""
    assert clubs.resolve("Ath Madrid B", "Spain") != clubs.resolve(
        "Ath Madrid", "Spain"
    )


def test_same_name_different_country():
    """Country scoping keeps unrelated same-named clubs apart."""
    assert clubs.resolve("Rangers", "Scotland") != clubs.resolve("Rangers", "USA")


def test_empty_name_is_unresolved():
    assert clubs.resolve("", "ENG") is None
    assert clubs.resolve("   ", "ENG") is None


def test_alias_table_has_no_collisions():
    """_build_index raises if two clubs claim one normalised form."""
    assert len(clubs.INDEX) > 0


def test_every_2026_27_club_resolves():
    squad = pd.read_csv("data/external/ucl_2026_27_clubs.csv")

    keys = {clubs.resolve(n, c) for n, c in zip(squad.club, squad.country)}

    assert len(squad) == 36
    assert None not in keys
    assert len(keys) == 36, "two clubs resolved to one key"


def test_no_ucl_club_splits_from_its_domestic_self():
    """A UCL side from a covered country must join its domestic record."""
    ucl = pd.read_parquet("data/processed/uefa_matches.parquet")
    dom = pd.read_parquet("data/processed/domestic_matches.parquet")

    domestic_keys = set()
    for country, grp in dom.groupby("country"):
        names = set(grp.home_team.dropna()) | set(grp.away_team.dropna())
        domestic_keys.update(clubs.resolve(n, country) for n in names)

    splits = []
    for col, ccol in [("home_team", "home_country"), ("away_team", "away_country")]:
        for name, country in set(zip(ucl[col], ucl[ccol])):
            if country not in clubs.COUNTRY_NAMES:
                continue
            key = clubs.resolve(name, country)
            if key not in domestic_keys:
                splits.append((country, name, key))

    assert not splits, f"unjoined UCL clubs: {sorted(set(splits))}"


@pytest.mark.parametrize(
    "name, country, expected",
    [
        # Europa and Conference clubs, reached once those competitions
        # were added.
        ("Heart of Midlothian", "SCO", "Hearts"),
        ("BK Häcken", "SWE", "Hacken"),
        ("Kuopion PS", "FIN", "KuPS"),
        ("PAOK Saloniki", "GRE", "PAOK"),
        ("1. FSV Mainz 05", "GER", "Mainz"),
        # Renamed mid-archive: football-data.co.uk switched spelling in
        # February 2021 and both must land on one club.
        ("U Craiova", "Romania", "U Craiova"),
        ("Univ. Craiova", "Romania", "U Craiova"),
        ("CS Universitatea Craiova", "ROU", "U Craiova"),
    ],
)
def test_resolves_europa_era(name, country, expected):
    assert clubs.resolve(name, country) == expected


@pytest.mark.parametrize(
    "left, right, country",
    [
        # Dropping the founding year would merge these, and they are
        # different clubs.
        ("U Craiova", "U Craiova 1948", "Romania"),
        ("Granada", "Granada 74", "Spain"),
        # Wimbledon FC became MK Dons; AFC Wimbledon is a separate club.
        ("Wimbledon", "AFC Wimbledon", "England"),
        # Two unrelated Wislas and two unrelated Cluj sides.
        ("Wisla", "Wisla Plock", "Poland"),
        ("U. Cluj", "CFR Cluj", "Romania"),
    ],
)
def test_namesakes_stay_apart(left, right, country):
    assert clubs.resolve(left, country) != clubs.resolve(right, country)


def test_no_domestic_namesakes_collide():
    """No two distinct domestic club names may share a resolved key.

    Guards the case that the founding-year rule got wrong: a normaliser
    change that merges two real clubs is silent and corrupts every
    rating for both.
    """
    from collections import defaultdict

    dom = pd.read_parquet("data/processed/domestic_matches.parquet")

    collisions = {}
    for country, grp in dom.groupby("country"):
        names = set(grp.home_team.dropna()) | set(grp.away_team.dropna())

        buckets = defaultdict(set)
        for name in names:
            buckets[clubs.resolve(name, country)].add(name)

        for key, group in buckets.items():
            if len(group) > 1:
                collisions[(country, key)] = sorted(group)

    # One club, several spellings over the years. Each was checked
    # against its match dates: the spellings are consecutive, never
    # concurrent, which is what a rename looks like in this archive.
    intentional = {
        ("Belgium", "BEL:lommel"),          # Lommel -> Lommel SK
        ("England", "ENG:telford united"),  # Telford United -> AFC Telford United
        ("France", "FRA:red star"),         # Red Star 93 -> Red Star
        ("Romania", "U Craiova"),           # renamed mid-2020/21
    }

    # Spellings differing only in case or punctuation are the same club.
    real = {
        k: v for k, v in collisions.items()
        if k not in intentional
        and len({n.lower().replace(".", "").replace("'", "") for n in v}) > 1
    }

    assert not real, f"distinct clubs sharing a key: {real}"
