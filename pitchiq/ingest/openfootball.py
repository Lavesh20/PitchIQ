"""Download and parse openfootball's UEFA club-competition archive.

One repository holds all three competitions plus their qualifying
rounds, as plain text in the football.txt format, one file per season
and competition. It is the only free source with history this deep, and
unlike the football-data.org free tier it has no season paywall.

Qualifying rounds matter more than their profile suggests. Domestic
matches say nothing about how leagues compare -- a club dominating the
Azerbaijani league looks exactly like one dominating La Liga -- so the
only evidence tying leagues to a common scale is clubs from different
leagues playing each other. The qualifying rounds are where the small
federations actually appear, which makes them the densest source of
that evidence in the archive.

Two things vary across the archive and both are handled here.

Section headers come in two grammars. Pre-2023/24 files name the round
on its own ("Round of 16"); later files prefix a phase ("Finals, Round
of 16", "League, Matchday 3"). Seasons 2020/21 to 2022/23 also slipped
into German for the last two groups and say "Gruppe G".

Score lines carry extra time and shootouts inline, so a knockout leg can
read ``1-4 pen. 0-1 a.e.t. (0-1, 0-1)``. Naive parsing records the
shootout as the match score; :func:`_parse_tail` pulls the pieces apart
and keeps regulation-time goals as the modelling target.
"""

from __future__ import annotations

import re
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/openfootball/champions-league/master"
RAW = Path("data/raw/openfootball")

# Repository file stem -> competition code. The repo is named after the
# Champions League but carries the Europa and Conference Leagues too.
COMPETITIONS: dict[str, str] = {
    "cl": "UCL",
    "el": "UEL",
    "conf": "UECL",
    "clq": "UCL_Q",
    "elq": "UEL_Q",
    "confq": "UECL_Q",
}

# Which seasons actually exist upstream, so we do not hammer the host
# with requests for files that were never published.
AVAILABLE: dict[str, range] = {
    "cl": range(2011, 2026),
    "el": range(2020, 2025),
    "conf": range(2021, 2025),
    "clq": range(2024, 2026),
    "elq": range(2024, 2026),
    "confq": range(2024, 2026),
}

PHASE_STAGES: dict[str, str] = {
    "group": "GROUP_STAGE",
    "gruppe": "GROUP_STAGE",
    "league": "LEAGUE_PHASE",
    "playoffs": "PLAYOFFS",
    "play-offs": "PLAYOFFS",
}

# Matched on a lowercase prefix so "Round of 16, 1st leg" and "Round of
# 16" agree. Order matters: "quarter" and "semi" must be tested before
# the bare "final", or Quarterfinals lands in FINAL.
ROUND_PREFIXES: list[tuple[str, str]] = [
    ("round of 32", "LAST_32"),
    ("sechzehntelfinale", "LAST_32"),
    ("round of 16", "LAST_16"),
    ("last 16", "LAST_16"),
    ("achtelfinale", "LAST_16"),
    ("quarter", "QUARTER_FINALS"),
    ("viertelfinale", "QUARTER_FINALS"),
    ("semi", "SEMI_FINALS"),
    ("halbfinale", "SEMI_FINALS"),
    ("third", "THIRD_PLACE"),
    ("play-off", "PLAYOFFS"),
    ("playoff", "PLAYOFFS"),
    ("knockout", "PLAYOFFS"),
    ("final", "FINAL"),
]

SECTION_RE = re.compile(r"^\s*▪\s*(.+)$")

DATE_RE = re.compile(
    r"^\s{2,}(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"([A-Z][a-z]{2}\s+\d{1,2})(?:\s+(\d{4}))?\s*$"
)

KICKOFF_RE = re.compile(r"^\s*\d{1,2}[:.]\d{2}\s+")

SPLIT_RE = re.compile(r"\s+v\s+")

# The score tail, in full generality:
#     2-1 (1-0)
#     1-4 pen. 0-1 a.e.t. (0-1, 0-1)
#     0-0
# When a.e.t. is present the leading pair is the after-extra-time score
# and the parenthesised pairs are (90 minutes, half time) -- that
# ordering is the only one under which goals never decrease, checked
# against every such line in the archive. Without a.e.t. the leading
# pair is full time and the single parenthesised pair is half time.
TAIL_RE = re.compile(
    r"(?:(?P<ph>\d+)-(?P<pa>\d+)\s+pen\.\s+)?"
    r"(?P<h>\d+)-(?P<a>\d+)"
    r"(?P<aet>\s+a\.e\.t\.)?"
    r"(?:\s+\((?P<p1h>\d+)-(?P<p1a>\d+)(?:,\s*(?P<p2h>\d+)-(?P<p2a>\d+))?\))?"
    r"\s*$"
)

COUNTRY_RE = re.compile(r"\s*\(([A-Z]{3})\)\s*$")

GROUP_RE = re.compile(r"^(?:group|gruppe)\s+([A-L])$", re.IGNORECASE)

MATCHDAY_RE = re.compile(r"matchday\s+(\d+)", re.IGNORECASE)

# Qualifying files head their sections "Round 1" or "1. Round"
# depending on the season.
QUAL_ROUND_RE = re.compile(
    r"^(?:round\s+(\d+)|(\d+)\.\s*round)$", re.IGNORECASE
)


def season_slug(start_year: int) -> str:
    """2011 -> '2011-12'. openfootball's directory name."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2011 -> '2011/12'. Our canonical season string."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def download(
    competitions: list[str] | None = None,
    pause: float = 0.3,
) -> dict[tuple[str, str], bool]:
    """Mirror every published season file for the given competitions."""
    competitions = competitions or list(COMPETITIONS)
    session = requests.Session()
    results: dict[tuple[str, str], bool] = {}

    for stem in competitions:
        target = RAW / COMPETITIONS[stem]
        target.mkdir(parents=True, exist_ok=True)

        for year in AVAILABLE[stem]:
            slug = season_slug(year)

            try:
                response = session.get(f"{BASE}/{slug}/{stem}.txt", timeout=30)
            except requests.RequestException:
                results[(stem, slug)] = False
                continue

            ok = response.status_code == 200 and bool(response.content.strip())

            if ok:
                (target / f"{slug}.txt").write_bytes(response.content)

            results[(stem, slug)] = ok
            time.sleep(pause)

    return results


def _classify(section: str) -> tuple[str, str | None, int | None]:
    """Section header -> (stage, group, matchday).

    "Group A" carries a group label; "Group, Matchday 1" (2023/24 only)
    carries a matchday but no group, because that season's file dropped
    the group letters entirely.
    """
    parts = [p.strip() for p in section.split(",")]

    group = GROUP_RE.match(parts[0])

    if group and len(parts) == 1:
        return "GROUP_STAGE", group.group(1).upper(), None

    qualifier = QUAL_ROUND_RE.match(parts[0])

    if qualifier:
        number = qualifier.group(1) or qualifier.group(2)

        return f"QUAL_ROUND_{number}", None, None

    matchday = MATCHDAY_RE.search(section)

    if matchday:
        phase = PHASE_STAGES.get(parts[0].lower())

        if phase:
            return phase, None, int(matchday.group(1))

    # Otherwise the round name is the last comma-separated part; a
    # leading "Finals" is a grouping label and carries no meaning.
    tail = parts[-1].lower()

    for prefix, stage in ROUND_PREFIXES:
        if tail.startswith(prefix):
            return stage, None, None

    return section.upper().replace(" ", "_"), None, None


def _parse_tail(text: str) -> tuple[int, dict] | None:
    """Split a score tail into regulation, extra-time and shootout scores.

    Returns ``(start, scores)`` where ``start`` is the index at which the
    tail begins -- everything before it is the away team -- or ``None``
    when the line carries no score.
    """
    found = TAIL_RE.search(text)

    if not found:
        return None

    lead = (int(found.group("h")), int(found.group("a")))

    first = (
        (int(found.group("p1h")), int(found.group("p1a")))
        if found.group("p1h")
        else None
    )
    second = (
        (int(found.group("p2h")), int(found.group("p2a")))
        if found.group("p2h")
        else None
    )

    if found.group("aet"):
        aet, full, half = lead, first or lead, second
    else:
        aet, full, half = None, lead, first

    pens = (
        (int(found.group("ph")), int(found.group("pa")))
        if found.group("ph")
        else None
    )

    return found.start(), {
        # Models are fitted on regulation-time goals, so fthg/ftag stay
        # the 90-minute score even when a tie went to extra time.
        "fthg": full[0],
        "ftag": full[1],
        "hthg": half[0] if half else None,
        "htag": half[1] if half else None,
        "aet_home": aet[0] if aet else None,
        "aet_away": aet[1] if aet else None,
        "pen_home": pens[0] if pens else None,
        "pen_away": pens[1] if pens else None,
    }


def _split_country(name: str) -> tuple[str, str | None]:
    """'Arsenal FC (ENG)' -> ('Arsenal FC', 'ENG')."""
    found = COUNTRY_RE.search(name)

    if not found:
        return name.strip(), None

    return COUNTRY_RE.sub("", name).strip(), found.group(1)


def parse_file(path: Path, start_year: int, competition: str) -> list[dict]:
    """Parse one cl.txt into match records."""
    rows: list[dict] = []

    stage: str | None = None
    group: str | None = None
    matchday: int | None = None
    current: date | None = None

    # Files only restate the year when it rolls over, so we carry it.
    year = start_year

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        section = SECTION_RE.match(line)

        if section:
            stage, group, matchday = _classify(section.group(1).strip())
            continue

        stamp = DATE_RE.match(line)

        if stamp:
            if stamp.group(2):
                year = int(stamp.group(2))

            current = datetime.strptime(
                f"{stamp.group(1)} {year}", "%b %d %Y"
            ).date()
            continue

        body = KICKOFF_RE.sub("", line).strip()

        halves = SPLIT_RE.split(body, maxsplit=1)

        if len(halves) != 2:
            continue

        tail = _parse_tail(halves[1])

        if tail is None:
            continue

        cut, scores = tail

        home, home_country = _split_country(halves[0])
        away, away_country = _split_country(halves[1][:cut])

        if not home or not away:
            continue

        rows.append(
            {
                "competition": competition,
                "season": season_label(start_year),
                "date": current,
                "stage": stage,
                "group": group,
                "matchday": matchday,
                "home_team": home,
                "away_team": away,
                "home_country": home_country,
                "away_country": away_country,
                **scores,
            }
        )

    return rows


def build(out_path: Path = Path("data/processed/uefa_matches.parquet")) -> pd.DataFrame:
    """Parse every mirrored file into one frame."""
    rows: list[dict] = []

    for competition in sorted(COMPETITIONS.values()):
        folder = RAW / competition

        if not folder.is_dir():
            continue

        for path in sorted(folder.glob("*.txt")):
            found = re.match(r"^(\d{4})-\d{2}$", path.stem)

            if not found:
                continue

            rows.extend(parse_file(path, int(found.group(1)), competition))

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["source"] = "openfootball"

    # Result of the 90 minutes, matching fthg/ftag. A tie decided in
    # extra time or on penalties still shows as a draw here, which is
    # what a goals model should be scored against.
    df["ftr"] = "D"
    df.loc[df["fthg"] > df["ftag"], "ftr"] = "H"
    df.loc[df["fthg"] < df["ftag"], "ftr"] = "A"

    df = df.sort_values(["date", "competition", "home_team"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return df
