"""Download and normalise football-data.co.uk match data.

Two product lines live on that site:

*main*  https://www.football-data.co.uk/mmz4281/{season}/{div}.csv
        22 European divisions, 1993/94 onward, with match stats
        (shots, corners, cards) and a wide book of bookmaker odds.

*extra* https://www.football-data.co.uk/new/{country}.csv
        16 further leagues, 2012/13 onward, one file per country,
        closing odds only and no match stats.

Both are normalised onto the schema in ``COLUMNS`` so downstream code
never has to care which line a row came from.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://www.football-data.co.uk"
RAW = Path("data/raw/football_data_uk")

# Division code -> (country, league name, tier). Tier feeds the
# league-strength prior; the model needs to know a second division is
# not a first division.
DIVISIONS: dict[str, tuple[str, str, int]] = {
    "E0": ("England", "Premier League", 1),
    "E1": ("England", "Championship", 2),
    "E2": ("England", "League One", 3),
    "E3": ("England", "League Two", 4),
    "EC": ("England", "National League", 5),
    "SC0": ("Scotland", "Premiership", 1),
    "SC1": ("Scotland", "Championship", 2),
    "SC2": ("Scotland", "League One", 3),
    "SC3": ("Scotland", "League Two", 4),
    "D1": ("Germany", "Bundesliga", 1),
    "D2": ("Germany", "2. Bundesliga", 2),
    "I1": ("Italy", "Serie A", 1),
    "I2": ("Italy", "Serie B", 2),
    "SP1": ("Spain", "La Liga", 1),
    "SP2": ("Spain", "Segunda Division", 2),
    "F1": ("France", "Ligue 1", 1),
    "F2": ("France", "Ligue 2", 2),
    "N1": ("Netherlands", "Eredivisie", 1),
    "B1": ("Belgium", "Pro League", 1),
    "P1": ("Portugal", "Primeira Liga", 1),
    "T1": ("Turkey", "Super Lig", 1),
    "G1": ("Greece", "Super League", 1),
}

# The /new/ line is one file per country covering every season at once.
EXTRA_COUNTRIES: dict[str, str] = {
    "ARG": "Argentina", "AUT": "Austria", "BRA": "Brazil", "CHN": "China",
    "DNK": "Denmark", "FIN": "Finland", "IRL": "Ireland", "JPN": "Japan",
    "MEX": "Mexico", "NOR": "Norway", "POL": "Poland", "ROU": "Romania",
    "RUS": "Russia", "SWE": "Sweden", "SWZ": "Switzerland", "USA": "USA",
}

# Odds columns we keep. The ``C`` infix means closing; closing lines are
# the sharpest public forecast available and are what we benchmark
# against. Pinnacle (PSC) is the sharp book, Avg is the market consensus,
# Max is best available price and only matters for staking simulation.
ODDS_COLUMNS = [
    "PSCH", "PSCD", "PSCA",
    "AvgCH", "AvgCD", "AvgCA",
    "MaxCH", "MaxCD", "MaxCA",
    "B365CH", "B365CD", "B365CA",
    "PSH", "PSD", "PSA",
    "AvgH", "AvgD", "AvgA",
    "B365H", "B365D", "B365A",
    "AHCh", "AvgCAHH", "AvgCAHA",
    "AvgC>2.5", "AvgC<2.5",
]

STAT_COLUMNS = [
    "HS", "AS", "HST", "AST",
    "HC", "AC", "HF", "AF",
    "HY", "AY", "HR", "AR",
    "HxG", "AxG",
]

CORE_COLUMNS = [
    "source", "country", "league", "tier", "div", "season",
    "date", "home_team", "away_team",
    "fthg", "ftag", "ftr", "hthg", "htag",
]


@dataclass(frozen=True)
class Fetch:
    """One file we tried to pull."""

    url: str
    path: Path
    ok: bool
    status: int
    n_bytes: int


def season_code(start_year: int) -> str:
    """1993 -> '9394'. The site's four-digit season slug."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """1993 -> '1993/94'. Our canonical season string."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def _download(url: str, path: Path, session: requests.Session) -> Fetch:
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = session.get(url, timeout=60)
    except requests.RequestException:
        return Fetch(url, path, False, 0, 0)

    # The site answers HTTP 300 for a season/division that does not exist
    # yet, so anything other than a 200 means "no data", not "broken".
    if response.status_code != 200 or not response.content.strip():
        return Fetch(url, path, False, response.status_code, 0)

    path.write_bytes(response.content)

    return Fetch(url, path, True, 200, len(response.content))


def download_main(
    start: int = 1993,
    end: int = 2026,
    divisions: list[str] | None = None,
    pause: float = 0.15,
) -> list[Fetch]:
    """Pull every {season, division} CSV from the mmz4281 line."""
    divisions = divisions or list(DIVISIONS)
    session = requests.Session()
    results = []

    for year in range(start, end + 1):
        code = season_code(year)

        for div in divisions:
            url = f"{BASE}/mmz4281/{code}/{div}.csv"
            path = RAW / "main" / code / f"{div}.csv"

            results.append(_download(url, path, session))
            time.sleep(pause)

    return results


def download_extra(pause: float = 0.15) -> list[Fetch]:
    """Pull the one-file-per-country /new/ line."""
    session = requests.Session()
    results = []

    for code in EXTRA_COUNTRIES:
        url = f"{BASE}/new/{code}.csv"
        path = RAW / "extra" / f"{code}.csv"

        results.append(_download(url, path, session))
        time.sleep(pause)

    return results


def download_fixtures() -> Fetch:
    """Upcoming fixtures across all main divisions, with pre-match odds."""
    session = requests.Session()

    return _download(
        f"{BASE}/fixtures.csv",
        RAW / "fixtures.csv",
        session,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    """Read one of their CSVs defensively.

    Older files are latin-1, many carry ragged trailing commas, and a
    few have blank rows padding the bottom of the sheet.
    """
    raw = path.read_bytes()

    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            io.StringIO(text),
            on_bad_lines="skip",
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    # Unnamed columns are the ragged-comma artefacts.
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]

    # Three column vocabularies exist across the archive. Fold the two
    # older/alternate ones onto the modern names here, before anything
    # downstream looks for them.
    #
    #   HT/AT          the Greek files of the early 2000s. HTHG/HTAG are
    #                  separate columns, so a bare HT is unambiguous.
    #   Home/Away/HG/  the whole /new/ line, which never adopted the
    #   AG/Res         mmz4281 naming.
    if "HomeTeam" not in df.columns:
        df = df.rename(
            columns={
                "HT": "HomeTeam", "AT": "AwayTeam",
                "Home": "HomeTeam", "Away": "AwayTeam",
                "HG": "FTHG", "AG": "FTAG", "Res": "FTR",
            }
        )

    if "Date" not in df.columns or "HomeTeam" not in df.columns:
        return pd.DataFrame()

    return df[df["Date"].notna()]


def _parse_dates(series: pd.Series) -> pd.Series:
    """Their date column mixes dd/mm/yy and dd/mm/yyyy, sometimes in one file."""
    parsed = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(series, format="%d/%m/%y", errors="coerce")

    return parsed.fillna(short)


def _blank_frame(index: pd.Index, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(np.nan, index=index, dtype="float64") for c in columns}
    )


def _normalise(df: pd.DataFrame, extras: dict) -> pd.DataFrame:
    """Project one raw frame onto the canonical schema."""
    out = pd.DataFrame(index=df.index)

    for key, value in extras.items():
        out[key] = value

    out["date"] = _parse_dates(df["Date"])
    out["home_team"] = df["HomeTeam"].astype("string").str.strip()
    out["away_team"] = df["AwayTeam"].astype("string").str.strip()

    for target, source in [
        ("fthg", "FTHG"), ("ftag", "FTAG"), ("ftr", "FTR"),
        ("hthg", "HTHG"), ("htag", "HTAG"),
    ]:
        out[target] = df[source] if source in df.columns else np.nan

    # Stat and odds coverage varies wildly by season; absent columns
    # become all-NA rather than being dropped, so every season shares
    # one schema and downstream code can test for nullness instead of
    # for column existence.
    keep = [c for c in STAT_COLUMNS + ODDS_COLUMNS if c in df.columns]
    missing = [c for c in STAT_COLUMNS + ODDS_COLUMNS if c not in df.columns]

    out = pd.concat(
        [out, df[keep], _blank_frame(df.index, missing)],
        axis=1,
    )

    out = out[out["date"].notna()]
    out = out[out["home_team"].notna() & out["away_team"].notna()]

    return out


def load_main() -> pd.DataFrame:
    """Parse every downloaded mmz4281 file into one frame."""
    frames = []

    for path in sorted((RAW / "main").glob("*/*.csv")):
        df = _read_csv(path)

        if df.empty:
            continue

        div = path.stem
        country, league, tier = DIVISIONS.get(div, ("Unknown", div, 1))

        # Recover the start year from the four-digit slug. 93 -> 1993,
        # 26 -> 2026: the site has no pre-1990 data, so a two-digit
        # value below 90 is a 2000s season.
        head = int(path.parent.name[:2])
        year = 1900 + head if head >= 90 else 2000 + head

        frames.append(
            _normalise(
                df,
                {
                    "source": "football-data.co.uk",
                    "country": country,
                    "league": league,
                    "tier": tier,
                    "div": div,
                    "season": season_label(year),
                },
            )
        )

    if not frames:
        return pd.DataFrame(columns=CORE_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def load_extra() -> pd.DataFrame:
    """Parse the /new/ line, whose columns are named differently."""
    frames = []

    for path in sorted((RAW / "extra").glob("*.csv")):
        df = _read_csv(path)

        if df.empty:
            continue

        country = EXTRA_COUNTRIES.get(path.stem, path.stem)

        normalised = _normalise(
            df,
            {
                "source": "football-data.co.uk/new",
                "country": country,
                "tier": 1,
                "div": path.stem,
            },
        )

        # Season here is a real column, not a per-file constant. It is
        # "2016/2017" for split-year leagues and "2016" for the
        # calendar-year ones (Brazil, USA, Japan, ...).
        normalised["league"] = df.loc[normalised.index, "League"].astype("string")
        normalised["season"] = df.loc[normalised.index, "Season"].astype("string")

        frames.append(normalised)

    if not frames:
        return pd.DataFrame(columns=CORE_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def build(out_path: Path = Path("data/processed/domestic_matches.parquet")) -> pd.DataFrame:
    """Normalise everything on disk into one parquet."""
    df = pd.concat([load_main(), load_extra()], ignore_index=True)

    # Concatenating frames that differ in which columns were present
    # can widen dtypes back to object, so pin the ones we index on.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]

    for column in ["home_team", "away_team", "div"]:
        df[column] = df[column].astype("string")

    for column in ["fthg", "ftag", "hthg", "htag"] + STAT_COLUMNS + ODDS_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["date", "div", "home_team"]).reset_index(drop=True)
    df["match_id"] = (
        df["div"] + "_"
        + df["date"].dt.strftime("%Y%m%d") + "_"
        + df["home_team"].str.replace(" ", "", regex=False) + "_"
        + df["away_team"].str.replace(" ", "", regex=False)
    )

    df = df[["match_id"] + CORE_COLUMNS + STAT_COLUMNS + ODDS_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    return df
