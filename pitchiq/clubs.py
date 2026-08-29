"""Resolve club names across sources.

Three vocabularies have to be reconciled:

    football-data.co.uk   "Ath Madrid", "Sp Lisbon", "Paris SG"
    openfootball (UCL)    "Club Atletico de Madrid", "Sporting Clube de
                          Portugal", "Paris Saint-Germain FC"
    UEFA draw report      "Atletico Madrid", "Sporting CP",
                          "Paris Saint-Germain"

Fuzzy string matching is the obvious approach and the wrong one: it
silently pairs "Sporting Clube de Braga" with "Sporting Clube de
Portugal", and there is no similarity threshold that separates them
while still catching "Sp Lisbon". So resolution here is exact.

:func:`normalise` strips the parts of a club name that carry no
identity -- accents, legal forms, founding years -- and everything else
goes through :data:`ALIASES`, a hand-checked table. A name that resolves
to nothing returns ``None`` rather than a best guess, and
:func:`coverage_report` exists to keep that set visible.
"""

from __future__ import annotations

import re
import unicodedata

# Characters that survive NFKD because they are letters in their own
# right rather than a base plus a combining mark.
CHAR_MAP = str.maketrans(
    {
        "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "å": "a", "Å": "a",
        "ß": "ss", "đ": "d", "Đ": "d", "ł": "l", "Ł": "l",
        "ı": "i", "ð": "d", "þ": "th", "œ": "oe",
    }
)

# Tokens that identify a legal form or a founding year, not a club.
# "sp" is deliberately absent: football-data.co.uk uses it as the
# abbreviation in both "Sp Lisbon" and "Sp Braga", where it is the only
# thing distinguishing them from other Lisbon and Braga sides.
NOISE_TOKENS = {
    "fc", "cf", "afc", "ac", "as", "sc", "sk", "kv", "bc", "fk", "gnk",
    "ss", "ssc", "osc", "cd", "ud", "rc", "sv", "vfb", "bsc", "pae",
    "sfp", "club", "clube", "de", "del", "di", "da", "do", "the",
    "calcio", "futebol", "football", "futbol", "fussball", "spor",
}

YEAR_TOKEN = re.compile(r"^(1[89]\d\d|\d{2})$")


def normalise(name: str) -> str:
    """Reduce a club name to its identifying core.

    'FC Bayern München' and 'Bayern Munich' do not converge here -- no
    rule turns München into Munich -- which is exactly what
    :data:`ALIASES` is for. What this does handle is the mechanical
    noise: accents, legal forms, founding years, punctuation.
    """
    text = name.translate(CHAR_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()

    text = re.sub(r"[.'`’\-/&]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)

    tokens = [
        t for t in text.split()
        if t not in NOISE_TOKENS and not YEAR_TOKEN.match(t)
    ]

    return " ".join(tokens)


# Canonical key -> the spellings that do NOT reduce to it under
# normalise(). The key is always the football-data.co.uk spelling, so a
# domestic row resolves through this table too rather than by accident.
#
# Everything absent from this table resolves on its normalised form
# alone -- "Galatasaray SK" and "Galatasaray" already agree once the
# legal form is dropped, and listing such pairs here would be noise.
ALIASES: dict[str, list[str]] = {
    # England
    "Man City": ["Manchester City", "Manchester City FC"],
    "Man United": ["Manchester United", "Manchester United FC"],
    "Newcastle": ["Newcastle United", "Newcastle United FC"],
    "Tottenham": ["Tottenham Hotspur", "Tottenham Hotspur FC"],
    "Leicester": ["Leicester City", "Leicester City FC"],
    # Spain
    "Ath Madrid": [
        "Atletico Madrid", "Atlético Madrid", "Club Atlético de Madrid",
    ],
    "Ath Bilbao": ["Athletic Club", "Athletic Bilbao"],
    "Sociedad": ["Real Sociedad", "Real Sociedad de Fútbol"],
    "Betis": ["Real Betis", "Real Betis Balompié"],
    "Espanol": ["Espanyol", "RCD Espanyol de Barcelona"],
    "Vallecano": ["Rayo Vallecano", "Rayo Vallecano de Madrid"],
    "Celta": ["Celta Vigo", "RC Celta de Vigo"],
    "La Coruna": ["Deportivo La Coruña", "RC Deportivo de La Coruña"],
    # Germany
    "Bayern Munich": ["Bayern München", "FC Bayern München"],
    "Dortmund": ["Borussia Dortmund"],
    "Leverkusen": ["Bayer Leverkusen", "Bayer 04 Leverkusen"],
    "M'gladbach": ["Borussia Mönchengladbach", "Bor. Mönchengladbach"],
    "Ein Frankfurt": ["Eintracht Frankfurt"],
    "Wolfsburg": ["VfL Wolfsburg"],
    "Union Berlin": ["1. FC Union Berlin"],
    "Schalke 04": ["FC Schalke 04"],
    "Hoffenheim": ["1899 Hoffenheim", "TSG 1899 Hoffenheim"],
    "FC Koln": ["1. FC Köln", "FC Köln"],
    # Italy
    "Inter": ["Inter Milan", "Internazionale", "FC Internazionale Milano"],
    "Lazio": ["Lazio Roma"],
    # France. Monaco plays in Ligue 1, so MCO resolves against France.
    "Paris SG": ["Paris Saint-Germain", "Paris Saint-Germain FC"],
    "Lyon": ["Olympique Lyonnais"],
    "Marseille": ["Olympique Marseille", "Olympique de Marseille"],
    "Lens": ["Racing Club de Lens", "RC Lens"],
    "Rennes": ["Stade Rennais"],
    "Brest": ["Stade Brestois 29", "Stade Brestois"],
    "Montpellier": ["Montpellier HSC"],
    # Portugal. "Sp Lisbon" and "Sp Braga" are the reason fuzzy matching
    # cannot be used anywhere in this module.
    "Sp Lisbon": [
        "Sporting CP", "Sporting Lisbon", "Sporting Clube de Portugal",
    ],
    "Sp Braga": ["Braga", "Sporting Braga", "Sporting Clube de Braga"],
    "Benfica": ["SL Benfica", "Sport Lisboa e Benfica"],
    # Netherlands
    "PSV Eindhoven": ["PSV"],
    "Feyenoord": ["Feyenoord Rotterdam"],
    # Belgium
    "Gent": ["KAA Gent"],
    "Genk": ["KRC Genk"],
    "Anderlecht": ["RSC Anderlecht"],
    "Antwerp": ["Royal Antwerp", "Royal Antwerp FC"],
    "St. Gilloise": ["Union Saint-Gilloise", "Royale Union Saint-Gilloise"],
    # Greece
    "AEK": ["AEK Athen", "AEK Athens", "AEK Athens FC"],
    "Olympiakos": ["Olympiacos", "Olympiakos Piraeus"],
    # Turkey. football-data.co.uk still files Basaksehir under the old
    # Istanbul Buyuksehir Belediyespor name.
    "Buyuksehyr": ["İstanbul Başakşehir", "Istanbul Basaksehir", "Basaksehir"],
    # Austria
    "Salzburg": ["RB Salzburg", "Red Bull Salzburg", "FC Red Bull Salzburg"],
    "Austria Vienna": ["Austria Wien"],
    "LASK": ["LASK Linz"],
    # Denmark, Scandinavia
    "FC Copenhagen": ["Copenhagen", "FC København"],
    # Poland
    "Legia": ["Legia Warszawa", "Legia Warsaw"],
    # Romania. Steaua was renamed FCSB after the trademark ruling.
    "FCSB": ["Steaua", "Steaua Bucharest", "FC Steaua Bucureşti"],
    "Otelul": ["Oţelul Galaţi"],
    # Russia
    "CSKA Moscow": ["CSKA Moskva"],
    "Lokomotiv Moscow": ["Lokomotiv Moskva"],
    "Spartak Moscow": ["Spartak Moskva"],
    "Zenit": ["Zenit St. Petersburg", "Zenit St Petersburg"],
    # Countries with no division in the archive. These clubs have no
    # domestic history to join to, but they still recur across UCL
    # seasons under spellings that differ from the UEFA draw report, so
    # they need one stable key each.
    "Slavia Prague": ["Slavia Praha", "SK Slavia Praha"],
    "Sparta Prague": ["Sparta Praha", "AC Sparta Praha"],
    "Red Star Belgrade": ["Crvena Zvezda", "FK Crvena Zvezda"],
    "Dinamo Zagreb": ["GNK Dinamo Zagreb"],
    "Dynamo Kyiv": ["Dinamo Kiev", "Dynamo Kiev"],
    "Shakhtar Donetsk": ["FK Shakhtar Donetsk"],
    "Slovan Bratislava": ["ŠK Slovan Bratislava"],
    "Qarabağ": ["Qarabağ Ağdam FK", "Qarabag"],
    "Sabah": ["Sabah FK"],
    "Ludogorets": ["PFC Ludogorets Razgrad", "Ludogorets Razgrad"],
    "APOEL": ["APOEL Nikosia", "APOEL Nicosia"],
    "Viktoria Plzen": ["Viktoria Plzeň", "Viktoria Plzen"],
    "Ferencvaros": ["Ferencvárosi TC", "Ferencvaros"],
    "Astana": ["FK Astana"],
    "Kairat": ["FK Kairat"],
    "Pafos": ["Paphos FC", "Pafos FC"],
    "Sheriff": ["FC Sheriff", "Sheriff Tiraspol"],
    "BATE": ["BATE Borisov"],
    "Maribor": ["NK Maribor"],
    # Clubs reached through the Europa and Conference Leagues. Each
    # pairs a football-data.co.uk spelling with the fuller name
    # openfootball uses.
    "Brighton": ["Brighton & Hove Albion"],
    "West Ham": ["West Ham United"],
    "Hearts": ["Heart of Midlothian"],
    "Fiorentina": ["ACF Fiorentina"],
    "Nice": ["OGC Nice"],
    "Heidenheim": ["1. FC Heidenheim 1846"],
    "Mainz": ["1. FSV Mainz 05"],
    "SK Rapid": ["Rapid Wien", "Rapid Vienna"],
    "Charleroi": ["Sporting Charleroi"],
    "Standard": ["Standard Liège", "Standard Liege"],
    "Guimaraes": ["Vitória Guimarães", "Vitoria Guimaraes"],
    "Lausanne": ["FC Lausanne-Sport"],
    "St. Patricks": ["St Patrick's Athletic"],
    "Aris": ["Aris Saloniki", "Aris Thessaloniki"],
    "PAOK": ["PAOK Saloniki", "PAOK Thessaloniki"],
    # Nordic and Baltic sides, where the club suffix carries no identity
    # but is not safe to strip globally.
    "Brondby": ["Brøndby IF", "Brondby IF"],
    "Silkeborg": ["Silkeborg IF"],
    "Rosenborg": ["Rosenborg BK"],
    "Tromso": ["Tromsø IL"],
    "AIK": ["AIK Solna"],
    "Hacken": ["BK Häcken"],
    "Djurgarden": ["Djurgårdens IF"],
    "Hammarby": ["Hammarby IF"],
    "Elfsborg": ["IF Elfsborg"],
    "HJK": ["HJK Helsinki"],
    "Ilves": ["Ilves Tampere"],
    "KuPS": ["Kuopion PS"],
    "SJK": ["SJK Seinäjoki"],
    "VPS": ["Vaasan PS"],
    # Poland and Romania. Wisla Krakow and Wisla Plock are distinct, as
    # are the two Craiova clubs and the two Cluj clubs.
    "Jagiellonia": ["Jagiellonia Białystok"],
    "Rakow": ["Raków Częstochowa"],
    "Wisla": ["Wisła Kraków", "Wisla Krakow"],
    "U Craiova": [
        "Univ. Craiova", "Universitatea Craiova", "CS Universitatea Craiova",
    ],
    "U. Cluj": ["Universitatea Cluj"],
    "Corvinul": ["FC Corvinul Hunedoara"],
}

# openfootball country code -> football-data.co.uk country name. A code
# absent here has no domestic division in the archive, so its clubs can
# only be rated from continental matches plus a league-strength prior.
COUNTRY_NAMES = {
    "ENG": "England", "ESP": "Spain", "GER": "Germany", "ITA": "Italy",
    "FRA": "France", "POR": "Portugal", "NED": "Netherlands",
    "BEL": "Belgium", "TUR": "Turkey", "GRE": "Greece", "SCO": "Scotland",
    "AUT": "Austria", "NOR": "Norway", "DEN": "Denmark",
    "SUI": "Switzerland", "SWE": "Sweden", "RUS": "Russia",
    "POL": "Poland", "ROU": "Romania", "IRL": "Ireland", "FIN": "Finland",
    # Monaco has no league of its own; the club plays in Ligue 1.
    "MCO": "France",
}

COUNTRY_CODES = {name: code for code, name in COUNTRY_NAMES.items()}
COUNTRY_CODES["France"] = "FRA"

# Clubs that normalise onto an existing club but are a different side.
# Dropping the founding year is right for "Bologna FC 1909", which has
# no namesake, and wrong here: "U Craiova 1948" and "U Craiova" are two
# clubs, as are "Granada 74" and "Granada". Matched on the exact
# spelling, before normalisation, and scoped by country.
EXACT: dict[tuple[str, str], str] = {
    # Wimbledon FC moved away and became MK Dons; AFC Wimbledon is the
    # separate, supporter-founded club.
    ("ENG", "Wimbledon"): "ENG:wimbledon fc",
    ("ROU", "U Craiova 1948"): "ROU:u craiova 1948",
    ("ESP", "Granada 74"): "ESP:granada 74",
    ("ESP", "Extremadura UD"): "ESP:extremadura ud",
}


def _build_index() -> dict[str, str]:
    index: dict[str, str] = {}
    clashes: dict[str, set[str]] = {}

    for key, spellings in ALIASES.items():
        for spelling in [key, *spellings]:
            form = normalise(spelling)

            if not form:
                continue

            if form in index and index[form] != key:
                clashes.setdefault(form, {index[form]}).add(key)

            index[form] = key

    if clashes:
        detail = "; ".join(f"{f!r} -> {sorted(k)}" for f, k in clashes.items())
        raise ValueError(f"alias table maps one form to several clubs: {detail}")

    return index


INDEX = _build_index()


def country_code(country: str | None) -> str | None:
    """Accept either 'GER' or 'Germany' and return the three-letter code.

    Codes for countries with no domestic division -- UKR, CZE, AZE and
    the rest -- pass through unchanged. Their clubs still need a stable
    key so that a side appearing in several UCL seasons resolves to one
    club across all of them.
    """
    if not country:
        return None

    text = str(country).strip()

    if len(text) == 3 and text.isupper():
        # Route through the country name so that codes sharing a league
        # collapse together -- MCO clubs play in France, so MCO -> FRA.
        name = COUNTRY_NAMES.get(text)

        return COUNTRY_CODES.get(name, text) if name else text

    return COUNTRY_CODES.get(text)


def resolve(name: str, country: str | None = None) -> str | None:
    """Canonical club key, in two tiers.

    A spelling in :data:`ALIASES` resolves to that club's canonical key.
    Anything else resolves to its own normalised form, scoped by country
    so that two unrelated clubs sharing a name in different countries
    stay apart.

    Returns ``None`` only for an empty name. Nothing here guesses: two
    spellings join if and only if they agree after normalisation or the
    alias table says they are the same club.
    """
    if not name or not str(name).strip():
        return None

    text = str(name).strip()
    code = country_code(country)

    if (code, text) in EXACT:
        return EXACT[(code, text)]

    form = normalise(text)

    if not form:
        return None

    if form in INDEX:
        return INDEX[form]

    return f"{code}:{form}" if code else form


def is_curated(key: str) -> bool:
    """Whether a key came from the hand-checked alias table."""
    return key in ALIASES


def coverage_report(
    names: list[str], countries: list[str | None] | None = None
) -> tuple[dict[str, str], list[str]]:
    """Split names into resolved and unresolved."""
    countries = countries or [None] * len(names)
    resolved, unresolved = {}, []

    for name, country in zip(names, countries):
        key = resolve(name, country)

        if key is None:
            unresolved.append(name)
        else:
            resolved[name] = key

    return resolved, sorted(set(unresolved))
