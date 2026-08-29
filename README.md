# PitchIQ

A match-prediction engine for the UEFA Champions League, built to forecast
the 2026/27 season.

## Why it is built this way

The Champions League is a small tournament. Fifteen seasons of it come to
roughly 4,000 matches, spread across 350 clubs that rarely meet twice.
That is nowhere near enough to learn how good a club is.

So club strength is learned from **domestic leagues**, where the same
sides play each other repeatedly — about 300,000 matches across 27
countries — and those ratings are then carried onto Champions League
fixtures.

That transfer has one hard problem. A club dominating the Azerbaijani
league and a club dominating La Liga look identical in isolation: both
win, both score around two goals a game. Nothing in domestic results
says which league is harder. The only evidence that ties leagues to a
common scale is clubs from different leagues playing each other, which
is why this project ingests the Europa and Conference Leagues and their
qualifying rounds as well. The qualifiers matter most — they are where
the smaller federations actually appear.

## Data sources

| Source | Coverage | Licence |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | 27 countries, 38 divisions, 1993/94 → 2026/27, with bookmaker closing odds | Free for personal use; not redistributed here |
| [openfootball](https://github.com/openfootball/champions-league) | UCL / Europa / Conference plus qualifiers, 2011/12 → 2025/26 | Public domain |
| [football-data.org](https://www.football-data.org/) | Champions League, recent seasons | Free tier |

Match data is **not committed to this repository**. The download scripts
fetch it, and everything under `data/raw/` and `data/processed/` is
generated. The one exception is `data/external/`, which holds the
2026/27 league-phase draw — hand-assembled and not reproducible by
script.

The draw was compiled from UEFA's published draw report and the
corresponding Wikipedia article (CC BY-SA), then validated
structurally: 36 clubs, four home and four away each, one opponent per
seeding pot in each direction, no repeated pairings, no two clubs from
the same association.

## Layout

```
pitchiq/
  clubs.py              club-name resolution across sources
  config.py             credentials and paths
  ingest/
    football_data_uk.py domestic results and odds
    openfootball.py     UEFA club competitions
  features/  models/  sim/  eval/
scripts/
  download_domestic.py  fetch and normalise domestic data
  download_uefa.py      fetch and normalise UEFA data
  check_coverage.py     how much of the UEFA field joins to domestic data
data/external/          the 2026/27 draw
tests/
```

## Club-name resolution

Three sources spell the same club three ways: `Ath Madrid`,
`Club Atlético de Madrid`, `Atlético Madrid`. Joining them is the part
that quietly breaks everything downstream, so `pitchiq/clubs.py` does it
exactly, never by similarity.

Fuzzy matching is not used anywhere. There is no similarity threshold
that pairs `Sporting Clube de Portugal` with `Sp Lisbon` while keeping
it away from `Sporting Clube de Braga`. Instead a name is reduced to its
identifying core — accents, legal forms and founding years removed — and
anything that still does not converge goes through a hand-checked alias
table. A name that resolves to nothing returns `None` rather than a best
guess.

Some cases only a human catches:

- Başakşehir is filed upstream under its former name, `Buyuksehyr`.
- Steaua Bucureşti became `FCSB` after the trademark ruling.
- Monaco has no domestic league of its own and plays in Ligue 1.
- `U Craiova` and `U Craiova 1948` are different clubs, as are `Granada`
  and `Granada 74`, and `Wimbledon` and `AFC Wimbledon`. Stripping the
  founding year merges each pair, so an exact-match tier runs first.

A test scans every domestic club name for two distinct clubs sharing a
resolved key, because that failure is silent and corrupts the ratings of
both.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .

cp .env.example .env      # then fill in the values
```

`.env` holds credentials and is gitignored. Nothing reads it except
`pitchiq/config.py`, and no value is ever logged.

## Building the datasets

```bash
python scripts/download_domestic.py   # ~300k domestic matches, a few minutes
python scripts/download_uefa.py       # ~4k UEFA matches
python scripts/check_coverage.py      # join quality report
pytest
```

## Status

Data layer is complete and tested. The rating and prediction layers are
not built yet.

- [x] Domestic ingest — 299,803 matches, 27 countries, 1993 → 2026
- [x] UEFA ingest — 4,240 matches across three competitions and qualifiers
- [x] Club-name resolution, no unresolved splits
- [x] 2026/27 draw, structurally validated
- [ ] Elo ratings
- [ ] Dixon-Coles goals model with time decay
- [ ] League-strength calibration
- [ ] Evaluation against bookmaker closing odds (RPS, log-loss)
- [ ] Monte Carlo simulation of the 36-team league phase and knockout bracket

Four of the 36 clubs in the 2026/27 field — Shakhtar Donetsk, Slavia
Prague, Slovan Bratislava and Sabah — play in countries with no division
in the domestic archive. Their ratings can only come from European
matches plus a league prior. Sabah has ten such matches, so its
predictions will carry wide uncertainty, and are reported as such rather
than presented with false confidence.

## Documentation

[`docs/how-it-works.md`](docs/how-it-works.md) — the full account: data
sources, name resolution, each model and why it exists, the evaluation
method and results, the simulator, validation against 2025/26, and the
known limitations.

## Licence

MIT. See `LICENSE`.

Licence covers the code. The data is not redistributed here and remains
subject to each provider's own terms.
