# PitchIQ

A match-prediction engine for the UEFA Champions League, built to forecast
the 2026/27 season.

It reads three decades of football results, learns how good every club
is, and plays the tournament ten thousand times.

```
Arsenal vs Tottenham  ->  Arsenal 75.7%, draw 15.7%, Tottenham 8.5%
Arsenal over a season ->  top 8 in 73.9%, wins it in 20.5%
```

| | RPS on 37,765 held-out matches |
|---|---|
| knowing nothing | 0.2282 |
| **PitchIQ** | **0.2085** |
| bookmaker closing line | 0.2036 |

Lower is better. **We cover about 80% of the distance from ignorance to
the professionals**, using only past results and dates — no injuries, no
lineups, no xG.

304,039 matches · 1,533 clubs · 83 countries · 179 tests

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
  matches.py            one date-ordered stream over every source
  config.py             credentials and paths
  ingest/               football-data.co.uk and openfootball
  features/             the causal pre-match feature layer
  models/               elo, league strength, dixon-coles, boosting,
                        uncertainty
  sim/                  tournament and league Monte Carlo
  eval/                 metrics, calibration, market, backtest
scripts/                download, train, predict, simulate, evaluate
data/external/          the 2026/27 draw
tests/                  179 tests
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

## Running it

```bash
python scripts/download_domestic.py       # ~300k domestic matches
python scripts/download_uefa.py           # ~4k UEFA matches
python scripts/check_coverage.py          # join quality report

python scripts/build_features.py          # 40s -> 55 features per match
python scripts/train.py                   # 7s  -> the fitted goals model
python scripts/bootstrap_uncertainty.py   # ~25 min, run once

python scripts/predict.py                 # all 144 fixture forecasts
python scripts/simulate.py                # 10,000 seasons
pytest
```

Training and bootstrapping happen once; prediction and simulation load
what they saved.

### The measurement suite

```bash
python scripts/evaluate_features.py       # do the features beat elo alone?
python scripts/evaluate_boosted.py        # four models on one split
python scripts/evaluate_market.py         # against the bookmaker line
python scripts/evaluate_calibration.py    # are the probabilities honest?
python scripts/evaluate_simulator.py      # are the season ranges honest?
python scripts/evaluate_uncertainty.py    # did sampling the ratings help?
```

## Status

End to end and measured.

- [x] Domestic ingest — 299,803 matches, 1993 → 2026
- [x] UEFA ingest — 4,240 matches, six competitions
- [x] Club-name resolution, no unresolved splits
- [x] 2026/27 draw, structurally validated
- [x] Elo ratings, with goal-difference weighting
- [x] League-strength calibration — per-country scale and offset
- [x] Dixon-Coles goals model with time decay — 2,862 parameters, 7s
- [x] Pre-match feature layer — 55 features, leakage-proof by design
- [x] Gradient boosting over those features
- [x] Monte Carlo simulation of the league phase and knockout bracket
- [x] Evaluation against bookmaker closing odds, with intervals
- [x] Probability calibration, on matches and on whole seasons
- [x] Parameter uncertainty propagated into the simulator

### What the measurement found

**We know nothing the bookmakers do not.** Blending model and market
put 0% weight on the model in all three folds, and log-odds stacking
gained 0.00009 RPS — indistinguishable from zero. The remaining gap is
team news, which is a data problem rather than a modelling one.

**Dixon-Coles is better calibrated than the market** — expected
calibration error 0.0033 against 0.0068. When it says 70%, it means 70%.

**Season-long ranges were too narrow, and are now close to honest.**
Across 4,225 real club-seasons, finishing positions regressed on
predicted positions with a slope of 0.828 where 1.000 is calibrated.
Sampling the ratings from a 200-refit bootstrap moved it to 0.884 and
brought 90% interval coverage to 90.7%. That closes 19% of the gap; the
rest is squad change between seasons, which resampling the past cannot
capture.

### Known limits

Four of the 36 clubs in the 2026/27 field — Shakhtar Donetsk, Slavia
Prague, Slovan Bratislava and Sabah — play in countries with no division
in the domestic archive. Their ratings come from European matches plus a
league prior, and the bootstrap reports them as the least certain in the
field: Sabah's attack rating carries six times Manchester City's spread.

**No squad information.** Injuries, suspensions, lineups and transfers
are absent. This is essentially the whole remaining gap to the market.
FBref is Cloudflare-blocked and its terms forbid automated access; no
free lineup source was found.

**No European odds.** The archive prices domestic leagues only, so the
market comparison is a domestic finding. How PitchIQ compares to the
market on Champions League football is untested and untestable with this
data.

## Documentation

[`docs/architecture.md`](docs/architecture.md) — start here. The whole
system explained from scratch with diagrams: pipeline, name resolution,
the feature layer's leakage guarantee, each model, the simulator, and
how everything is validated.

[`docs/how-it-works.md`](docs/how-it-works.md) — the detailed account
with every result, every interval, and the reasoning behind each
decision, including the claims that were overturned along the way.

## Licence

MIT. See `LICENSE`.

Licence covers the code. The data is not redistributed here and remains
subject to each provider's own terms.
