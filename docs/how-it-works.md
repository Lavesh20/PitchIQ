# PitchIQ — how it works

A record of what this project is, how each piece was built, what was
measured, and what is still wrong with it.

---

## 1. The problem

Forecast the UEFA Champions League 2026/27: for each of the 36 clubs,
the chance of finishing in each league-phase position, reaching each
knockout round, and winning the competition.

The obvious approach — train on Champions League history — does not
work. Fifteen seasons of it come to about 4,000 matches spread across
350 clubs that rarely meet twice. There is nothing there to learn club
strength from.

So strength is learned from **domestic leagues**, where the same sides
play each other repeatedly, and then carried onto European fixtures.

That transfer has one hard problem underneath it. A club dominating the
Azerbaijani league and a club dominating La Liga look identical in
isolation: both win, both score around two goals a game. Nothing in
domestic results says which league is harder. The only evidence tying
leagues to a common scale is **clubs from different leagues playing each
other**, and there is very little of it.

Everything below is shaped by that constraint.

---

## 2. Data

### Sources

| Source | Coverage | Access |
|---|---|---|
| [football-data.co.uk](https://www.football-data.co.uk/) | 27 countries, 38 divisions, 1993/94 → 2026/27, with bookmaker closing odds | CSV, no key |
| [openfootball](https://github.com/openfootball/champions-league) | Champions, Europa and Conference Leagues plus qualifiers, 2011/12 → 2025/26 | Plain text, public domain |
| UEFA draw report + Wikipedia | The 2026/27 league-phase draw | Hand-compiled |

Rejected after testing: **football-data.org** (free tier stops at three
seasons of history), **FBref** (Cloudflare 403, and its terms forbid
automated access), **worldfootballR** (repo archived September 2025),
**ClubElo** (only the `/Fixtures` endpoint responded; club history and
date snapshots timed out with zero bytes across every timeout tried).

### The qualifying-round discovery

The openfootball repository is named after the Champions League but
holds all three competitions and their qualifying rounds in the same
season directories: `cl.txt`, `el.txt`, `conf.txt`, `clq.txt`,
`elq.txt`, `confq.txt`.

Adding them was the single highest-value data decision in the project:

| | Before | After |
|---|---|---|
| UEFA matches | 1,997 | **4,240** |
| Cross-league matches | ~1,968 | **4,206** |
| Countries linked | 21 | **56** |
| Distinct clubs | 136 | **346** |

Qualifying rounds matter more than their profile suggests. They are
where Latvian, Faroese, Bosnian and Kazakh clubs actually appear — the
league-strength evidence that does not exist anywhere else.

It also gave the four clubs with no domestic league a record:

| Club | European matches |
|---|---|
| Shakhtar Donetsk | 104 |
| Slavia Prague | 64 |
| Slovan Bratislava | 44 |
| Sabah | 10 |

### The final match stream

`pitchiq/matches.py` merges everything into one date-ordered table:

```
304,039 matches   1,533 clubs   83 countries   1993-07-23 → 2026-08-28
   299,799 domestic
     4,240 UEFA
```

Downloaded and derived data is **not** committed. Both parquets
regenerate from the download scripts, and neither provider's bulk data
is ours to redistribute. The one exception is `data/external/`, holding
the 2026/27 draw, which no script reproduces.

---

## 3. Club-name resolution

Three vocabularies describe the same clubs:

```
football-data.co.uk   Ath Madrid       Sp Lisbon       Paris SG
openfootball          Club Atletico    Sporting Clube  Paris Saint-
                      de Madrid        de Portugal     Germain FC
UEFA draw report      Atletico Madrid  Sporting CP     Paris Saint-Germain
```

Joining them wrongly is silent and corrupts every rating on both sides
of the join. This was where the project had originally stalled.

**Fuzzy matching is not used anywhere.** There is no similarity
threshold that pairs `Sporting Clube de Portugal` with `Sp Lisbon` while
keeping it away from `Sporting Clube de Braga`. Instead, resolution runs
in three tiers:

1. **Exact table**, consulted first, for names that must not merge.
2. **Alias table**, hand-checked, for spellings that genuinely diverge.
3. **Normalised form**, country-scoped, for everything else — accents,
   legal forms and founding years stripped.

An unresolved name returns `None`, never a best guess.

### Cases only a human catches

- Başakşehir is filed upstream under its former name, `Buyuksehyr`.
- Steaua Bucureşti became `FCSB` after the trademark ruling.
- Monaco has no domestic league and plays in Ligue 1, so `MCO` → `FRA`.
- football-data.co.uk renamed `U Craiova` to `Univ. Craiova` mid-season
  in February 2021; both are one club.

### The founding-year bug

Stripping founding years is right for `Bologna FC 1909`, which has no
namesake. It is wrong here:

```
U Craiova    vs  U Craiova 1948     different clubs
Granada      vs  Granada 74         different clubs
Wimbledon    vs  AFC Wimbledon      different clubs
```

Each pair would have merged into one blended phantom club. That is why
the exact-match tier runs before normalisation, and why a test scans all
300,000 domestic club names for any two distinct clubs sharing a
resolved key.

**Result:** zero unresolved splits. All 36 clubs of the 2026/27 field
resolve. 32 of them have domestic history; the other four are the
uncovered countries above.

---

## 4. The models

Four were built. Three are stepping stones; the fourth is the product.

### 4.1 Elo — one number per club

Each club holds a rating. Before a match, an expected score:

```
E_home = 1 / (1 + 10 ^ (-(R_home + H - R_away) / 400))
```

with `H` ≈ 65 rating points of home advantage. Afterwards, points move
from loser to winner in proportion to how surprising the result was,
scaled by margin of victory (a two-goal win counts 1.5×, tapering after
that). The update is zero-sum: nothing is created.

A club carries the same rating into every competition, so when
Bodø/Glimt beat Inter, points move directly from Serie A into the
Eliteserien. League strength emerges from continental results rather
than needing a parameter.

Ratings are recorded **before** each match is applied, so the history is
usable for backtesting without leaking the result being predicted.

304,039 matches in 0.8 seconds.

### 4.2 The pathology Elo could not fix

Its European rankings came out wrong in a specific way:

| Rank | Club | |
|---|---|---|
| 12 | Celtic | dominates a small closed league |
| 16 | Bournemouth | never plays in Europe |
| 17 | Zenit | banned from Europe since 2022, rating frozen |
| 24 | Atlético Madrid | |
| 25 | Brentford | never plays in Europe |

Elo is zero-sum, so a league cannot inflate as a bloc without winning in
Europe. What it *can* do is let a dominant club hoard its own league's
points. Celtic take points off the same Scottish sides across 38 matches
a season and play perhaps eight European ties to correct it. Their
rating measures *how far above Scottish football they are*, which is not
the same thing as how good they are.

### 4.3 League strength — the fix that failed first

The first attempt gave each country an **offset**. It changed almost
nothing: Celtic moved from 12th to 12th.

The failure was informative. Shifting every Scottish club down 100
points preserves the gaps between them, so Celtic still sits 400 points
above Rangers. The problem is not *where Scotland sits* — it is *how
spread out Scotland is*.

The working model has two corrections per country:

```
adjusted = league_mean + scale × (rating − league_mean) + offset
```

`scale` below 1 pulls a league's clubs toward their own average,
squeezing a lead built on domestic dominance rather than European
evidence. Both parameters are fitted on cross-country matches only, and
both are shrunk toward "no correction" so a country with 44 European
matches cannot swing far on noise.

What it learned, unsupervised:

| Country | Offset | Scale | European matches |
|---|---|---|---|
| Italy | +53 | 0.94 | 415 |
| Spain | +35 | 1.03 | 549 |
| Germany | +31 | 1.07 | 477 |
| England | +21 | 0.99 | 538 |
| Greece | −115 | 0.83 | 98 |
| Romania | −117 | 0.82 | 44 |
| **Scotland** | **−126** | **0.76** | 83 |

Scotland is marked down on both counts. Celtic falls from 12th to 84th;
Atlético climbs from 24th to 14th.

### 4.4 Outcome mapper

Elo yields an expected score, not three probabilities, and the split
between draw and win is not recoverable from it analytically. Rather
than assume a shape, a multinomial logistic regression learns the
mapping from rating difference to home/draw/away, fitted on domestic
matches so European ones stay clean for evaluation.

### 4.5 Dixon-Coles — the product

Each club gets an **attack** and a **defence** rating. For club *i* at
home against *j*:

```
log λ = attack[i] + defence[j] + γ        (home goals)
log μ = attack[j] + defence[i]            (away goals)
```

Two departures from plain Poisson, both from Dixon and Coles (1997):

**Low-score dependence.** Real matches produce more 0-0 and 1-1, and
fewer 1-0 and 0-1, than independence predicts. A single parameter ρ
corrects those four scorelines:

```
τ(0,0) = 1 − λμρ      τ(0,1) = 1 + λρ
τ(1,0) = 1 + μρ       τ(1,1) = 1 − ρ
```

**Time decay.** Every match is weighted `exp(−ξ · days_ago)`.

Everything is fitted jointly by maximum likelihood over all matches at
once, domestic and European together.

#### The fit

```
training set : 304,039 real matches
parameters   : 2,862   (1,430 clubs × 2, plus γ and ρ)
algorithm    : L-BFGS-B
iterations   : 258
loss         : 64,113 → 59,779
time         : 7 seconds
```

Fitted values landed where the literature says they should: **γ = 0.228**
(hosts score ~26% more) and **ρ = −0.039**, negative as Dixon and Coles
found.

The speed comes from writing the gradients analytically. A numerical
gradient over 2,862 parameters would need 2,862 likelihood evaluations
per step × 258 steps ≈ 738,000 passes over the data — hours instead of
seconds. Because a subtly wrong gradient still converges, silently, to
the wrong answer, the gradients are verified against numerical
differences in the test suite.

#### Dixon-Coles solves the league problem on its own

This was a genuine surprise. It was expected to inherit Elo's blind
spot. It does not:

| Club | Elo rank | Dixon-Coles rank |
|---|---|---|
| Celtic | 12th | **44th** |
| Bournemouth | 16th | **72nd** |
| Bodø/Glimt | 18th | 75th |

No explicit correction. Fitting all matches jointly propagates
cross-league evidence through shared opponents in a way sequential Elo
cannot.

---

## 5. Evaluation

### Method

Everything is trained before a cutoff and scored after it. Any
hyperparameter — the ensemble weight in particular — is chosen on a
**validation window that ends before the test window begins**.

The measure is **RPS** (ranked probability score): how far the forecast
probabilities sat from what happened, respecting that home/draw/away are
ordered. Lower is better; 0 is perfect. Accuracy is not used as a
headline, because always predicting "home win" scores ~45% and has
learned nothing.

### Domestic matches, against the market

38,128 held-out matches with bookmaker closing odds:

| | RPS | Log-loss | Accuracy |
|---|---|---|---|
| Base rate | 0.2281 | 1.0750 | 43.7% |
| Elo | 0.2101 | 1.0223 | 48.6% |
| **Bookmaker closing line** | **0.2037** | 1.0018 | 50.4% |

Elo closes **74%** of the gap between knowing nothing and the market —
using one number per club, with no injuries, lineups, transfers or xG.
The closing line is the sharpest public forecast that exists.

### European matches

1,981 held out. **Which model wins depends entirely on how often it may
refit**, and getting this wrong produced a misleading result at first:

| | Frozen at cutoff | Refit as you go |
|---|---|---|
| Elo + league strength | 0.2191 | **0.2052** |
| Dixon-Coles | **0.2092** | 0.2075 |
| Ensemble (validated weight) | 0.2094 | **0.2040** |

Elo collapses when frozen — one number per club goes stale fast.
Dixon-Coles barely moves.

**Forecasting a whole league phase before a ball is kicked is the frozen
case**, which is the actual job here, and Dixon-Coles wins it clearly.
Updating predictions as a season unfolds is the rolling case, where Elo
is slightly ahead and blending helps.

An early version compared rolling Elo against frozen Dixon-Coles and
concluded Elo was better. That was measuring the refit schedule, not the
models.

---

## 6. Simulation

Per-match probabilities cannot answer "who wins the trophy". Each
fixture has ~120 possible scorelines and there are 144 fixtures, so the
season space cannot be enumerated. Instead the season is played out at
random 10,000 times and the outcomes counted.

### One simulated season

1. Sample a scoreline for each of the 144 league-phase fixtures
2. Build the table — 3 points a win, 1 a draw
3. Sort by UEFA's tiebreakers
4. Split: **1–8** to the round of 16, **9–24** to a two-legged play-off,
   **25–36** eliminated
5. Play the play-off, round of 16, quarter-finals, semi-finals, final

Two-legged ties are aggregated over both legs. Level after 180 minutes
goes to extra time (goal rates scaled by 30/90), then to penalties,
modelled as a coin flip. The away-goals rule was abolished in 2021 and
is not applied. The final is played at a neutral venue, modelled by
averaging both orientations of the scoreline grid.

The bracket is fixed rather than redrawn, so a league position matters
twice: for skipping the play-off, and for which half of the draw a club
lands in. Pairings *within* a seeding band are drawn, and the simulation
randomises them.

Every ordered club pairing's scoreline distribution is precomputed once,
so 10,000 seasons take **0.6 seconds**.

### Internal consistency

Every conservation law holds exactly, not approximately:

```
top 8 places sum to        8.000
last 16                   16.000
quarter-finalists          8.000
finalists                  2.000
champions                  1.000
mean finishing position   18.50   (must be 18.5)
```

No club reaches a round without the previous one. Nobody in the top 8
plays a play-off. The simulated draw rate is 21.7%; the measured real
European rate is 20.4%.

---

## 7. Validation against a known season

The strongest test available: retrain on data ending **July 2025**, then
simulate 2025/26 — a season the model has never seen — and compare
against what actually happened.

| | |
|---|---|
| Position correlation | **0.79** |
| Mean absolute position error | 5.1 places |
| **Actual positions inside the 90% interval** | **33/36 = 92%** |

The last line is the one that matters. A well-calibrated model puts
about 90% of outcomes inside its 90% band. **The uncertainty estimates
are honest.**

PSG finished **11th** in the league phase and won the trophy. The model
gave them 4.8% — a genuine upset, appropriately humble rather than
pretending to have seen it coming. The model's own favourite, Manchester
City at 22%, finished 8th.

---

## 8. Results — UCL 2026/27

| Club | Avg pts | Top 8 | R16 | QF | SF | Final | Wins it |
|---|---|---|---|---|---|---|---|
| Arsenal | 16.8 | 74% | 95% | 74% | 52% | 35% | **20.4%** |
| Manchester City | 16.0 | 64% | 93% | 71% | 48% | 31% | **18.4%** |
| Bayern Munich | 16.7 | 72% | 94% | 69% | 45% | 28% | **14.8%** |
| Liverpool | 16.3 | 67% | 91% | 62% | 38% | 21% | **10.7%** |
| Paris Saint-Germain | 15.2 | 55% | 86% | 56% | 32% | 16% | 7.9% |
| Barcelona | 15.2 | 55% | 86% | 54% | 29% | 15% | 6.7% |
| Real Madrid | 15.1 | 53% | 85% | 54% | 30% | 14% | 6.6% |
| Inter Milan | 14.6 | 47% | 81% | 47% | 23% | 11% | 4.5% |
| … | | | | | | | |
| Sabah | 3.3 | 0.0% | 0.1% | — | — | — | 0.0% |

Full table: `data/predictions/ucl_2026_27_simulation.csv`.

---

## 9. Known limitations

**The favourites are probably too strong.** Arsenal at 20.4% is higher
than a bookmaker would quote (typically 15–18%). The model treats its
own ratings as certain: it knows Arsenal's attack is +0.94, but not that
it is ±0.15. Propagating that uncertainty would flatten the top and push
probability toward the field. This is the single most worthwhile
improvement outstanding.

**Four clubs have no domestic data.** Shakhtar Donetsk, Slavia Prague,
Slovan Bratislava and Sabah play in countries with no division in the
archive. Their ratings come from European matches alone. Sabah has ten,
so its numbers are the least trustworthy in the table and should be read
as such.

**Four tiebreakers are missing.** Implemented: points, goal difference,
goals scored, away goals, wins, away wins. UEFA has four more after that
(opponents' collective points, goal difference and goals, then
disciplinary points). They decide roughly one simulated season in a
thousand.

**No squad information.** No injuries, suspensions, transfers, manager
changes or fixture congestion. This is most of the remaining gap to the
bookmaker line.

**Penalty shootouts are a coin flip.** There is weak evidence of
persistent team skill in shootouts; it is not modelled.

**Russian clubs are stale.** Barred from Europe since 2022, their
ratings froze and have been accruing from a closed league since. Not
relevant to 2026/27, but wrong in the historical table.

---

## 10. Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .
cp .env.example .env        # then fill in

python scripts/download_domestic.py   # ~300k matches, a few minutes
python scripts/download_uefa.py       # ~4k matches
python scripts/check_coverage.py      # join-quality report

python scripts/train.py               # 7s  -> data/models/dixon_coles.json
python scripts/predict.py             # 144 fixture forecasts
python scripts/simulate.py            # 10,000 seasons

python scripts/backtest_2025_26.py    # validate against a known season
pytest                                # 109 tests
```

Training happens **once**. Prediction and simulation load the saved
model rather than refitting.

---

## 11. Layout

```
pitchiq/
  clubs.py                  name resolution across sources
  config.py                 credentials and paths
  matches.py                one date-ordered stream over every source
  ingest/
    football_data_uk.py     domestic results and odds
    openfootball.py         UEFA competitions and qualifiers
  models/
    elo.py                  sequential ratings
    league_strength.py      per-country offset and scale
    outcome.py              rating gap -> outcome probabilities
    dixon_coles.py          attack/defence -> scorelines
    ensemble.py             blend, plus calibrated scoreline grid
    store.py                save and load trained models
  sim/
    tournament.py           Monte Carlo over the whole competition
  eval/
    metrics.py              RPS, log loss, odds de-vigging
scripts/                    download, train, predict, simulate, backtest
data/
  external/                 the 2026/27 draw (committed)
  models/                   trained models (generated)
  predictions/              forecasts and simulation output (committed)
  raw/  processed/          downloaded and derived data (generated)
tests/                      109 tests
```

---

## 12. Decisions worth remembering

**Learn strength from domestic leagues, not the Champions League.** 300k
matches against 4k. The tournament is too small to learn from.

**Ingest the qualifying rounds.** They looked like a footnote and were
the highest-value data decision made. Doubled the cross-league sample
and tripled the number of linked countries.

**Never fuzzy-match club names.** Exact tiers, hand-checked aliases,
`None` when unknown, and a test that scans for silent merges.

**Verify analytic gradients numerically.** A wrong gradient converges
quietly to a wrong answer.

**Fix the refit schedule before comparing models.** Rolling Elo against
frozen Dixon-Coles measures the schedule, not the models.

**Choose hyperparameters on a validation window.** The first ensemble
weight was picked on the test set and the resulting number was
optimistic.

**Report the number that is true, not the one that flatters.** In the
frozen case the ensemble adds nothing over Dixon-Coles alone, and the
documentation says so.
