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
is slightly ahead.

An early version compared rolling Elo against frozen Dixon-Coles and
concluded Elo was better. That was measuring the refit schedule, not the
models.

### A correction to an earlier claim

This document previously presented the ensemble as an improvement. A
paired bootstrap does not support that:

```
ensemble vs elo + league strength   -0.0012   p ~ 0.135   NOT significant
ensemble vs dixon-coles             -0.0052   p ~ 0.000   significant
elo + league strength vs elo        -0.0028   p ~ 0.027   significant
```

The league-strength correction is real. **The ensemble's edge over its
own best component is not established.** On 1,981 European matches a
difference of 0.0012 is indistinguishable from luck, and the honest
statement is that the blend and its best part cannot be told apart.

Every comparison in the rest of this document ships with an interval
for that reason.

---

## 5A. The feature layer

The models above see four things per match: two clubs, the goals, the
date. Nothing about form, fatigue, or a side creating chances and
missing them.

55 numbers were added, all knowable **before kick-off**: form over 3, 5
and 10 matches; goals for and against; home-only and away-only records;
days since the last match and matches in the last fortnight; shots and
shots on target; head-to-head; Elo; and fixture context.

### Leakage, and why the loop is slow on purpose

A feature that accidentally contains information from after the match
is catastrophic *and* invisible: scores improve, everything looks
brilliant, and the model is worthless in use.

The usual construction is a rolling window plus a `shift`, and a hope
that the shift is right. That was rejected. Instead the code walks the
fixture list in date order holding per-club memory and, for each match,
reads the memory and writes the row **before** folding that match's
result in. A match cannot see itself, because its result has not
arrived when its row is written.

The property is structural rather than a rule to be maintained. It is
tested anyway: alter any match to 9–0, rebuild, and assert nothing at or
before it moves — plus the mirror test, that altering a result *must*
change the next row, since code returning zeroes would pass the first
test perfectly.

Cost: 40 seconds instead of 3.

### Do the features earn their place?

Same model class, same split, same rows; only the columns vary.

| | RPS on 38,869 held-out matches |
|---|---|
| Elo alone | 0.2101 |
| **Elo + features** | **0.2091** |
| features without Elo | 0.2127 |

```
gain +0.00106 RPS   95% CI [+0.00073, +0.00138]
```

Real, and small. Note the third row: **Elo still carries most of the
signal**. Form and rest add on top; they do not replace it.

---

## 5B. Gradient boosting

The fourth model the plan asked for. Not a downloaded brain — a blank
algorithm that builds a small decision tree, sees where it was wrong,
builds another to fix those mistakes, and repeats a few hundred times.

It can express what a formula cannot: *a short rest hurts, but only
after a European away leg.*

Walk-forward by season, every model refit at each boundary:

| | All 37,765 | European 1,359 |
|---|---|---|
| base rate | 0.2282 | 0.2314 |
| dixon-coles | 0.2141 | **0.2054** |
| elo | 0.2099 | 0.2092 |
| logistic on the same features | 0.2088 | 0.2103 |
| **gradient boosting** | **0.2085** | 0.2085 |

| Comparison | All | European |
|---|---|---|
| boosting vs logistic | +0.00033 **sig** | +0.00177 **sig** |
| boosting vs elo | +0.00132 **sig** | +0.00067 not sig |
| boosting vs dixon-coles | +0.00552 **sig** | −0.00309 **not sig** |

Two separate findings. **On domestic football boosting wins clearly.**
**On European football nothing is distinguishable** — every interval
crosses zero, because 1,359 matches cannot separate forecasts differing
by 0.003.

What the trees actually use:

```
elo_diff           206      <- dominates
elo_expected       122
form over 10        23
form over 3         17
shots on target     13      <- the unused columns earn their place
rest days            9
```

Read honestly: **boosting is "Elo, with small corrections."**

### The refit-schedule trap, again

The first run of this comparison showed boosting ahead by 0.0084. That
number was junk: Dixon-Coles was fitted once and asked to predict three
years forward while boosting's Elo features kept updating. Refitting
both per season cut the gap to 0.0055.

**A third of the apparent gain was refit schedule.** The same mistake as
before, in a new place.

**Dixon-Coles stays regardless.** Boosting predicts win/draw/loss, never
2–1, and the league phase is settled on goal difference.

---

## 5C. Against the bookmakers

The plan asks three questions and only two had been answered. The third
is the one that matters: **do we know anything the market does not?**

If we do, blending should beat the market alone.

36,325 priced domestic matches, walk-forward:

```
market closing line    0.2036   <- best
stacked                0.2035
model + market         0.2036
gradient boosting      0.2086
base rate              0.2281
```

Three attempts to find an edge:

- **Linear blend** — the validation year chose **0% weight on the
  model**, in all three folds, off a curve that rises from the first
  step. Every drop of model made it worse.
- **Log-odds stacking**, which can reshape rather than only slide:
  +0.00009 RPS, CI [−0.00005, +0.00022]. Nothing.
- **By division** — no soft spot. Fifth-tier English football is priced
  about as sharply as the Premier League.

**Conclusion: PitchIQ carries no information the closing line lacks.**

Read the other way: knowing nothing scores 0.2281, the market 0.2036,
and we reach 0.2086. **We cover about 80% of that distance** using only
past results and dates — no injury news, no lineups.

The remaining 20% is team news. That is a data problem, not a modelling
problem, and no amount of further modelling closed it.

**Scope limit:** football-data.co.uk prices domestic leagues only. There
are **no odds for Champions League matches** in this archive, so nothing
here says how the model compares to the market on the competition it
was built for.

---

## 5D. Calibration

A different property from accuracy. The plan states the test (§23): *if
PitchIQ predicts 70% for a class repeatedly, that class should happen
about 70% of the time.*

A model can rank matches perfectly and still overstate every number it
prints — fine for anything about ordering, useless for a simulator that
multiplies probabilities across a season.

| model | RPS | Brier | ECE | over-confidence |
|---|---|---|---|---|
| elo | 0.2099 | 0.3059 | 0.0063 | +0.0104 |
| dixon-coles | 0.2141 | 0.3101 | **0.0033** | +0.0060 |
| boosting | 0.2085 | 0.3045 | 0.0058 | +0.0096 |
| market | 0.2036 | 0.2995 | 0.0068 | −0.0079 |

**Dixon-Coles is better calibrated than the bookmakers.** When it says
70%, it means 70%.

Everything tracks the truth closely up to about 70% stated probability.
Above that all three models fall below the line — they claim more than
happens:

```
dixon-coles       n     claimed   happened
  70%-80%      1,332      0.742      0.724
  80%-90%        386      0.832      0.806
  90%-100%        28      0.921      0.714
```

Reliability diagram: `data/figures/reliability.png`.

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

### 6.1 Parameter uncertainty

The simulator above has a flaw: **it treats the ratings as facts.**
Arsenal's attack is 1.04 in all 10,000 seasons, and so is the rating of
a club seen 22 times. The possibility that we have simply misjudged a
team never enters.

Two penalty takers make the point:

| | taken | scored | rating |
|---|---|---|---|
| Player A | 100 | 80 | 80% |
| Player B | 5 | 4 | 80% |

Same number, completely different confidence. The old simulator treats
both as exactly 80%, every time.

**How much this costs, measured.** Every complete domestic league season
since 2015 — **4,225 club-seasons** — was simulated from a model that
knew nothing past its first fixture, and where clubs actually finished
was regressed on where they were predicted to finish:

```
actual = 0.828 x predicted        95% CI [0.799, 0.858]
```

A calibrated forecast gives 1.000. **Predictions are about 21% more
spread out than reality supports** — good clubs pushed too far up, weak
ones too far down. The errors cancel to zero overall, but only because
the extremes are wrong in opposite directions.

Domestic leagues rather than European ones because of sample size: UEFA
replaced the group stage in 2024/25, so only two seasons have used the
current format — 72 club-seasons, far too few to tell a calibrated
forecast from an over-confident one.

**The fix.** Refit Dixon-Coles 200 times on reweighted copies of the
record and let each simulated season draw a different set of ratings.
The reweighting is a Bayesian bootstrap — Dirichlet weights rather than
draw-with-replacement, so a club with 22 matches cannot be left with
none at all and collapse to the ridge prior, which would read as huge
uncertainty when it is an artefact of the resampling.

The spread it finds is exactly where it should be:

```
Manchester City   attack 1.149  +/- 0.038
Bayern Munich     attack 1.244  +/- 0.040
Shakhtar Donetsk  attack 0.382  +/- 0.139
Sabah             attack 0.271  +/- 0.243     <- 6x wider
```

**What it achieved.** Same 1,448 club-seasons, same fixtures, same seed;
the only difference is fixed versus drawn ratings:

| | fixed | sampled |
|---|---|---|
| slope | 0.856 | **0.884** |
| 90% interval coverage | 89.6% | **90.7%** |
| coverage, clubs predicted top 3 | 85.9% | **92.2%** |
| mean interval width | 13.7 | 14.1 places |

```
paired change in slope +0.0274   95% CI [+0.0225, +0.0327]   significant
```

Real, significant, and aimed correctly — the favourites band improved
6.3 points, and the 90% interval is now honest. **But it closes 19% of
the gap, not all of it.** The bootstrap captures *"we may have measured
this club wrong"*; it cannot capture *"this club changed over the
summer"*, and that is most of what remains.

**What it did not achieve.** It was predicted here that Arsenal's title
chance would fall from 20.4% to around 17%. It did not:

| | fixed | sampled |
|---|---|---|
| Arsenal wins it | 20.4% | 20.5% |
| top six clubs combined | 78.9% | 77.8% |
| Barcelona top 8 | 55.1% | **52.3%** |
| Manchester City top 8 | 63.6% | **61.8%** |
| Sabah top 8 | 0.0% | 0.15% |

The reason is visible in the spreads above: **the contenders have the
tightest error bars of anyone.** Manchester City's rating is not in
doubt. The uncertainty sits with clubs that have no title chance either
way. And winning the competition means surviving four knockout ties,
which is already close to four coin flips, so rating uncertainty adds
little on top of randomness that large.

The effect shows up on **qualification**, where one season's spread of
results decides things, and washes out by the trophy.

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

Ten thousand seasons, with ratings drawn from the 200-sample bootstrap
so the forecast carries our uncertainty about the clubs as well as the
luck inside the matches.

| Club | Top 8 | Wins it |
|---|---|---|
| Arsenal | 73.9% | **20.5%** |
| Manchester City | 61.8% | **18.1%** |
| Bayern Munich | 71.6% | **14.5%** |
| Liverpool | 67.4% | **10.7%** |
| Paris Saint-Germain | 54.8% | 8.0% |
| Real Madrid | 53.4% | 6.3% |
| Barcelona | 52.3% | 6.0% |
| Inter Milan | 46.6% | 4.4% |
| … | | |
| Sabah | 0.15% | 0.0% |

Full table: `data/predictions/ucl_2026_27_simulation.csv`.

The sampled run takes 38 seconds rather than 0.6, because the scoreline
grids are rebuilt once per parameter draw.

---

## 9. Known limitations

**Predictions are still about 17% too spread out.** Measured on 4,225
club-seasons, finishing positions regress on predicted positions with a
slope of 0.884 after the uncertainty work, against 1.000 for a
calibrated forecast. Sampling the ratings closed 19% of the original
gap. What remains is squad change between seasons — transfers, managers,
rebuilds — which resampling the past cannot capture.

**Arsenal at 20.5% is still higher than a bookmaker would quote**
(typically 15–18%), and it is now clear that parameter uncertainty is
not the explanation: the contenders are the best-measured clubs in the
field. The likeliest causes are the missing squad information and the
absence of any European market to calibrate against.

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

python scripts/build_features.py       # 40s -> 55 features per match
python scripts/train.py               # 7s  -> data/models/dixon_coles.json
python scripts/bootstrap_uncertainty.py  # ~25 min, run once
python scripts/predict.py             # 144 fixture forecasts
python scripts/simulate.py            # 10,000 seasons

python scripts/backtest_2025_26.py    # validate against a known season
python scripts/evaluate_boosted.py    # four models on one split
python scripts/evaluate_market.py     # model against the closing line
python scripts/evaluate_calibration.py   # are the probabilities honest?
python scripts/evaluate_simulator.py     # are the season ranges honest?
python scripts/evaluate_uncertainty.py   # did sampling the ratings help?
pytest                                # 179 tests
```

Training and bootstrapping happen **once**. Prediction and simulation
load the saved model and samples rather than refitting.

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
  features/
    rolling.py              the causal pass; leakage impossible by design
    build.py                assembles rolling + Elo + fixture context
  models/
    elo.py                  sequential ratings
    league_strength.py      per-country offset and scale
    outcome.py              rating gap -> outcome probabilities
    dixon_coles.py          attack/defence -> scorelines
    boosted.py              gradient boosting over the feature layer
    ensemble.py             blend, plus calibrated scoreline grid
    uncertainty.py          bootstrap: error bars on every rating
    store.py                save and load trained models
  sim/
    tournament.py           Monte Carlo over the whole competition
    league.py               a domestic season, for validation at scale
    draws.py                spreads seasons across parameter draws
  eval/
    metrics.py              RPS, log loss, Brier, odds de-vigging
    calibration.py          reliability curves and over-confidence
    market.py               bookmaker odds -- evaluation only, never a feature
    backtest.py             the walk-forward pass, cached
scripts/                    download, train, predict, simulate, evaluate
data/
  external/                 the 2026/27 draw (committed)
  figures/                  reliability diagram
  models/                   trained models and bootstrap samples (generated)
  predictions/              forecasts and simulation output (committed)
  raw/  processed/          downloaded and derived data (generated)
tests/                      179 tests
```

---

## 12. Decisions worth remembering

**Learn strength from domestic leagues, not the Champions League.** 300k
matches against 4k. The tournament is too small to learn from.

**Bookmaker odds live in `eval/`, never in `features/`.** If a price
became a model input the trees would learn to copy it; the backtest
would look superb and the model would be worthless on any unpriced
fixture — which includes every Champions League match in the archive.
A test asserts the odds columns appear in no feature list.

**Every headline comparison ships with an interval.** Four claims made
during this build were overturned by paired bootstraps, two of them the
author's own from the same session. Without an interval, "0.2085 against
0.2092" says nothing about whether the gap would survive a different set
of matches.

**Refit every model on the same schedule before comparing them.** Twice
a comparison here measured which model had seen more recent football
rather than which model was better, and both times it favoured the newer
model by a wide margin.

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
