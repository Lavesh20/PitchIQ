# PitchIQ — backlog

Ordered work list, merged from three sources:

- **Plan** — `PitchIQ_Full_Project_Plan.md`, the original project plan
- **Gaps** — `finalReport.md`, the gap analysis
- **Build** — findings from building and testing the current system

Ordered by value per unit of effort, not by the priority labels in the
gap report. Three of that report's four P0 items are Phase 7 roadmap
work rather than gaps in a V1 that has not shipped; they are listed
under *Deferred* with the reasoning.

Status against the plan's own V1 criteria (§35): six of eight met.
The two outstanding are **a reproducible feature pipeline** and
**demonstrated probability calibration**, which is why items 1 and 6
sit where they do.

---

## Tier 1 — cheap, and closes stated V1 criteria

### 1. Reliability curve and Brier score
**Source:** Plan §23 · **Effort:** ~1h

The plan states the test explicitly: *"If PitchIQ predicts 70% for a
class repeatedly, the observed frequency of that class should be close
to 70%."* That has never been plotted.

Interval coverage from the simulator (33/36 inside the 90% band) is a
different measurement and does not substitute for it.

**Done when:** a reliability diagram exists for match probabilities on
held-out data, Brier score sits alongside RPS and log loss in
`eval/metrics.py`, and any miscalibration is quantified rather than
assumed absent.

### 2. Significance testing as standard
**Source:** Gaps §12 · **Effort:** ~1h · **Partly done**

A paired bootstrap has been run once and immediately overturned a
reported result. On 1,981 held-out European matches:

```
ensemble vs elo+LS       -0.0012   p≈0.135   NOT significant
ensemble vs dixon-coles  -0.0052   p≈0.000   significant
elo+LS   vs dixon-coles  -0.0040   p≈0.040   significant
elo+LS   vs elo          -0.0028   p≈0.027   significant
```

The league-strength correction is real. **The ensemble's edge over its
best single component is not established.**

**Done when:** every headline comparison ships with a confidence
interval, and `how-it-works.md` is corrected — it currently presents the
ensemble as an improvement that the data does not support.

### 3. Correct the documentation
**Source:** Build · **Effort:** ~30m

Follows from item 2. `how-it-works.md` §5 overstates the ensemble.
Restate the frozen/rolling comparison with intervals, and say plainly
that in the frozen case — the one that matches the actual task —
Dixon-Coles alone is not distinguishable from the blend.

---

## Tier 2 — the substantive modelling work

### 4. Use the match statistics already downloaded
**Source:** Plan §26 · **Effort:** ~1 day · **Highest-value item here**

These columns have been in `domestic_matches.parquet` since the first
download and no model has ever read them:

| Column | Meaning | Coverage since 2015/16 |
|---|---|---|
| `HST`/`AST` | shots on target | 54% |
| `HS`/`AS` | shots | 54% |
| `HC`/`AC` | corners | 54% |
| `HY`/`AY`/`HR`/`AR` | cards | 58% |
| `hthg`/`htag` | half-time score | 63% |

Shots on target is among the strongest known predictors of underlying
team strength, precisely because it is less noisy than goals. A club
that creates chances but is finishing badly looks weak on goals alone,
and the model currently has no way to tell that apart from genuine
weakness.

**Approach:** fit attack/defence on shots on target as well as goals,
and use the shots-based ratings as a prior or second signal for the
goals model. Coverage is partial, so it must degrade gracefully to
goals-only where statistics are absent.

**Done when:** held-out RPS improves with a confidence interval that
excludes zero, or the idea is recorded as tested and rejected.

### 5. Propagate parameter uncertainty
**Source:** Gaps §4 (P0), Build (`how-it-works.md` §9) · **Effort:** ~2 days

The simulator treats fitted ratings as facts. It knows Arsenal's attack
is +0.94; it does not know that figure is roughly ±0.15, and it has no
idea that Sabah's ±0.4 is far wider on ten matches.

This is the direct cause of the favourites being too strong: Arsenal at
20.4% against a bookmaker's 15–18%.

**Approach:** sample parameter sets rather than using point estimates —
bootstrap over matches, or sample from the inverse Hessian at the
optimum. Draw a fresh parameter set per simulated season.

**Done when:** top-club probabilities flatten, tail probabilities rise,
interval coverage holds at or above current levels, and held-out RPS
does not degrade.

### 6. Calibration across multiple seasons
**Source:** Gaps §11 · **Effort:** ~half a day

The 92% coverage figure comes from one season and 36 clubs — 36 data
points. That is far too thin to claim the uncertainty estimates are
sound.

**Done when:** the 2025/26 backtest is repeated for 2020/21 through
2024/25 and coverage is reported per season with an interval.

---

## Tier 3 — the model comparison the plan asked for

### 7. Feature layer
**Source:** Plan §19 · **Effort:** ~2 days

Models currently see only `(home, away, goals, date)`. The plan
specifies rolling form over 3/5/10 matches, home and away splits,
**rest days**, competition stage and matchday, and head-to-head where
the sample supports it.

Rest days is the notable omission — a real effect, and the dates needed
to compute it are already in the match stream.

**Done when:** features are computed strictly from matches before each
fixture's date, with a leakage test proving it, and a feature table is
reproducible from the match stream.

### 8. Gradient boosting
**Source:** Plan §22, Model 4 · **Effort:** ~1 day · **Depends on 7**

The plan names four models to compare. Elo, logistic regression and
Dixon-Coles exist; XGBoost or LightGBM does not. It is the model that
would consume the feature layer, and the one most likely to find
interactions the structural models cannot express.

**Done when:** benchmarked against Dixon-Coles on the same walk-forward
split, with an interval on the difference.

### 9. Model versus market
**Source:** Plan §27 · **Effort:** ~half a day

The plan asks for three comparisons: model only, market only, and
**model + market**. Only the first two exist.

The third answers the question that actually matters — whether PitchIQ
carries information the market does not, or merely reproduces it.

**Done when:** a blend of model and de-vigged closing odds is scored
against the market alone, with an interval.

---

## Tier 4 — robustness

### 10. League-strength stability
**Source:** Gaps §13 · **Effort:** ~half a day

Romania's −117 offset rests on 44 matches. Bootstrap the country
offsets and scales and report intervals; shrink harder where evidence
is thin.

### 11. Complete the tiebreakers
**Source:** Gaps §15, Build · **Effort:** ~2h

Implemented: points, goal difference, goals scored, away goals, wins,
away wins. Missing: opponents' collective points, goal difference and
goals, then disciplinary points. Affects roughly one simulated season
in a thousand — worth doing for correctness, not for accuracy.

### 12. Rating staleness as a general problem
**Source:** Gaps §17 · **Effort:** ~half a day

Russian clubs froze in 2022 and have been accruing rating in a closed
league since. Russia is the visible case; the general one is any club
whose rating has not been tested against outside opposition recently.
Track time since last cross-league match and widen uncertainty with it.

### 13. Competition rules as data
**Source:** Gaps §14 · **Effort:** ~half a day

Format rules are embedded in `sim/tournament.py`. UEFA changed the
format in 2024/25 and again for 2026/27. Move bands, bracket and
tiebreakers into a versioned config keyed by season.

---

## Deferred — Phase 7 in the plan, not V1

The gap report marks these P0. They are real, and they are the plan's
own later phases (§26, §28), not gaps in the current stage.

### Live prediction
**Source:** Plan §28, Gaps §6

Needs minute-by-minute match state. No free source provides it, and
in-play modelling is a different system from pre-match forecasting.

### Player and squad information
**Source:** Plan §26, Gaps §5

Injuries, lineups, suspensions, transfers. **Blocked by data
availability, not design:** FBref is Cloudflare-blocked and its terms
forbid automated access; no free lineup source was found. Revisit only
alongside a decision on a paid provider such as Sportmonks.

This is most of the remaining gap to the bookmaker line, so it is the
highest-value *deferred* item, not a low-value one.

### Deployment and API
**Source:** Gaps §16

Never scoped. Revisit if PitchIQ becomes a product rather than an
engine.

---

## Rejected

**"Individual match forecasting is not a first-class layer"**
(Gaps §7, marked P0) — it already is. `scripts/predict.py` writes all
144 fixture forecasts, with probabilities, expected goals and likeliest
scoreline, to `data/predictions/ucl_2026_27_fixtures.csv`.

The gap report was written from `how-it-works.md` without reading the
repository, which also explains why it lists model versioning as absent
when `models/store.py` already records config, match counts, date range
and timestamp.

---

## Known deviation from the plan

The plan scopes V1 as **Champions League only** (§2) and makes the
source comparison between openfootball and football-data.org the
immediate next move (§34).

Neither happened. UCL alone is roughly 4,000 matches across clubs that
rarely meet twice — too sparse to learn stable ratings — so the build
went straight to domestic leagues plus all three UEFA competitions,
which is the expansion §2 anticipated for later. football-data.org was
dropped after testing showed its free tier returns only three seasons
of history, which removed the need for that comparison.

Recorded here because it was a deliberate departure, not an oversight.
