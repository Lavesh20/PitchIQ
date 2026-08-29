# PitchIQ — Final Gap Analysis Report

## 1. Purpose

This report reviews the current PitchIQ design and identifies the remaining gaps between:

1. the **current research/model implementation**, and
2. a **robust, production-ready football forecasting system** capable of pre-match, tournament-level, and eventually live predictions.

The review is based primarily on the current `PitchIQ — how it works` document, including its data pipeline, club resolution, Elo, league-strength correction, Dixon-Coles model, evaluation, tournament simulation, validation results, known limitations, and current repository layout.

The existing document is already technically mature. The gaps below are therefore not a statement that the model is fundamentally broken. Most are areas where the system needs stronger statistical treatment, broader feature coverage, operational controls, or explicit production architecture.

---

# 2. Executive Summary

The current PitchIQ system already has a strong core:

```text
300k+ domestic matches
        +
4k+ UEFA matches
        ↓
canonical club resolution
        ↓
Elo / league strength / Dixon-Coles
        ↓
match score distributions
        ↓
Monte Carlo UCL simulation
        ↓
2026/27 tournament probabilities
```

The strongest parts are:

- very large training data relative to the UCL-only approach
- use of domestic football to estimate club strength
- inclusion of UEFA qualifying rounds
- strict club-name resolution
- explicit prevention of some forms of leakage
- analytical-gradient implementation with numerical verification
- chronological evaluation
- separation of frozen vs rolling evaluation
- full tournament simulation
- internal simulation consistency checks
- honest documentation of known limitations

However, the system still has several important gaps.

### Highest-priority gaps

| Priority | Gap | Why it matters |
|---|---|---|
| P0 | Parameter uncertainty is not propagated | Causes overconfident favourites and overly sharp tournament probabilities |
| P0 | Live prediction architecture is missing | This is required for the original PitchIQ objective of live match forecasting |
| P0 | Player/squad information is missing | Team-level ratings cannot react to injuries, suspensions, lineups, transfers, or squad quality |
| P0 | Individual match forecasting is not a first-class product layer | Tournament simulation depends on fixture probabilities and needs a clearly defined match-forecast engine |
| P1 | Source reconciliation is not formalized enough | Multiple providers can disagree and bad joins silently corrupt ratings |
| P1 | Data as-of timestamps are not explicit | Preventing leakage requires knowing exactly when information became available |
| P1 | Model/data versioning is incomplete | Predictions need reproducibility and auditability |
| P1 | Calibration validation needs to be stronger | A small number of observed teams/positions is not enough to establish reliable uncertainty |
| P1 | Competition rules are too embedded in simulation logic | UEFA rules can change between seasons |
| P1 | Data freshness and pipeline monitoring are missing | A production forecast system must detect stale or incomplete feeds |
| P2 | League-strength model needs deeper robustness analysis | Sparse cross-league evidence can produce unstable country effects |
| P2 | Penalty shootouts are oversimplified | Team-specific shootout tendencies are ignored |
| P2 | Russian-club/rating staleness is not generalized as a model-state problem | Other unavailable or isolated teams can create the same issue |
| P2 | Prediction drift/model drift monitoring is missing | Performance can deteriorate without obvious pipeline failures |
| P2 | Statistical significance / uncertainty around headline metrics is missing | Small score differences may not be meaningful |
| P2 | Deployment/API/product architecture is under-specified | The research pipeline is not yet a complete product |

---

# 3. What Is Already Strong

## 3.1 Data Scale

The current system combines approximately:

- 304,039 matches overall
- 299,799 domestic matches
- 4,240 UEFA matches
- 1,533 clubs
- 83 countries

The document explicitly explains why expanding beyond UCL-only data was necessary: the Champions League alone is too sparse to learn stable club strength. This is one of the strongest architectural choices in the project.

## 3.2 Qualifying Rounds

Adding UEFA qualifying rounds increased:

```text
UEFA matches       1,997 → 4,240
Cross-league data ~1,968 → 4,206
Countries linked     21 → 56
Distinct clubs       136 → 346
```

This is important because qualifying rounds create otherwise-missing cross-country links.

## 3.3 Club Identity Resolution

The current three-tier strategy is strong:

```text
Exact table
    ↓
Hand-checked alias table
    ↓
Country-scoped normalization
    ↓
None if unresolved
```

The design correctly rejects automatic fuzzy matching.

This is especially important for clubs with deceptive name similarity.

## 3.4 Statistical Model Stack

The system builds progressively:

```text
Elo
 ↓
League-strength adjustment
 ↓
Outcome mapper
 ↓
Dixon-Coles
 ↓
Ensemble / scoreline distribution
```

This is a sensible research progression.

## 3.5 Evaluation Discipline

The project explicitly corrected an early evaluation mistake where the comparison between models was confounded by different refit schedules.

It also documents that hyperparameters must be selected on a validation window that ends before the test window.

That is excellent practice.

## 3.6 Tournament Simulation

The simulator covers:

- league phase
- qualification
- play-offs
- round of 16
- quarter-finals
- semi-finals
- final
- extra time
- penalties
- abolished away-goals rule
- bracket structure
- draw mechanics

It also performs conservation checks.

This is a strong foundation.

---

# 4. Gap 1 — Parameter Uncertainty Is Not Propagated

## Status

**Critical / P0**

## What is currently done

The model estimates attack/defence/rating parameters and then treats those fitted values as fixed when running the tournament simulation.

The current document itself identifies this:

> The model treats its own ratings as certain.

## Why this is a gap

A fitted rating is an estimate, not a fact.

For a well-observed elite club, uncertainty may be relatively small.

For a club with limited European history, uncertainty may be large.

For example:

```text
Club A:
strength = 1650 ± 20

Club B:
strength = 1510 ± 120
```

If both are treated as exactly 1650 and 1510, the simulator becomes artificially confident.

This likely contributes directly to the document's observation that the favourites appear too strong.

## What should be done

Propagate parameter uncertainty into simulation.

Target architecture:

```text
Fit model
   ↓
Estimate parameter uncertainty
   ↓
Sample plausible parameter set
   ↓
Generate match probabilities
   ↓
Simulate tournament
   ↓
Repeat
```

Potential statistical implementations include:

- bootstrap
- approximate covariance/Hessian sampling
- Bayesian posterior sampling
- hierarchical Bayesian models

## Success criterion

The system should show whether:

```text
Top-favourite probabilities flatten
tail probabilities increase
interval coverage improves
```

without materially degrading predictive scoring.

---

# 5. Gap 2 — Player and Squad Information

## Status

**Critical / P0**

## Current state

The system currently has no:

- injuries
- suspensions
- transfers
- manager changes
- expected lineup
- starting XI strength
- player availability

The current document explicitly lists this as a limitation.

## Why this is a gap

Club-level historical ratings cannot react quickly to squad changes.

A club's effective strength before a fixture depends on who is actually available.

Conceptually:

```text
Base team strength
        +
Expected XI
        +
Player availability
        +
Bench quality
        ↓
Effective match strength
```

This matters particularly for elite teams where a small number of high-impact players can move expected scoring substantially.

## What should be done

Introduce a player/squad layer in stages.

### Stage 1

- expected XI
- injury status
- suspension status

### Stage 2

- player-level contribution estimates
- position-aware impact
- replacement quality

### Stage 3

- transfer changes
- managerial changes
- squad depth

Do not add all of this at once.

---

# 6. Gap 3 — Live Prediction Is Missing

## Status

**Critical / P0**

## Current state

The current document is primarily a pre-season / tournament forecasting system.

It does not define a production live pipeline.

## Why this is a gap

The original PitchIQ goal included prediction of live matches.

A tournament forecast and a live match forecast are different products.

A live system must respond to events such as:

```text
0–0, 20'
Goal
Red card
Penalty
Substitution
xG movement
Shot volume
Game state
```

## Required architecture

```text
Live provider
      ↓
Event ingestion
      ↓
Current match state
      ↓
Feature update
      ↓
Live model
      ↓
Updated probability
      ↓
Repeat
```

## Minimum live features

```text
minute
score
home/away state
red cards
yellow cards
substitutions
shots
xG
possession
events
```

## Success criterion

For a live fixture, PitchIQ should be able to produce:

```text
P(home win)
P(draw)
P(away win)
```

at arbitrary timestamps during the match.

---

# 7. Gap 4 — Individual Match Forecasting Is Not a First-Class Layer

## Status

**Critical / P0**

## Current state

The current system is framed around forecasting the whole UCL competition.

## Why this is a gap

Tournament simulation requires fixture-level probability distributions.

The model therefore needs an explicit interface:

```text
Match Forecast Engine
```

rather than treating match probabilities as an implicit internal byproduct.

## Required input

```text
home club
away club
prediction timestamp
available information
```

## Required output

```text
home win probability
draw probability
away win probability
expected goals
scoreline probability grid
confidence / uncertainty
```

The tournament simulator should consume this output.

Architecture:

```text
Team Strength Engine
        ↓
Match Forecast Engine
        ↓
Tournament Simulator
```

This separation also makes live forecasting much easier later.

---

# 8. Gap 5 — Source Reconciliation Needs a Formal Contract

## Status

**High / P1**

## Current state

The project uses multiple sources and has strong name-resolution rules.

## Why this is a gap

If two sources disagree, PitchIQ needs a deterministic rule for deciding:

```text
which record wins
```

Possible conflicts:

```text
different kickoff date
different score
different team name
missing match
different stage
different competition classification
```

Incorrect reconciliation can silently contaminate the entire model.

## Required design

For overlapping source records:

```text
Source A
Source B
Source C
   ↓
Canonical identity
   ↓
Match comparison
   ↓
quality score
   ↓
preferred source
   ↓
provenance retained
```

Every canonical record should retain:

```text
source
source_record_id
ingestion_timestamp
```

---

# 9. Gap 6 — Explicit As-Of-Time Semantics

## Status

**High / P1**

## Why this is a gap

The model can avoid obvious score leakage and still leak information if it uses data that became available after the prediction timestamp.

Examples:

```text
injury announced at 15:00
lineup announced at 19:00
match starts 20:00
```

A 14:00 prediction cannot use the lineup.

## Required fields

For externally sourced information, preserve:

```text
observed_at
effective_at
source_updated_at
```

The feature pipeline should operate under:

```text
AS_OF = prediction timestamp
```

and reject any future information.

This should be tested automatically.

---

# 10. Gap 7 — Model and Dataset Versioning

## Status

**High / P1**

## Why this is a gap

A prediction needs to be reproducible.

If PitchIQ produces:

```text
Arsenal champion probability = 20.4%
```

we must know exactly:

```text
model version
training cutoff
data version
feature version
code version
hyperparameters
random seed
```

## Required model metadata

Example:

```text
model_id
trained_at
training_cutoff
dataset_version
feature_version
code_version
random_seed
```

This is essential for:

- debugging
- experiment comparison
- auditing
- production rollbacks
- historical prediction tracking

---

# 11. Gap 8 — Calibration Evidence Needs to Be Broader

## Status

**High / P1**

## Current state

The system reports strong simulation coverage for the 2025/26 known-season validation.

## Why this is not enough

A 36-club tournament gives only a small sample for assessing interval calibration.

The reported 33/36 coverage is encouraging but does not establish that all uncertainty intervals are reliable.

## What should be done

Evaluate calibration across multiple historical seasons and confidence levels:

```text
50%
68%
80%
90%
95%
```

Report:

```text
empirical coverage
expected coverage
calibration slope
calibration intercept
reliability plots
```

Use repeated historical holdouts, not one known season.

---

# 12. Gap 9 — Statistical Significance of Model Comparisons

## Status

**High / P1-P2**

## Current state

Examples of reported differences include:

```text
0.2040
vs
0.2052
```

## Why this is a gap

A small difference in RPS does not automatically mean one method is genuinely better.

Need to establish whether performance differences are robust rather than sampling noise.

## What should be done

Add:

- paired bootstrap confidence intervals
- repeated walk-forward windows
- paired forecast tests where appropriate
- confidence intervals around performance metrics

Headline results should be reported as:

```text
RPS = 0.2040
95% CI = ...
```

where appropriate.

---

# 13. Gap 10 — League-Strength Model Robustness

## Status

**High / P1-P2**

## Current state

The working correction is:

```text
adjusted =
league_mean
+ scale × (rating − league_mean)
+ offset
```

The model uses cross-country matches to infer country corrections.

This is a thoughtful solution.

## Why it is still a gap

Some countries have much more international evidence than others.

For example:

```text
549 European matches
vs
44 European matches
```

A sparse country can generate a much noisier estimate.

## What should be added

Perform sensitivity analyses for:

- minimum sample size
- parameter shrinkage
- prior strength
- season range
- removing individual countries
- removing individual clubs

The goal is to quantify:

> How much can one country's inferred strength move if sparse evidence changes?

---

# 14. Gap 11 — Competition Rules Should Be Versioned

## Status

**High / P1**

## Current state

The tournament simulator contains detailed UCL structural rules.

## Why this is a gap

Competition rules can change from season to season.

Hardcoding rules into simulation logic makes future maintenance dangerous.

## Better architecture

```text
Competition Rules
       ↓
Tournament Engine
```

Example conceptual configuration:

```text
rules/2026_27.yaml
rules/2027_28.yaml
```

Rules should define:

```text
league-phase size
qualification boundaries
play-off structure
seeding
draw restrictions
tiebreakers
extra-time rules
penalty rules
final venue rules
```

Then:

```text
simulate(season="2026/27")
```

loads the correct rule set.

---

# 15. Gap 12 — Tournament Tiebreakers Are Incomplete

## Status

**High / P1**

The current documentation acknowledges missing UEFA tiebreakers.

This is not catastrophic because the document estimates they affect only a small number of simulations, but it is still a rules-completeness gap.

## What should be done

Implement the complete official order of tiebreakers.

The simulator should then have a test such as:

```text
given controlled table
→ verify exact UEFA ranking order
```

This should be treated as a deterministic rules-engine test.

---

# 16. Gap 13 — Penalty Shootouts Are Oversimplified

## Status

**Medium / P2**

## Current state

Penalty shootouts are modelled as a coin flip.

## Why this is a gap

There can be systematic team-level differences in:

- penalty taker quality
- goalkeeper penalty saving
- coaching
- tournament experience

A coin flip is a reasonable baseline, but it throws away potentially useful information.

## Future improvement

Introduce a weakly regularized shootout model only after establishing that historical evidence supports meaningful signal.

Do not overfit this area because shootouts are rare.

---

# 17. Gap 14 — Rating Staleness Needs a General Solution

## Status

**Medium / P2**

The document highlights Russian clubs as one example of stale ratings.

## Why this is broader than Russia

The same problem can arise whenever a club:

- stops playing in Europe
- changes domestic competition
- has a long inactive period
- changes ownership/structure
- changes squad dramatically

## Required concept

Each rating should have:

```text
rating_age
last_updated
activity_level
uncertainty
```

A frozen rating should become more uncertain over time rather than simply remaining a fixed number forever.

---

# 18. Gap 15 — Data Freshness and Ingestion Monitoring

## Status

**High / P1 for production**

## Current state

The repository documents download commands, but not a formal freshness/monitoring layer.

## Why this is a gap

Production data can fail silently.

Examples:

```text
Expected UEFA matches today: 8
Downloaded: 6
```

or:

```text
Last update: 12 hours ago
```

The model may still run and produce plausible-looking but stale forecasts.

## Required monitoring

Track:

```text
last successful ingestion
expected records
received records
new records
changed records
missing records
source latency
HTTP errors
schema changes
```

---

# 19. Gap 16 — Model Drift and Data Drift Monitoring

## Status

**Medium / P2**

Once PitchIQ is live, the model's environment changes.

Possible drift:

```text
scoring rates
home advantage
team strength distribution
competition format
transfer behaviour
manager effects
```

## Required monitoring

Track:

```text
feature drift
prediction drift
calibration drift
RPS/log-loss over time
data distribution changes
```

The system should alert when performance deteriorates.

---

# 20. Gap 17 — 2026/27 Validation Should Be Multi-Layered

## Status

**High / P1**

The known-season backtest is excellent.

But the final system should validate at three levels.

### Level 1 — Match level

```text
1X2
scoreline
goals
```

### Level 2 — Season level

```text
league table
finishing position
top 8
R16
QF
SF
final
champion
```

### Level 3 — Calibration

```text
probability intervals
coverage
reliability
```

A model can be strong at individual matches while still being overconfident in the tournament simulation.

---

# 21. Gap 18 — The Product Architecture Is Under-Specified

## Status

**High / P1**

The current document is primarily an ML/research repository.

A production PitchIQ service needs:

```text
Data ingestion
      ↓
Canonical data
      ↓
Feature service
      ↓
Model service
      ↓
Prediction API
      ↓
Simulation API
      ↓
Dashboard
```

Possible service boundaries:

```text
ingestion service
feature service
forecast service
simulation service
API
frontend
monitoring
```

The research code should eventually be separated from online serving code.

---

# 22. Gap 19 — Live Event Ordering and Idempotency

## Status

**High / P1 when live work begins**

Live sports feeds can arrive:

- late
- duplicated
- out of order
- corrected

The live architecture must therefore support:

```text
event_id
sequence
observed_at
match_clock
idempotency
```

Example:

```text
Goal event received twice
→ apply once
```

or:

```text
Substitution arrives late
→ reconcile state
```

This is not required for the current offline model, but it is mandatory for a real live system.

---

# 23. Gap 20 — Retraining Strategy Is Not Fully Defined

## Status

**Medium / P2**

The current system distinguishes frozen and refit evaluation, which is good.

But production needs an explicit policy.

Questions to define:

```text
When is the model retrained?
What triggers retraining?
How much new data is required?
Does every new match update parameters?
Do live predictions use a fixed model?
How are new model versions promoted?
```

Potential policy:

```text
pre-season full retrain
+
scheduled periodic refresh
+
controlled in-season updates
```

The exact schedule should be evidence-driven.

---

# 24. Gap 21 — No Explicit Experiment Tracking

## Status

**Medium / P2**

Multiple model variants will eventually exist:

```text
DC-v1
DC-v2
DC+xG
DC+xG+players
Elo-v3
Ensemble-v4
```

A formal experiment record should capture:

```text
dataset
features
model
hyperparameters
validation window
test window
metrics
random seed
artifacts
```

MLflow or an equivalent system would help once experimentation expands.

---

# 25. Gap 22 — Feature Value Has Not Yet Been Quantified

## Status

**Medium / P2**

The roadmap proposes adding:

```text
xG
injuries
lineups
players
transfers
```

but each feature family should be evaluated experimentally.

Do not assume:

```text
more data = better model
```

For each feature family:

```text
baseline
      vs
baseline + feature
```

evaluate:

```text
RPS
log loss
Brier score
calibration
stability
```

and keep it only if the improvement survives out-of-sample testing.

---

# 26. Gap 23 — Market Benchmark Integration Needs More Formal Treatment

## Status

**Medium / P2**

The project already compares against bookmaker closing lines.

That is valuable.

The next step is to formalize three benchmarks:

```text
naive/base-rate model
PitchIQ model
market-implied model
```

and eventually:

```text
PitchIQ
Market
PitchIQ + Market
```

This clarifies whether PitchIQ is genuinely learning predictive information beyond the strongest public benchmark.

---

# 27. Gap 24 — Historical Naming / Identity Tests Need to Be a Permanent Gate

## Status

**High / P1**

The document correctly identifies identity resolution as one of the most dangerous silent failure modes.

It should therefore become a hard CI/data-pipeline gate.

Required tests:

```text
unresolved names = 0
unexpected merge count = 0
duplicate canonical IDs = 0
known split-club cases remain separate
```

This is especially important as new seasons and new providers are added.

---

# 28. Gap 25 — No Formal Data Contract

## Status

**Medium / P2**

The pipeline should have an explicit schema contract.

Example:

```text
match_id: required
date: required
home_team_id: required
away_team_id: required
home_score: required for completed match
away_score: required for completed match
competition: required
stage: controlled vocabulary
```

A schema contract protects downstream models from silent upstream changes.

---

# 29. Gap 26 — Uncertainty Needs to Exist at Multiple Levels

This is related to Gap 1 but broader.

PitchIQ needs uncertainty around:

```text
team strength
match outcome
scoreline
league-strength parameters
tournament probabilities
```

A single final probability is not enough for decision-quality forecasting.

Ideal hierarchy:

```text
Parameter uncertainty
        ↓
Match uncertainty
        ↓
Tournament uncertainty
```

---

# 30. Gap 27 — The Current "Favourite Too Strong" Problem Is a Model Risk, Not Just a Cosmetic Issue

The document says Arsenal is around 20.4% to win and may be above a typical bookmaker range.

This should be treated as a concrete model-risk investigation.

Possible causes include:

```text
parameter certainty
model misspecification
correlated simulation assumptions
insufficient squad uncertainty
home/away interaction
rating transfer assumptions
league-strength assumptions
```

The project should test each hypothesis separately.

---

# 31. Recommended Priority Roadmap

## P0 — Must Fix Before Calling the System Robust

```text
1. Parameter uncertainty
2. Match-level forecasting API/interface
3. Player/squad information architecture
4. Live prediction architecture
5. Explicit as-of-time semantics
```

## P1 — Required Before Production

```text
6. Source reconciliation contract
7. Model/data versioning
8. Complete UEFA tiebreaker engine
9. Stronger calibration evaluation
10. Data freshness monitoring
11. Live event idempotency/order handling
12. Retraining policy
13. Product/API architecture
```

## P2 — Important Research Improvements

```text
14. League-strength sensitivity
15. Rating staleness/uncertainty
16. Statistical significance
17. Penalty model
18. Drift monitoring
19. Experiment tracking
20. Feature ablation studies
21. Market benchmark framework
22. Formal data contracts
```

---

# 32. Recommended Target Architecture

The mature PitchIQ system should become:

```text
                        PITCHIQ
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   DATA PLATFORM      TEAM STRENGTH       RULES ENGINE
        │                  │                  │
 domestic + UEFA      Elo / DC / league     UCL format
 identity             strength              tiebreakers
 validation            uncertainty           draw rules
 provenance
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                  MATCH FORECAST ENGINE
                           │
             ┌─────────────┴─────────────┐
             ↓                           ↓
        PRE-MATCH                    LIVE MODEL
             │                           │
             └─────────────┬─────────────┘
                           ↓
                 TOURNAMENT SIMULATOR
                           │
                           ↓
                  PROBABILITY OUTPUTS
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
           API          Dashboard     Reports
```

---

# 33. What Should NOT Be Done Yet

To prevent overengineering, do not immediately build:

```text
deep neural network
LLM-based football reasoning
huge player embeddings
real-time dashboard
microservice architecture
```

before the statistical core is validated.

The correct sequence is:

```text
correct data
   ↓
correct identity
   ↓
correct leakage control
   ↓
correct uncertainty
   ↓
correct evaluation
   ↓
richer features
   ↓
live system
   ↓
product
```

---

# 34. Final Assessment

## Current maturity

### Research / modelling

**Strong**

The current system has a credible statistical foundation and substantially more mature methodology than a simple UCL win-prediction project.

### Data engineering

**Good, but needs production hardening**

The large-scale ingestion and identity work are strong, but formal provenance, schema contracts, freshness checks, and source-reconciliation rules should become explicit.

### Tournament simulation

**Strong**

The simulator is unusually complete for a first implementation, especially because it models the actual competition structure and performs internal consistency checks.

### Uncertainty

**Main statistical weakness**

Treating fitted team parameters as fixed is the most important modelling weakness already identified by the project itself.

### Player/squad information

**Major predictive-data gap**

The current system intentionally operates without it, but it is likely a major remaining source of forecast error.

### Live forecasting

**Major architecture gap**

The current research document does not yet define the online state, event ingestion, feature updates, or live inference loop required by the original PitchIQ vision.

### Production product

**Not yet complete**

The current repository is a strong research engine; it is not yet a complete production forecasting platform.

---

# 35. Final Recommended Definition of PitchIQ

PitchIQ should ultimately be thought of as:

```text
                    PITCHIQ
                       │
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
 Team Intelligence  Match Forecasting  Tournament Forecasting
       │               │                │
       ├─ strength     ├─ pre-match     ├─ league phase
       ├─ squad        ├─ live          ├─ playoffs
       ├─ players      ├─ scorelines    ├─ knockouts
       └─ uncertainty  └─ probabilities └─ champion
```

The core research model is already in place.

The next stage is to make it **uncertainty-aware, player-aware, time-aware, source-safe, reproducible, and eventually live**.

---

# 36. Immediate Next Action

Do not start another broad data-source hunt yet.

The highest-value next engineering task is:

```text
Canonical master data
        ↓
Lock match-level forecasting interface
        ↓
Introduce parameter uncertainty
        ↓
Re-run 2025/26 historical validation
        ↓
Check whether favourite probabilities flatten
        ↓
Verify calibration
```

Only after that should richer player/xG/live data be added.

