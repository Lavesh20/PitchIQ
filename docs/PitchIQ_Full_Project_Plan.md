# PitchIQ — UEFA Champions League Prediction Engine

## 1. Project Overview

**Project:** PitchIQ  
**V1 domain:** UEFA Champions League only  
**Initial objective:** Build our own machine-learning system that predicts Home Win / Draw / Away Win probabilities for UEFA Champions League matches.

Long-term objective: evolve PitchIQ into a live football intelligence system that continuously updates match probabilities using historical data, team strength, form, player context, match events, and live state.

```text
Historical Data
      ↓
Data Quality
      ↓
Feature Engineering
      ↓
Pre-Match Model
      ↓
Backtesting
      ↓
Calibration
      ↓
Live Data
      ↓
Live Prediction Model
      ↓
Football Analytics Platform
```

---

## 2. Scope Decision

### V1: Champions League only

We intentionally narrowed the project to UEFA Champions League first.

Reasons:

- Controlled domain
- Consistent competition context
- Enough historical matches
- Easier validation
- Easier identification of data-quality issues before scaling

After UCL is stable, the architecture can expand to domestic leagues and other UEFA competitions.

---

## 3. First Prediction Target

The first model predicts:

```text
P(Home Win)
P(Draw)
P(Away Win)
```

Example:

```text
Real Madrid vs Bayern Munich

Home: 52%
Draw: 25%
Away: 23%
```

The output should be probabilities, not just a single class.

Later, the system can predict expected goals, exact score, BTTS, totals, next goal, qualification probability, and live win probability.

---

# 4. Data Strategy

We discussed two approaches.

## One provider

A single provider is operationally simpler when it provides:

- historical matches
- current fixtures
- events
- statistics
- xG
- lineups
- player information
- live data

Sportmonks was identified as a strong future candidate for a richer all-in-one role.

## Multiple sources

For historical data, we chose a more disciplined version of the multi-source approach:

> For each season, compare the available sources and choose the better source for that particular season.

Multiple sources are **not** treated as independent training examples.

For the same match:

```text
Source A ──┐
           ├── compare ──> choose best record
Source B ──┘
```

This avoids double-counting while allowing us to use the strongest available source.

---

# 5. Sources Investigated

## 5.1 football-data.org

We successfully authenticated using:

```text
X-Auth-Token
```

Successful API endpoints used:

```text
/v4/competitions/CL
/v4/competitions/CL/matches
/v4/matches/{match_id}
```

### Data observed

Match-level records include:

- competition
- season
- match ID
- date/time
- status
- matchday
- stage
- home team
- away team
- full-time score
- half-time score
- referees

Example conceptual record:

```text
match_id
utcDate
status
matchday
stage
homeTeam
awayTeam
score
referees
```

### Current access result

With the current token:

```text
2023/24 → 125 matches
2024/25 → 189 matches
2025/26 → 189 matches
```

Total:

```text
503 matches
```

Older seasons attempted through the API were restricted by the current subscription permissions.

The API response also provides throttling-related headers; the eventual client should inspect them and respect rate limits.

### Limitation

Our current access does not provide the rich feature set we ultimately want, such as full xG/event/player/statistics coverage.

---

# 6. OpenFootball

OpenFootball was investigated as a historical UCL source.

Dedicated season files contain:

- season information
- dates
- groups/stages
- kickoff times
- teams
- final scores
- half-time scores

Example source structure:

```text
Wed Sep 7 2022
18:45 AFC Ajax (NED) v Rangers FC (SCO) 4-0 (3-0)
21:00 SSC Napoli (ITA) v Liverpool FC (ENG) 4-1 (3-0)
```

We successfully downloaded:

```text
2011/12
2012/13
2013/14
2014/15
2015/16
2016/17
2017/18
2018/19
2019/20
2020/21
2021/22
2022/23
2023/24
```

The source also contains recent UCL seasons such as 2024/25 and 2025/26.

### OpenFootball limitation

OpenFootball is useful for historical/completed match data, but it is not our intended live minute-by-minute feed.

Therefore:

```text
Historical completed matches → useful
Live event stream            → not our source
Live xG                      → not our source
```

---

# 7. Kaggle Dataset Investigated

We investigated the Kaggle dataset:

**Champions League Dataset 1955–2023**

Conclusion:

- Useful as a reference/sanity-check source
- Not sufficient as the core ML training dataset

Its primary value is historical context and aggregated information rather than the rich match-level feature space required by PitchIQ.

---

# 8. StatsBomb Open Data Investigated

StatsBomb Open Data was identified as a potentially valuable rich-event source.

Potential information includes:

- shots
- passes
- carries
- pressures
- duels
- locations
- lineups
- substitutions
- shot-derived xG

Limitation:

- selected competition/season coverage
- not a guaranteed continuous current/live UCL feed

Best role:

```text
rich historical/research data
```

rather than the primary live provider.

---

# 9. Current Historical Data

## OpenFootball parser result

The reusable parser successfully produced:

```text
2011/12    121
2012/13    125
2013/14    124
2014/15    123
2015/16    122
2016/17    124
2017/18    125
2018/19    124
2019/20    118
2020/21    124
2021/22    123
2022/23    125
2023/24    122
```

Combined parsed total:

```text
1,600 matches
```

The dataset contains:

```text
season
date
stage
group
home_team
away_team
home_goals
away_goals
ht_home_goals
ht_away_goals
```

Observed data-quality output:

```text
Missing season:             0
Missing date:               0
Missing home_team:          0
Missing away_team:          0
Missing final scores:       0
Missing group:            520
Missing half-time scores:  92
```

`group` being blank for knockout matches is expected.

Some matches have no recorded half-time score in the OpenFootball source.

---

# 10. Historical Data Inventory

The current project therefore has:

```text
OpenFootball
2011/12 → 2023/24
1,600 parsed records
```

and:

```text
football-data.org
2023/24 → 2025/26
503 API records
```

There is overlap in 2023/24, which is intentional because we want to compare and select the better source rather than blindly merging both.

---

# 11. Source Comparison Work

We attempted to compare the two 2023/24 datasets.

Initial observed counts:

```text
football-data.org → 125
OpenFootball      → 122
```

The first comparison implementation produced an invalid result:

```text
Matched matches: 78
```

The problem was the team identity logic.

OpenFootball strings such as:

```text
Barcelona (ESP)
```

were converted to:

```text
ESP
```

while football-data.org uses a team TLA such as:

```text
FCB
```

`ESP` is a country code, not the team identifier.

As a result, almost every match appeared different.

The comparison that reported:

```text
Matched matches: 0
Result: OpenFootball has better coverage
```

must therefore be considered invalid.

### Lesson

> Do not use country codes or raw string similarity as the final team identity mechanism.

The source comparison is still unfinished.

---

# 12. Canonical Team Identity

PitchIQ needs its own team identity layer.

Concept:

```text
Source team name
       ↓
Canonical PitchIQ team ID
       ↓
Master team
```

Examples:

```text
Inter
FC Internazionale Milano
```

must resolve to one canonical club.

Likewise:

```text
Barcelona
FC Barcelona
```

must resolve to one canonical club.

Important distinction:

```text
Sport Lisboa e Benfica
Sporting Clube de Portugal
Sporting Clube de Braga
```

are three different clubs and must never be collapsed.

### Key rule

Fuzzy matching may generate suggestions, but it must never automatically establish final identity.

---

# 13. Data Architecture

Recommended structure:

```text
PitchIQ/
│
├── data/
│   ├── raw/
│   │   ├── football_data/
│   │   └── openfootball/
│   │
│   └── processed/
│
├── src/
├── scripts/
├── notebooks/
├── models/
├── reports/
├── tests/
└── .env
```

Raw data is preserved as received from the source.

Example:

```text
data/raw/openfootball/ucl_2022_23.txt
data/raw/football_data/ucl_2025_26.json
```

Processed data is generated from raw data.

---

# 14. Work Completed So Far

We have already completed:

### Environment

- Python virtual environment created
- Initial dependencies installed
- API token stored in `.env`

### API connectivity

- football-data.org authentication verified
- Champions League endpoint verified
- match-list endpoint verified
- individual match endpoint verified

### Current API ingestion

- 2023/24 downloaded
- 2024/25 downloaded
- 2025/26 downloaded

### Historical ingestion

- OpenFootball 2011/12 → 2023/24 downloaded
- Historical parser implemented
- 1,600 records parsed

### Initial source comparison

- Comparison script created
- First comparison exposed identity-normalization bug
- Bug identified

---

# 15. Recommended Master Match Schema

The final source-independent match table should look like:

```text
pitchiq_match_id

season
competition

date
stage
round
matchday
group

home_team_id
away_team_id

home_team_name
away_team_name

home_score
away_score

home_ht_score
away_ht_score

result

source
source_match_id
```

This lets the model use one canonical structure regardless of which provider supplied the underlying record.

---

# 16. Phase 1 — Finish Data Acquisition

## Step 1

Finish the 2023/24 source comparison with reliable canonical team identities.

## Step 2

Run the same comparison for every overlapping season.

## Step 3

Create a source quality score per season.

Example:

```text
coverage
date completeness
team completeness
score completeness
duplicate rate
stage completeness
half-time completeness
identity confidence
```

## Step 4

Select the preferred source per season.

Example:

```text
2011/12 → OpenFootball
...
2023/24 → winning source
2024/25 → winning source
2025/26 → winning source
```

## Step 5

Build one master dataset.

Target:

```text
data/processed/ucl_matches_master.csv
```

Exactly one canonical row per match.

---

# 17. Phase 2 — Data Quality

Before modelling, automatically check:

```text
Duplicate match records
Missing team IDs
Missing final scores
Invalid scores
Invalid dates
Unexpected stages
Same team as home and away
Duplicate date/home/away combinations
Season inconsistencies
```

The pipeline should fail loudly when serious validation fails.

---

# 18. Phase 3 — Canonical Team Registry

Create:

```text
team_registry
```

Potential fields:

```text
pitchiq_team_id
canonical_name
country
historical_names
football_data_id
openfootball_names
sportmonks_id
```

This becomes the identity backbone of the entire project.

---

# 19. Phase 4 — First Feature Set

Initially use features derived from historical results only.

## Recent form

```text
last_3_results
last_5_results
last_10_results
```

## Goals

```text
rolling goals scored
rolling goals conceded
goal difference
```

## Home/away performance

```text
home win rate
home goals scored
home goals conceded

away win rate
away goals scored
away goals conceded
```

## Team strength

Implement an Elo-style rating:

```text
home_elo
away_elo
elo_difference
```

## Competition context

```text
stage
matchday
competition_format
```

## Rest

```text
home_rest_days
away_rest_days
```

## Head-to-head

Use only where the sample is meaningful.

---

# 20. Critical Rule — Avoid Data Leakage

For every prediction:

```text
Historical matches before prediction timestamp
        ↓
Feature calculation
        ↓
Prediction
        ↓
Actual match result
```

Never allow the target match or later matches to influence the feature values used to predict that match.

Example:

A prediction made before a 2024 match cannot use a 2024 match played later in the season.

This must be enforced in code.

---

# 21. Competition Format Change

The Champions League changed format beginning in 2024/25.

Therefore the dataset contains different structural eras.

The system must preserve enough information to distinguish:

```text
traditional group-stage era
new league-phase era
knockout phases
```

Do not assume every season has identical competition structure.

---

# 22. Phase 5 — First Models

Start simple.

### Model 1

Elo baseline

### Model 2

Multinomial Logistic Regression

### Model 3

Poisson/Dixon-Coles-style goal model

### Model 4

Gradient boosting:

```text
XGBoost / LightGBM
```

The purpose of the first modelling stage is comparison and baseline establishment, not algorithmic complexity.

---

# 23. Probability Evaluation

Do not optimize only for accuracy.

Evaluate:

```text
Log Loss
Brier Score
Calibration
Accuracy
```

The most important question is:

> Are the probabilities reliable on unseen matches?

Example:

If PitchIQ predicts 70% for a class repeatedly, the observed frequency of that class should be close to 70% over a sufficiently large sample.

---

# 24. Phase 6 — Walk-Forward Backtesting

Do not randomly split football matches.

Use chronological evaluation.

Example:

```text
Train: 2011–2018
Test:  2019

Train: 2011–2019
Test:  2020

Train: 2011–2020
Test:  2021
...
```

Alternative rolling windows can be tested, but the essential property is:

> No future data is available when making a historical prediction.

---

# 25. Recency Handling

A long historical window is useful because it provides more examples.

However, older matches should not automatically have the same influence as recent matches.

Potential approaches:

```text
time-decay weights
rolling windows
recency-aware features
```

The exact method should be validated experimentally.

Do not assume that 2011 and 2025 should contribute identically.

---

# 26. Rich Data Enrichment

Once the result-only baseline is stable, investigate richer features:

```text
xG
shots
shots on target
possession
corners
passes
pressure
events
lineups
substitutions
formations
player statistics
injuries
suspensions
player availability
```

At this stage, evaluate whether a commercial source such as Sportmonks provides enough additional value to justify its cost.

---

# 27. Market Odds

Odds were seen in some source outputs, but the first PitchIQ model should treat market probabilities primarily as a benchmark.

Useful comparison:

```text
PitchIQ probability
Market probability
Actual result
```

Later experiments can compare:

```text
model only
market only
model + market
```

The goal is to determine whether PitchIQ adds predictive information rather than simply reproducing the market.

---

# 28. Phase 7 — Live Prediction

Once the pre-match model is reliable, add a live layer.

Live architecture:

```text
Live Provider
     ↓
Current Match State
     ↓
Feature Update
     ↓
Live Model
     ↓
New Probabilities
     ↓
Repeat
```

Potential live inputs:

```text
minute
score
goals
cards
red cards
substitutions
shots
xG
possession
live events
```

The live model can start from the pre-match probability and update as new evidence arrives.

---

# 29. Eventually: Pre-Match + Live Architecture

```text
                    PITCHIQ
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
      PRE-MATCH                  LIVE STATE
          │                         │
      historical                 score
      team strength              minute
      form                       events
      availability               xG
          │                         │
          └────────────┬────────────┘
                       ↓
                 LIVE PREDICTION
                       ↓
                Probability Stream
```

---

# 30. Future Prediction Targets

After the first 1X2 model is stable:

```text
match winner
expected home goals
expected away goals
exact score
over/under
BTTS
next goal
qualification
live win probability
```

Expand one target at a time.

---

# 31. Suggested Technical Stack

## Initial

```text
Python
Pandas / Polars
Requests
python-dotenv
Scikit-learn
XGBoost / LightGBM
PyArrow
Matplotlib
Jupyter
```

## Later

```text
PostgreSQL
MLflow
FastAPI
Redis
Docker
```

Potential live architecture:

```text
Live API
   ↓
Ingestion
   ↓
Queue / Redis
   ↓
Feature updater
   ↓
Prediction service
   ↓
API / dashboard
```

---

# 32. Testing Strategy

Create tests for:

## Data

```text
No duplicate canonical matches
No null team IDs
Final scores valid
Dates valid
Season valid
```

## Features

```text
No future match included
Rolling windows use only prior matches
Prediction timestamp respected
```

## Models

```text
Probabilities sum to 1
No NaN predictions
Expected schema
Reproducible training
```

## Live

```text
Duplicate events
Out-of-order events
Goal updates
Red-card updates
Substitution updates
Match-end handling
```

---

# 33. Engineering Principles

### Preserve raw data

Never overwrite raw source data.

### Normalize once

Build reusable identity and schema normalization layers.

### Keep provenance

Every canonical match should retain its original source and source record ID.

### One match = one training observation

Never duplicate a match because multiple providers contain it.

### Time-aware modelling

Football prediction is fundamentally temporal.

### Validate before optimizing

Data correctness comes before model sophistication.

### Baselines first

Always compare complex ML against simple strong baselines.

---

# 34. Immediate Next Move

The current project is blocked only by the unfinished source comparison.

The next task is:

```text
2023/24 OpenFootball
          +
2023/24 football-data.org
          ↓
correct canonical team mapping
          ↓
match-by-match comparison
          ↓
validate:
  count
  identity
  dates
  final scores
  half-time scores
  stage
          ↓
choose source
```

Once that works:

```text
run for all overlapping seasons
        ↓
source selection per season
        ↓
master UCL dataset
        ↓
data validation
        ↓
feature engineering
        ↓
first baseline model
```

---

# 35. Definition of Success for V1

PitchIQ V1 is successful when we have:

```text
Reliable UCL historical dataset
        ↓
Canonical team identities
        ↓
No future leakage
        ↓
Reproducible feature pipeline
        ↓
Chronological backtesting
        ↓
Calibrated probabilities
        ↓
Competitive baseline performance
        ↓
Pre-match UCL predictions
```

The goal is not to overfit historical results.

The goal is to build a forecasting system whose probability estimates remain credible on genuinely unseen matches.

---

# 36. Full Roadmap

```text
PHASE 1  — Data Discovery
        ✅ Completed

PHASE 2  — Source Investigation
        ✅ Mostly completed

PHASE 3  — Historical Ingestion
        ✅ Started / largely completed

PHASE 4  — Source Comparison
        🔄 CURRENT

PHASE 5  — Canonical Master Dataset
        ⏳

PHASE 6  — Data Validation
        ⏳

PHASE 7  — Feature Engineering
        ⏳

PHASE 8  — Baseline Models
        ⏳

PHASE 9  — Walk-Forward Backtesting
        ⏳

PHASE 10 — Probability Calibration
        ⏳

PHASE 11 — Rich Data Enrichment
        ⏳

PHASE 12 — Live Data Pipeline
        ⏳

PHASE 13 — Live Prediction Model
        ⏳

PHASE 14 — API / Dashboard / Product
        ⏳
```

---

# 37. Current State in One Sentence

**PitchIQ currently has a working Champions League API connection, recent UCL data from football-data.org, 2011/12–2023/24 historical data from OpenFootball, a working historical parser, and an identified but unfinished source-comparison layer; the next priority is to fix canonical team identity and complete the season-by-season source selection before starting ML.**
