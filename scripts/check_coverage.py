"""How much of the UCL universe joins to domestic-league data?

The check that matters is the split check: a UCL club whose key does not
appear among the domestic keys for its own country is a missing alias,
and would silently become a second, ratingless club.
"""

import pandas as pd

from pitchiq import clubs

ucl = pd.read_parquet("data/processed/uefa_matches.parquet")
dom = pd.read_parquet("data/processed/domestic_matches.parquet")

ucl_keys = {}
for col, ccol in [("home_team", "home_country"), ("away_team", "away_country")]:
    for name, country in zip(ucl[col], ucl[ccol]):
        ucl_keys[(name, country)] = clubs.resolve(name, country)

dom_by_country: dict[str, set[str]] = {}
for country, grp in dom.groupby("country"):
    code = clubs.country_code(country)
    names = set(grp.home_team.dropna()) | set(grp.away_team.dropna())
    dom_by_country.setdefault(code, set()).update(
        clubs.resolve(n, country) for n in names
    )

covered = set(clubs.COUNTRY_NAMES)

splits, joined, no_league = [], 0, 0
for (name, country), key in ucl_keys.items():
    if country not in covered:
        no_league += 1
        continue
    code = clubs.country_code(country)
    if key in dom_by_country.get(code, set()):
        joined += 1
    else:
        splits.append((country, name, key))

print("=== UCL club names vs domestic data ===")
print(f"join cleanly            : {joined}")
print(f"no domestic league      : {no_league}")
print(f"SPLIT (missing alias)   : {len(splits)}")
for country, name, key in sorted(splits):
    print(f"    [{country}] {name!r} -> {key!r}")

# Match-level coverage.
hk = [clubs.resolve(n, c) for n, c in zip(ucl.home_team, ucl.home_country)]
ak = [clubs.resolve(n, c) for n, c in zip(ucl.away_team, ucl.away_country)]
ucl = ucl.assign(home_key=hk, away_key=ak)

all_dom = set().union(*dom_by_country.values())
both = ucl.home_key.isin(all_dom) & ucl.away_key.isin(all_dom)
print(f"\nUCL matches with both sides in domestic data: {both.sum():,}/{len(ucl):,} ({both.mean():.1%})")

# The 36 we must predict.
squad = pd.read_csv("data/external/ucl_2026_27_clubs.csv")
squad["key"] = [clubs.resolve(n, c) for n, c in zip(squad.club, squad.country)]
squad["has_domestic"] = squad.key.isin(all_dom)

print(f"\n=== UCL 2026/27 squad of {len(squad)} ===")
print(f"with domestic history: {squad.has_domestic.sum()}")
missing = squad.loc[~squad.has_domestic, ["club", "country", "pot", "key"]]
if len(missing):
    print("without:")
    print(missing.to_string(index=False))
