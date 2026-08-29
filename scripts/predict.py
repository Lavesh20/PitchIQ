"""Load the saved model and forecast every 2026/27 league-phase fixture."""

from pathlib import Path

import pandas as pd

from pitchiq import clubs
from pitchiq.config import DATA
from pitchiq.models import store

OUT = DATA / "predictions"

model = store.load()
print("model provenance:")
for key, value in store.describe().items():
    print(f"  {key}: {value}")

squad = pd.read_csv(DATA / "external" / "ucl_2026_27_clubs.csv")
pairs = pd.read_csv(DATA / "external" / "ucl_2026_27_pairings.csv")

key = {club: clubs.resolve(club, cc) for club, cc in zip(squad.club, squad.country)}

rows = []
for home, away in zip(pairs.home, pairs.away):
    grid = model.score_matrix(key[home], key[away])
    outcome = model.predict(key[home], key[away])
    lam, mu = model.rates(key[home], key[away])

    flat = grid.ravel()
    best = int(flat.argmax())

    rows.append(
        {
            "home": home,
            "away": away,
            "p_home": round(outcome["H"], 4),
            "p_draw": round(outcome["D"], 4),
            "p_away": round(outcome["A"], 4),
            "xg_home": round(lam, 3),
            "xg_away": round(mu, 3),
            "likeliest_score": f"{best // grid.shape[1]}-{best % grid.shape[1]}",
            "p_likeliest": round(float(flat[best]), 4),
        }
    )

predictions = pd.DataFrame(rows)

OUT.mkdir(parents=True, exist_ok=True)
path = OUT / "ucl_2026_27_fixtures.csv"
predictions.to_csv(path, index=False)

print(f"\n{len(predictions)} fixtures -> {path}")
print(predictions.head(10).to_string(index=False))
