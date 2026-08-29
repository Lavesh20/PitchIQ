"""Train the goals model on every match on disk, and save it."""

import time

from pitchiq import matches
from pitchiq.models import dixon_coles as dc
from pitchiq.models import store

frame = matches.load()

print(f"training on {len(frame):,} matches "
      f"({frame.date.min().date()} to {frame.date.max().date()})")

started = time.time()
model = dc.fit(
    frame,
    dc.DixonColesConfig(xi=0.0010, ridge=0.5, home_advantage_by=None),
)
elapsed = time.time() - started

path = store.save(
    model,
    matches=frame,
    note="Dixon-Coles fitted on all domestic and UEFA matches",
)

print(f"  converged={model.converged}  in {elapsed:.1f}s")
print(f"  home advantage {model.home_advantage:.4f}   rho {model.rho:+.4f}")
print(f"  {len(model.attack):,} clubs, "
      f"{len(model.attack) * 2 + 2:,} parameters")
print(f"\nsaved -> {path}  ({path.stat().st_size / 1024:.0f} KB)")
