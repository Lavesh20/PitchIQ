"""Resample the record to find out how firmly each rating is pinned down.

Slow by design: two hundred full refits, no shortcuts on convergence.
It runs once and the samples are cached.
"""

import sys

import pandas as pd

from pitchiq import matches
from pitchiq.config import DATA
from pitchiq.models import dixon_coles as dc
from pitchiq.models import uncertainty

CUTOFF = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2026-07-01")
DRAWS = int(sys.argv[2]) if len(sys.argv) > 2 else 200

frame = matches.load()
frame = frame[frame.date < CUTOFF]

config = dc.DixonColesConfig(xi=0.0010, ridge=0.5)
print(f"{len(frame):,} matches to {CUTOFF.date()}, {DRAWS} resamples", flush=True)

samples = uncertainty.bootstrap(frame, config, draws=DRAWS)

path = DATA / "models" / f"uncertainty_{CUTOFF.date()}.npz"
path.parent.mkdir(parents=True, exist_ok=True)
samples.save(path)

print(f"\nwritten to {path}")
print("\n=== least certain ratings ===")
print(samples.spread(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n=== most certain ===")
print(samples.spread().tail(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
