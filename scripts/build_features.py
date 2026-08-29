"""Rebuild the feature table from the match stream and cache it.

Takes about forty seconds over the full record. Everything downstream
reads the parquet rather than repeating the pass.
"""

from pitchiq.features import build as features

frame = features.build()
features.save(frame)

print(f"{len(frame):,} matches, {len(features.feature_columns(frame))} features")
print(f"{frame.date.min().date()} to {frame.date.max().date()}")
print()
print(features.coverage(frame).to_string(index=False))
