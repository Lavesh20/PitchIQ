"""Mirror openfootball's UEFA club archive, then build the parquet."""

from pitchiq.ingest import openfootball as of

results = of.download()

ok = sum(results.values())
print(f"files: {ok}/{len(results)}")

for (stem, slug), got in sorted(results.items()):
    if not got:
        print(f"  MISSING {stem} {slug}")

df = of.build()

print(f"\nrows: {len(df):,}")
print(f"range: {df['date'].min().date()} -> {df['date'].max().date()}")
print("\nby competition:")
print(df.groupby("competition").size().to_string())
print("\nby season:")
print(df.groupby("season").size().to_string())
print("\nstages:")
print(df["stage"].value_counts().head(20).to_string())
