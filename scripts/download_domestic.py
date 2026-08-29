"""Pull the full football-data.co.uk archive, then build the parquet."""

from pitchiq.ingest import football_data_uk as fd

main = fd.download_main(start=1993, end=2026)
print(f"main:  {sum(f.ok for f in main)}/{len(main)} files")

extra = fd.download_extra()
print(f"extra: {sum(f.ok for f in extra)}/{len(extra)} files")

fixtures = fd.download_fixtures()
print(f"fixtures: ok={fixtures.ok} status={fixtures.status}")

df = fd.build()
print(f"\nrows: {len(df):,}")
print(f"seasons: {df['season'].nunique()}  divisions: {df['div'].nunique()}")
print(f"date range: {df['date'].min().date()} -> {df['date'].max().date()}")
