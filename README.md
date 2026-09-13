# hodge-rank

A small experiment: turn shared reviewers' coffee-shop ratings into a graph,
then split the comparisons into a ranking and disagreements that it cannot fit.

## Run

Install dependencies with `uv sync`. On NixOS, enter `nix develop` first.
Put the Yelp Open Dataset business and review NDJSON files in `data/yelp/`, then:

```sh
uv run hodge-rank ingest
uv run hodge-rank analyze --city "YOUR CITY" --state ST
```

If you already imported the data, skip `ingest`. Analysis reads the raw tables
without changing the database or saving results.

Use `--min-coraters 5` to require more shared reviewers, `--top 20` to show more
shops, or `--db path/to/data.duckdb` to use another database.

## How it works

Keep each person's latest review of each shop, and include reviewers of at least
three distinct local shops in Yelp's `Coffee & Tea` category. For each pair, take
the average rating difference and weight it by the number of shared reviewers.

The weighted Hodge decomposition splits that graph into:

- **Ranking:** differences explained by one score per shop.
- **Local cycles:** disagreement around triangles.
- **Global cycles:** the remaining harmonic component.

The command prints the ranking, the energy split, and a few pairs that disagree
with the ranking. It uses the largest connected group so the displayed scores
are comparable. The energy split describes the observed graph; it does not
establish taste groups or measure predictive accuracy.

The implementation uses DuckDB, NumPy, and SciPy. Run the tests with `uv run pytest`.
