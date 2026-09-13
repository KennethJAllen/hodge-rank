"""Import Yelp reviews and explore a city's coffee-shop comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np

from hodge_rank import __version__
from hodge_rank.comparisons import build_comparison_edges
from hodge_rank.graph import component_labels, largest_component, load_graph_from_db
from hodge_rank.hodge import decompose
from hodge_rank.ingest import ingest


def analyze(db: Path, city: str, state: str, min_coraters: int, top: int) -> None:
    with duckdb.connect(str(db), read_only=True) as con:
        build_comparison_edges(con, city, state, min_coraters)
        g = load_graph_from_db(con)
        names = dict(con.execute("SELECT business_id, name FROM city_business").fetchall())
    if not g.m:
        print("No comparisons meet the filters. Try a lower --min-coraters value.")
        return
    n_components, _, _ = component_labels(g)
    total_shops = g.n
    g = largest_component(g)
    d = decompose(g)
    print(f"{city}, {state}: {g.n} shops, {g.m} comparisons")
    if n_components > 1:
        print(f"Using the largest of {n_components} components ({g.n}/{total_shops} linked shops).")
    if d.e_total > 0:
        print(
            f"Energy: ranking {d.rankability:.1%}  "
            f"local cycles {d.curl_fraction:.1%}  "
            f"global cycles {d.harmonic_fraction:.1%}"
        )
    else:
        print("All pairwise mean differences are zero; there is no ranking signal.")
    print("\nRanking (scores are relative to this group):")
    for v in np.argsort(-d.s, kind="stable")[:top]:
        print(f"  {d.s[v]:+6.3f}  {names[g.vertices[v]]} [{g.vertices[v]}]")
    residual = g.y - d.y_grad
    print("\nLargest disagreements with the ranking (positive favors the second shop):")
    for e in np.argsort(-np.abs(residual), kind="stable")[: min(5, g.m)]:
        if abs(residual[e]) < 1e-8:
            break
        a, b = g.edges[e]
        print(
            f"  {names[g.vertices[a]]} -> {names[g.vertices[b]]}: "
            f"observed {g.y[e]:+.2f}, ranking predicts {d.y_grad[e]:+.2f} "
            f"({int(g.w[e])} co-raters)"
        )


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    import_cmd = commands.add_parser("ingest", help="Import Yelp business and review NDJSON")
    import_cmd.add_argument("--yelp-dir", type=Path, default=Path("data/yelp"))
    import_cmd.add_argument("--db", type=Path, default=Path("data/hodge_rank.duckdb"))
    analyze_cmd = commands.add_parser(
        "analyze", help="Print a city ranking and disagreement summary"
    )
    analyze_cmd.add_argument("--db", type=Path, default=Path("data/hodge_rank.duckdb"))
    analyze_cmd.add_argument("--city", required=True)
    analyze_cmd.add_argument("--state", required=True)
    analyze_cmd.add_argument("--min-coraters", type=positive_int, default=3)
    analyze_cmd.add_argument("--top", type=positive_int, default=10)
    args = parser.parse_args(argv)
    if args.command == "ingest":
        args.db.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(args.db)) as con:
            ingest(con, args.yelp_dir)
        print(f"Imported reviews into {args.db}")
    else:
        analyze(args.db, args.city, args.state, args.min_coraters, args.top)


if __name__ == "__main__":
    main()
