"""Import the two Yelp NDJSON files needed for comparisons."""

from __future__ import annotations

from pathlib import Path

import duckdb


def ingest(con: duckdb.DuckDBPyConnection, yelp_dir: Path) -> None:
    business = yelp_dir / "yelp_academic_dataset_business.json"
    review = yelp_dir / "yelp_academic_dataset_review.json"
    for path in (business, review):
        if not path.is_file():
            raise FileNotFoundError(f"Expected Yelp file at {path}")
    con.execute(
        """
        CREATE OR REPLACE TABLE business AS
        SELECT business_id::VARCHAR AS business_id, name::VARCHAR AS name,
               city::VARCHAR AS city, state::VARCHAR AS state,
               categories::VARCHAR AS categories
        FROM read_json_auto(?, format='newline_delimited')
        """,
        [str(business)],
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE review AS
        SELECT review_id::VARCHAR AS review_id, user_id::VARCHAR AS user_id,
               business_id::VARCHAR AS business_id, stars::DOUBLE AS stars,
               date::TIMESTAMP AS review_date
        FROM read_json_auto(?, format='newline_delimited')
        """,
        [str(review)],
    )
