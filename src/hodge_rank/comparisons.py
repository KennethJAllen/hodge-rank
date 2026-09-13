"""Turn a city's reviews into one weighted comparison per shop pair."""

from __future__ import annotations

import duckdb


def build_comparison_edges(
    con: duckdb.DuckDBPyConnection,
    city: str,
    state: str,
    min_coraters: int = 3,
) -> None:
    """Create temporary city_business and comparison_edge tables.

    Keep each user's latest review per shop, then require three distinct local
    coffee shops per reviewer. Every reviewer contributes once per pair.
    Positive y_ij means j is preferred to i. User means cancel in differences.
    Temporary tables also work on a read-only source database.
    """
    if min_coraters < 1:
        raise ValueError("min_coraters must be positive")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE city_business AS
        SELECT business_id, name FROM business
        WHERE UPPER(TRIM(city)) = UPPER(TRIM(?))
          AND UPPER(TRIM(state)) = UPPER(TRIM(?))
          AND categories LIKE '%Coffee & Tea%'
        """,
        [city, state],
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE comparison_edge AS
        WITH latest AS (
            SELECT r.user_id, r.business_id, r.stars
            FROM review r JOIN city_business b USING (business_id)
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY r.user_id, r.business_id
                ORDER BY r.review_date DESC, r.review_id DESC
            ) = 1
        ), active AS (
            SELECT * FROM latest
            QUALIFY COUNT(*) OVER (PARTITION BY user_id) >= 3
        )
        SELECT a.business_id AS i, b.business_id AS j,
               AVG(b.stars - a.stars)::DOUBLE AS y_ij,
               COUNT(*)::DOUBLE AS w_ij
        FROM active a JOIN active b
          ON a.user_id = b.user_id AND a.business_id < b.business_id
        GROUP BY a.business_id, b.business_id
        HAVING COUNT(*) >= ?
        """,
        [min_coraters],
    )
