import duckdb
import pytest

from hodge_rank.comparisons import build_comparison_edges


def test_comparisons_from_distinct_active_reviewers(sample_db):
    with duckdb.connect(str(sample_db), read_only=True) as con:
        build_comparison_edges(con, " seattle ", "wa")
        rows = con.execute("SELECT i, j, y_ij, w_ij FROM comparison_edge").fetchall()
        assert len(rows) == 10
        assert all(i < j and w == 4 for i, j, _, w in rows)
        means = {(i, j): y for i, j, y, _ in rows}
        assert means["b_a", "b_b"] == pytest.approx(-1.75)
        assert means["b_a", "b_c"] == pytest.approx(0)
        build_comparison_edges(con, "Seattle", "WA", min_coraters=5)
        assert con.execute("SELECT COUNT(*) FROM comparison_edge").fetchone()[0] == 0
        build_comparison_edges(con, "Seattle", "OTHER")
        assert con.execute("SELECT COUNT(*) FROM comparison_edge").fetchone()[0] == 0


def test_repeated_reviews_count_once_and_latest_wins(sample_db):
    with duckdb.connect(str(sample_db)) as con:
        con.execute("INSERT INTO review VALUES ('new', 'u_active1', 'b_a', 1, '2024-01-01')")
        con.execute("INSERT INTO review VALUES ('again1', 'u_inactive1', 'b_a', 2, '2024-01-01')")
        con.execute("INSERT INTO review VALUES ('again2', 'u_inactive1', 'b_a', 3, '2024-01-02')")
        build_comparison_edges(con, "Seattle", "WA")
        y, w = con.execute(
            "SELECT y_ij, w_ij FROM comparison_edge WHERE i='b_a' AND j='b_b'"
        ).fetchone()
        assert w == 4
        assert y == pytest.approx(-0.75)
