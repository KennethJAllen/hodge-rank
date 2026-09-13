import duckdb
import numpy as np
import pytest

from hodge_rank.comparisons import build_comparison_edges
from hodge_rank.graph import ComparisonGraph, largest_component, load_graph_from_db
from hodge_rank.hodge import decompose


def test_pipeline_recovers_known_scores(sample_db):
    with duckdb.connect(str(sample_db), read_only=True) as con:
        build_comparison_edges(con, "Seattle", "WA")
        g = load_graph_from_db(con)
    d = decompose(g)
    scores = dict(zip(g.vertices, d.s))
    assert scores == pytest.approx(
        {"b_a": 0.90, "b_b": -0.85, "b_c": 0.90, "b_d": -1.35, "b_bakery": 0.40}
    )
    assert d.e_total == pytest.approx(d.e_grad + d.e_curl + d.e_harm)


def test_largest_component_remaps_vertices_and_edges():
    g = ComparisonGraph.from_arrays(
        ["a", "b", "c", "d", "e"],
        np.array([[0, 2], [1, 3], [3, 4]]),
        np.array([9.0, 2.0, 3.0]),
        np.array([1.0, 4.0, 5.0]),
    )
    largest = largest_component(g)
    assert largest.vertices == ["b", "d", "e"]
    np.testing.assert_array_equal(largest.edges, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(largest.y, [2, 3])
    np.testing.assert_array_equal(largest.w, [4, 5])
