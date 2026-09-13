"""Synthetic-graph Hodge decomposition tests.

Each case pins down a ground-truth split of a known flow y into gradient,
curl, and harmonic components. These are the invariants the solver must
respect — if any of these break, the decomposition is miscomputed.
"""

from __future__ import annotations

import numpy as np
import pytest

from hodge_rank.graph import ComparisonGraph
from hodge_rank.hodge import decompose
from hodge_rank.triangles import enumerate_triangles


def _graph(n: int, edges: list[tuple[int, int]], y: list[float], w: list[float] | None = None):
    edges_arr = np.array(edges, dtype=np.int64)
    assert (edges_arr[:, 0] < edges_arr[:, 1]).all(), "edges must be canonical (i<j)"
    y_arr = np.array(y, dtype=np.float64)
    w_arr = np.ones(len(edges)) if w is None else np.array(w, dtype=np.float64)
    verts = [f"v{i}" for i in range(n)]
    return ComparisonGraph.from_arrays(verts, edges_arr, y_arr, w_arr)


# ---------------------------------------------------------------------------
# 1. Tree / path — any flow is a pure gradient.
# ---------------------------------------------------------------------------
def test_path_is_pure_gradient():
    g = _graph(4, [(0, 1), (1, 2), (2, 3)], [1.0, 2.0, 3.0])
    d = decompose(g)

    assert d.e_grad == pytest.approx(14.0, abs=1e-10)  # 1+4+9
    assert d.e_curl == pytest.approx(0.0, abs=1e-10)
    assert d.e_harm == pytest.approx(0.0, abs=1e-10)
    assert d.rankability == pytest.approx(1.0, abs=1e-10)
    # s is zero-mean on the (single) component.
    assert d.s.sum() == pytest.approx(0.0, abs=1e-10)
    # y_grad reconstructs y.
    np.testing.assert_allclose(d.y_grad, g.y, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Star (4 leaves, 1 hub) — any flow is a pure gradient.
# ---------------------------------------------------------------------------
def test_star_is_pure_gradient():
    # hub is 0, leaves are 1..4.
    g = _graph(5, [(0, 1), (0, 2), (0, 3), (0, 4)], [0.5, -1.0, 2.0, 0.25])
    d = decompose(g)
    assert d.e_curl == pytest.approx(0.0, abs=1e-10)
    assert d.e_harm == pytest.approx(0.0, abs=1e-10)
    assert d.rankability == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. Filled triangle — unit circulation is pure curl.
# ---------------------------------------------------------------------------
def test_triangle_with_circulation_is_pure_curl():
    # Edges in canonical order: (0,1), (0,2), (1,2).
    # Going 0→1→2→0 adds +y_{01} +y_{12} −y_{02}; for unit circulation
    # choose y_{01}=1, y_{12}=1, y_{02}=−1 ⇒ circulation = 3.
    g = _graph(3, [(0, 1), (0, 2), (1, 2)], [1.0, -1.0, 1.0])
    tris = enumerate_triangles(g)
    assert tris.shape == (1, 3)

    d = decompose(g, tris)
    assert d.e_grad == pytest.approx(0.0, abs=1e-10)
    assert d.e_harm == pytest.approx(0.0, abs=1e-10)
    assert d.e_curl == pytest.approx(3.0, abs=1e-10)
    assert d.rankability == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 4. Empty triangle (no 2-simplex) — unit circulation is pure harmonic.
# ---------------------------------------------------------------------------
def test_triangle_without_fill_is_pure_harmonic():
    g = _graph(3, [(0, 1), (0, 2), (1, 2)], [1.0, -1.0, 1.0])
    # Explicitly pass zero triangles to skip the 2-simplex.
    empty_tris = np.zeros((0, 3), dtype=np.int64)
    d = decompose(g, empty_tris)
    assert d.e_grad == pytest.approx(0.0, abs=1e-10)
    assert d.e_curl == pytest.approx(0.0, abs=1e-10)
    assert d.e_harm == pytest.approx(3.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 5. Square (4-cycle, no triangles) — circulating flow is pure harmonic.
# ---------------------------------------------------------------------------
def test_square_circulation_is_pure_harmonic():
    # Vertices arranged 0–1–3–2–0. Edges in canonical order:
    #   (0,1), (0,2), (1,3), (2,3).
    # Traversing 0→1→3→2→0 contributes +y01 +y13 −y23 −y02; pick
    # y01=1, y13=1, y23=−1, y02=−1 ⇒ circulation = 4.
    g = _graph(4, [(0, 1), (0, 2), (1, 3), (2, 3)], [1.0, -1.0, 1.0, -1.0])
    tris = enumerate_triangles(g)
    assert tris.shape == (0, 3)

    d = decompose(g, tris)
    assert d.e_grad == pytest.approx(0.0, abs=1e-10)
    assert d.e_curl == pytest.approx(0.0, abs=1e-10)
    assert d.e_harm == pytest.approx(4.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 6. Mixed: known grad + curl on a filled triangle.
# ---------------------------------------------------------------------------
def test_mixed_grad_plus_curl_on_triangle():
    # Filled triangle + a pendant edge to vertex 3.
    # Edges (canonical): (0,1), (0,2), (0,3), (1,2).
    # Construct y = B1^T s_true + B2 φ_true; recover (s, φ) up to gauge.
    edges = [(0, 1), (0, 2), (0, 3), (1, 2)]
    s_true = np.array([-1.0, 0.5, -0.5, 1.0])  # zero-mean
    # Only the (0,1,2) triangle exists.
    phi_true = np.array([0.7])

    # Hand-compute y.
    #   B1^T s per edge:
    #     (0,1): s1-s0 = 1.5
    #     (0,2): s2-s0 = 0.5
    #     (0,3): s3-s0 = 2.0
    #     (1,2): s2-s1 = -1.0
    #   B2 φ per edge (triangle (0,1,2)): edge (0,1)+(1,2)−(0,2), all ·φ:
    #     (0,1): +0.7
    #     (0,2): -0.7
    #     (0,3):  0.0
    #     (1,2): +0.7
    y = np.array([1.5 + 0.7, 0.5 - 0.7, 2.0, -1.0 + 0.7])

    g = _graph(4, edges, y.tolist())
    tris = enumerate_triangles(g)
    assert tris.shape == (1, 3)

    d = decompose(g, tris)

    # Energies add up and the harmonic part vanishes.
    assert d.e_harm == pytest.approx(0.0, abs=1e-10)
    assert d.e_total == pytest.approx(d.e_grad + d.e_curl + d.e_harm, abs=1e-10)

    # Gradient recovers s_true up to global constant.
    np.testing.assert_allclose(d.s - d.s.mean(), s_true - s_true.mean(), atol=1e-8)
    # Curl recovers phi_true exactly (triangle rank is 1).
    np.testing.assert_allclose(d.phi, phi_true, atol=1e-8)


# ---------------------------------------------------------------------------
# 7. Pythagoras: energies always sum to E_total (random weighted graph).
# ---------------------------------------------------------------------------
def test_energy_conservation_random():
    rng = np.random.default_rng(42)
    n = 10
    # Erdős–Rényi style edges.
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.4]
    edges = pairs
    y = rng.normal(size=len(edges))
    w = rng.uniform(1, 5, size=len(edges))
    g = _graph(n, edges, y.tolist(), w.tolist())

    d = decompose(g)
    assert d.e_total == pytest.approx(d.e_grad + d.e_curl + d.e_harm, rel=1e-8)


# ---------------------------------------------------------------------------
# 8. Weight-invariance sanity: scaling w by a constant leaves ratios alone.
# ---------------------------------------------------------------------------
def test_rankability_invariant_under_uniform_weight_scale():
    rng = np.random.default_rng(7)
    n = 8
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
    y = rng.normal(size=len(pairs))
    w = rng.uniform(1, 3, size=len(pairs))

    g1 = _graph(n, pairs, y.tolist(), w.tolist())
    g2 = _graph(n, pairs, y.tolist(), (w * 10.0).tolist())

    d1 = decompose(g1)
    d2 = decompose(g2)
    assert d1.rankability == pytest.approx(d2.rankability, rel=1e-8)


# ---------------------------------------------------------------------------
# 9. Triangle enumeration correctness.
# ---------------------------------------------------------------------------
def test_triangle_enumeration_K4():
    # K4 has C(4,3) = 4 triangles.
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    g = _graph(4, edges, [1.0] * len(edges))
    tris = enumerate_triangles(g)
    assert tris.shape == (4, 3)
    # All are canonical (i<j<k).
    assert ((tris[:, 0] < tris[:, 1]) & (tris[:, 1] < tris[:, 2])).all()


def test_weighted_filled_triangle_has_no_harmonic_residual():
    g = _graph(3, [(0, 1), (0, 2), (1, 2)], [1, -1, 1], [1, 2, 3])
    d = decompose(g)
    assert d.harmonic_fraction == pytest.approx(0, abs=1e-12)
    assert d.curl_fraction == pytest.approx(9 / 11, abs=1e-12)


def test_weighted_hodge_components_are_orthogonal_and_harmonic():
    from hodge_rank.graph import build_B1
    from hodge_rank.triangles import build_B2

    # A filled triangle joined to an unfilled square, with unequal weights.
    g = _graph(
        6,
        [(0, 1), (0, 2), (1, 2), (2, 3), (2, 5), (3, 4), (4, 5)],
        [2, -1, 3, 1, -2, 2, 1],
        [1, 2, 3, 4, 5, 6, 7],
    )
    tris = enumerate_triangles(g)
    d = decompose(g, tris)
    for a, b in [(d.y_grad, d.y_curl), (d.y_grad, d.y_harm), (d.y_curl, d.y_harm)]:
        assert np.dot(g.w * a, b) == pytest.approx(0, abs=1e-9)
    np.testing.assert_allclose(build_B1(g) @ (g.w * d.y_harm), 0, atol=1e-9)
    np.testing.assert_allclose(build_B2(g, tris).T @ d.y_harm, 0, atol=1e-9)
    np.testing.assert_allclose(d.y_grad + d.y_curl + d.y_harm, g.y, atol=1e-10)
    assert d.e_harm > 0.01


def test_zero_weight_edge_cannot_fill_triangle():
    g = _graph(3, [(0, 1), (0, 2), (1, 2)], [1, -1, 1], [1, 0, 3])
    d = decompose(g)
    assert d.rankability == pytest.approx(1, abs=1e-12)
    assert d.e_curl == pytest.approx(0, abs=1e-12)
    assert d.e_harm == pytest.approx(0, abs=1e-12)
