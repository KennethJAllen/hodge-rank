from __future__ import annotations

import numpy as np
from scipy import sparse

from hodge_rank.graph import ComparisonGraph


def enumerate_triangles(g: ComparisonGraph) -> np.ndarray:
    """Return all 2-simplices of the comparison graph as an (|T|, 3) int array.

    A triple (i, j, k) with i < j < k is a triangle iff all three edges
    (i, j), (i, k), (j, k) are present. Enumeration via sorted-forward-
    adjacency intersection: for each vertex i we list its neighbors j > i,
    and for each pair (j, k) of such neighbors we check whether (j, k)
    is an edge.
    """
    if g.m == 0:
        return np.zeros((0, 3), dtype=np.int64)

    forward: list[list[int]] = [[] for _ in range(g.n)]
    for i, j in g.edges:
        forward[int(i)].append(int(j))
    for nbrs in forward:
        nbrs.sort()

    edge_set = {(int(i), int(j)) for i, j in g.edges}

    tris: list[tuple[int, int, int]] = []
    for i in range(g.n):
        nbrs = forward[i]
        for idx_j in range(len(nbrs)):
            j = nbrs[idx_j]
            for k in nbrs[idx_j + 1 :]:
                if (j, k) in edge_set:
                    tris.append((i, j, k))
    if not tris:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def build_B2(g: ComparisonGraph, triangles: np.ndarray) -> sparse.csr_matrix:
    """Edge-triangle boundary, shape (m, |T|).

    For triangle t = (i, j, k), i<j<k:
        B2[(i,j), t] = +1, B2[(j,k), t] = +1, B2[(i,k), t] = -1.
    """
    if triangles.shape[0] == 0:
        return sparse.csr_matrix((g.m, 0))

    edge_index = {(int(a), int(b)): e for e, (a, b) in enumerate(g.edges)}

    rows = np.empty(3 * triangles.shape[0], dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(rows.shape, dtype=np.float64)

    for t_idx, (i, j, k) in enumerate(triangles):
        i, j, k = int(i), int(j), int(k)
        e_ij = edge_index[(i, j)]
        e_jk = edge_index[(j, k)]
        e_ik = edge_index[(i, k)]
        base = 3 * t_idx
        rows[base] = e_ij
        cols[base] = t_idx
        data[base] = +1.0
        rows[base + 1] = e_jk
        cols[base + 1] = t_idx
        data[base + 1] = +1.0
        rows[base + 2] = e_ik
        cols[base + 2] = t_idx
        data[base + 2] = -1.0

    return sparse.coo_matrix((data, (rows, cols)), shape=(g.m, triangles.shape[0])).tocsr()
