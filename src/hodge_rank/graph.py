from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import duckdb
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


@dataclass
class ComparisonGraph:
    """Immutable comparison graph.

    Attributes
    ----------
    vertices : list of business_id strings, length n.
    vertex_index : business_id -> row index.
    edges : (m, 2) int array; each row is (i, j) with i < j (vertex indices).
    y : (m,) float array of edge flows.
    w : (m,) float array of edge weights (co-rater counts).
    """

    vertices: list[str]
    vertex_index: dict[str, int] = field(repr=False)
    edges: np.ndarray
    y: np.ndarray
    w: np.ndarray

    @property
    def n(self) -> int:
        return len(self.vertices)

    @property
    def m(self) -> int:
        return int(self.edges.shape[0])

    @classmethod
    def from_arrays(
        cls,
        vertices: Sequence[str],
        edges: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> "ComparisonGraph":
        verts = list(vertices)
        return cls(
            vertices=verts,
            vertex_index={v: i for i, v in enumerate(verts)},
            edges=np.asarray(edges, dtype=np.int64),
            y=np.asarray(y, dtype=np.float64),
            w=np.asarray(w, dtype=np.float64),
        )


def load_graph_from_db(con: duckdb.DuckDBPyConnection) -> ComparisonGraph:
    """Load the `comparison_edge` table into a ComparisonGraph."""
    verts = [
        r[0]
        for r in con.execute(
            "SELECT business_id FROM ("
            "  SELECT i AS business_id FROM comparison_edge "
            "  UNION "
            "  SELECT j AS business_id FROM comparison_edge"
            ") ORDER BY business_id"
        ).fetchall()
    ]
    vertex_index = {v: i for i, v in enumerate(verts)}

    rows = con.execute("SELECT i, j, y_ij, w_ij FROM comparison_edge ORDER BY i, j").fetchall()
    if not rows:
        return ComparisonGraph.from_arrays(
            verts, np.zeros((0, 2), dtype=np.int64), np.zeros(0), np.zeros(0)
        )
    edges = np.array([(vertex_index[i], vertex_index[j]) for i, j, _, _ in rows], dtype=np.int64)
    y = np.array([r[2] for r in rows], dtype=np.float64)
    w = np.array([r[3] for r in rows], dtype=np.float64)
    return ComparisonGraph.from_arrays(verts, edges, y, w)


def build_B1(g: ComparisonGraph) -> sparse.csr_matrix:
    """Vertex-edge incidence, shape (n, m). B1[i, e]=-1 and B1[j, e]=+1 for edge e=(i, j)."""
    m = g.m
    rows = np.concatenate([g.edges[:, 0], g.edges[:, 1]])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    data = np.concatenate([-np.ones(m), np.ones(m)])
    return sparse.coo_matrix((data, (rows, cols)), shape=(g.n, m)).tocsr()


def component_labels(g: ComparisonGraph) -> tuple[int, np.ndarray, np.ndarray]:
    """Return (n_components, vertex_label, edge_label)."""
    if g.m == 0:
        return g.n, np.arange(g.n, dtype=np.int64), np.zeros(0, dtype=np.int64)
    adj = sparse.coo_matrix(
        (np.ones(g.m), (g.edges[:, 0], g.edges[:, 1])), shape=(g.n, g.n)
    ).tocsr()
    n_cc, labels = connected_components(adj, directed=False)
    edge_labels = labels[g.edges[:, 0]]  # both endpoints share a component label
    return int(n_cc), labels.astype(np.int64), edge_labels.astype(np.int64)


def largest_component(g: ComparisonGraph) -> ComparisonGraph:
    """Keep a single connected group so all displayed scores are comparable."""
    if not g.n:
        return g
    _, labels, edge_labels = component_labels(g)
    label = int(np.argmax(np.bincount(labels)))
    vertices = np.flatnonzero(labels == label)
    remap = np.full(g.n, -1, dtype=np.int64)
    remap[vertices] = np.arange(len(vertices))
    keep = edge_labels == label
    return ComparisonGraph.from_arrays(
        [g.vertices[v] for v in vertices], remap[g.edges[keep]], g.y[keep], g.w[keep]
    )
