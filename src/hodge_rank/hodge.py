from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from hodge_rank.graph import ComparisonGraph, build_B1
from hodge_rank.triangles import build_B2, enumerate_triangles


@dataclass
class HodgeDecomposition:
    """Result of the weighted combinatorial Hodge decomposition.

    Energies use the W-inner product on edges: <a, b>_W = Σ w_e a_e b_e.
    The gradient and weighted curl spaces are W-orthogonal. Their
    orthogonal complement is harmonic, so Pythagoras gives
        E_total == E_grad + E_curl + E_harm.
    """

    s: np.ndarray
    phi: np.ndarray
    y_grad: np.ndarray
    y_curl: np.ndarray
    y_harm: np.ndarray
    e_total: float
    e_grad: float
    e_curl: float
    e_harm: float

    @property
    def rankability(self) -> float:
        return self.e_grad / self.e_total if self.e_total > 0 else 0.0

    @property
    def curl_fraction(self) -> float:
        return self.e_curl / self.e_total if self.e_total > 0 else 0.0

    @property
    def harmonic_fraction(self) -> float:
        return self.e_harm / self.e_total if self.e_total > 0 else 0.0


def _lsqr_solve(A: sparse.spmatrix, b: np.ndarray, atol: float, btol: float) -> np.ndarray:
    """Thin wrapper with tight tolerances. Returns the solution vector."""
    sol, *_ = lsqr(A, b, atol=atol, btol=btol, iter_lim=10 * max(A.shape))
    return sol


def decompose(
    g: ComparisonGraph,
    triangles: np.ndarray | None = None,
    atol: float = 1e-12,
    btol: float = 1e-12,
) -> HodgeDecomposition:
    """Weighted Hodge decomposition of (y, w) on the graph g.

    If triangles is None, enumerate them from g.
    Zero-weight edges carry no observed energy and cannot complete triangles.
    Their output coordinates are placeholders, not inferred harmonic flows.
    """
    if triangles is None:
        triangles = enumerate_triangles(g)

    if not np.isfinite(g.w).all() or (g.w < 0).any():
        raise ValueError("Edge weights must be finite and nonnegative")
    # A zero-weight edge is absent from the observed complex. In particular,
    # it cannot complete a filled triangle.
    if (g.w == 0).any() and len(triangles):
        supported = {tuple(e) for e in g.edges[g.w > 0]}
        triangles = np.asarray(
            [
                (a, b, c)
                for a, b, c in triangles
                if (a, b) in supported and (a, c) in supported and (b, c) in supported
            ],
            dtype=np.int64,
        ).reshape(-1, 3)

    B1 = build_B1(g)
    B2 = build_B2(g, triangles)

    sqrt_w = np.sqrt(g.w)

    # Step 1: gradient. Solve min ‖diag(sqrt_w) (y − B1ᵀ s)‖₂
    # A = diag(sqrt_w) @ B1ᵀ has shape (m, n), rank = n − (# connected components).
    # LSQR from x₀=0 yields the minimum-norm least-squares solution, which is
    # orthogonal to ker(A) = span of per-component indicator vectors, so s is
    # zero-mean within each connected component (gauge fixed automatically).
    A_grad = sparse.diags(sqrt_w) @ B1.T
    b_grad = sqrt_w * g.y
    s = _lsqr_solve(A_grad, b_grad, atol, btol)
    y_grad = B1.T @ s

    # Step 2: curl. The adjoint of B2.T in the edge W-inner product is
    # W^-1 B2 (unit triangle metric). B1 W (W^-1 B2) = B1 B2 = 0.
    y_rest = g.y - y_grad
    if B2.shape[1] > 0:
        inv_w = np.divide(1.0, g.w, out=np.zeros_like(g.w), where=g.w > 0)
        curl_operator = sparse.diags(inv_w) @ B2
        A_curl = sparse.diags(sqrt_w) @ curl_operator
        b_curl = sqrt_w * y_rest
        phi = _lsqr_solve(A_curl, b_curl, atol, btol)
        y_curl = curl_operator @ phi
    else:
        phi = np.zeros(0)
        y_curl = np.zeros_like(g.y)

    y_harm = y_rest - y_curl

    e_total = float((g.w * g.y * g.y).sum())
    e_grad = float((g.w * y_grad * y_grad).sum())
    e_curl = float((g.w * y_curl * y_curl).sum())
    e_harm = float((g.w * y_harm * y_harm).sum())

    return HodgeDecomposition(
        s=s,
        phi=phi,
        y_grad=y_grad,
        y_curl=y_curl,
        y_harm=y_harm,
        e_total=e_total,
        e_grad=e_grad,
        e_curl=e_curl,
        e_harm=e_harm,
    )
