"""
Finite-horizon lifted-system machinery implementing the exact Sheikhi,
Gleizer, Mohajerin Esfahani & Keviczky (2025) construction (arXiv:2509.01347,
Section IV): observability/lifting matrix O_L, Toeplitz fault map T^c_L,
parity/nullspace annihilator K^y_L (K^y_L O_L = 0), and the projected fault
dictionary D^c_L = K^y_L T^c_L (their Eq. 8).

Numerically unchanged from the original derivation.
Explicit numerical tolerance is a module-level constant, not a silent
default buried in a function signature.
"""
import numpy as np

# Explicit numerical tolerance for rank/nullspace computations throughout
# the structural engine. Used consistently by dictionaries.py and
# criterion.py -- never silently overridden per-call without being logged.
RANK_TOL = 1e-8


def observability_matrix_L(Acl, C, L):
    """O_L: stack C, C*Acl, ..., C*Acl^(L-1) -> shape (L*ny, n)."""
    n = Acl.shape[0]
    rows = [C]
    Ak = np.eye(n)
    for _ in range(L - 1):
        Ak = Ak @ Acl
        rows.append(C @ Ak)
    return np.vstack(rows)


def nullspace_annihilator(O_L, tol=RANK_TOL):
    """K^y_L: any full-row-rank matrix with K^y_L @ O_L = 0, of shape
    (L*ny - rank(O_L), L*ny). Built via SVD: O_L = U S V^T; the rows of
    K^y_L are the columns of U past the numerical rank of O_L."""
    U, s, Vt = np.linalg.svd(O_L, full_matrices=True)
    rank = int(np.sum(s > tol * max(O_L.shape) * (s[0] if len(s) else 1)))
    K = U[:, rank:].T
    resid = float(np.linalg.norm(K @ O_L))
    return K, rank, resid


def toeplitz_dictionary(Acl, Bcol, Ccol, Dcol, L):
    """T^c_L: the raw (unprojected) finite-horizon Toeplitz map from an
    L-dimensional arbitrary additive fault sequence on this single channel
    to the stacked output window, shape (L*ny, L)."""
    n = Acl.shape[0]
    ny = Ccol.shape[0]
    D = np.zeros((L * ny, L))
    for k in range(L):
        col = np.zeros(L * ny)
        col[k * ny:(k + 1) * ny] = Dcol
        x = Bcol.copy()
        for t in range(k + 1, L):
            col[t * ny:(t + 1) * ny] = Ccol @ x
            x = Acl @ x
        D[:, k] = col
    return D


def projected_dictionary(Acl, Bcol, Ccol, Dcol, L, K=None):
    """D^c_L = K^y_L @ T^c_L (Sheikhi Eq. 8). If K is not supplied it is
    built fresh from observability_matrix_L/nullspace_annihilator (Ccol
    determines the diagnostic output map)."""
    if K is None:
        O_L = observability_matrix_L(Acl, Ccol, L)
        K, _, _ = nullspace_annihilator(O_L)
    T = toeplitz_dictionary(Acl, Bcol, Ccol, Dcol, L)
    return K @ T


def rank_and_basis(D, tol=RANK_TOL):
    """SVD-derived numerical rank and an orthonormal basis for Range(D).

    Uses the SAME explicit rank convention as nullspace_annihilator()
    (rank = count of singular values exceeding tol * max(shape) * s_max),
    so every rank-dependent quantity in this module -- the observability
    rank, the nullspace dimension, and now the dictionary ranks used by
    principal_angles/d_cap -- is computed by one consistent rule rather
    than several different tolerance conventions scattered across
    functions (an earlier audit found that the previous
    principal_angles() built its basis from an unpivoted QR of a
    potentially rank-deficient D, which is not in general a robust
    orthonormal basis for Range(D) when D is rank-deficient or
    ill-conditioned -- an unpivoted QR can leave near-zero pivots in
    arbitrary positions rather than trailing columns, so truncating to
    the first r columns can silently mix a genuine basis vector with
    numerical noise. SVD does not have this failure mode: U's columns are
    always ordered by decreasing singular value, so U[:, :r] is always a
    genuine orthonormal basis for the numerically-r-dimensional part of
    Range(D), for any D, rank-deficient or not."""
    if D.shape[1] == 0:
        return np.zeros((D.shape[0], 0)), 0, np.array([])
    U, s, _ = np.linalg.svd(D, full_matrices=False)
    thresh = tol * max(D.shape) * (s[0] if len(s) else 1.0)
    rank = int(np.sum(s > thresh))
    Q = U[:, :max(rank, 0)]
    return Q, rank, s


def principal_angles(D1, D2, tol=RANK_TOL, angle_tol=1e-4):
    """Principal angles between Range(D1) and Range(D2), via SVD-derived
    orthonormal bases (rank_and_basis, above) and SVD of the cross-Gram
    matrix. intersection_dim counts angles numerically below angle_tol
    (an explicit tolerance, distinct from RANK_TOL which governs the
    rank/basis computations feeding this)."""
    if D1.shape[1] == 0 or D2.shape[1] == 0:
        return np.array([]), 0, 0, 0
    Q1, r1, _ = rank_and_basis(D1, tol=tol)
    Q2, r2, _ = rank_and_basis(D2, tol=tol)
    if r1 == 0 or r2 == 0:
        return np.array([]), 0, r1, r2
    M = Q1.T @ Q2
    s = np.clip(np.linalg.svd(M, compute_uv=False), -1, 1)
    angles = np.arccos(s)
    intersection_dim = int(np.sum(angles < angle_tol))
    return angles, intersection_dim, r1, r2


def d_cap_via_rank_identity(D1, D2, tol=RANK_TOL):
    """Independent cross-check of the principal-angle intersection
    dimension, via the rank identity
        dim(Range(D1) ∩ Range(D2)) = rank(D1) + rank(D2) - rank([D1 D2])
    (Sheikhi Lemma IV.4). Computed with the SAME rank_and_basis()
    convention as principal_angles(), so the two routes are directly
    comparable rather than using two different numerical-rank rules that
    could disagree for a reason having nothing to do with the geometry."""
    _, r1, _ = rank_and_basis(D1, tol=tol)
    _, r2, _ = rank_and_basis(D2, tol=tol)
    if D1.shape[1] == 0 or D2.shape[1] == 0:
        stacked_rank = r1 + r2
    else:
        stacked = np.hstack([D1, D2])
        _, stacked_rank, _ = rank_and_basis(stacked, tol=tol)
    d_cap_rank = r1 + r2 - stacked_rank
    return d_cap_rank, r1, r2, stacked_rank
