"""
The frozen structural discernibility criterion and mechanism
classification.

CRITERION (never redefine this):
    d_cap = dim( Range(D_p_L) ∩ Range(D_s_L) )   [Sheikhi Lemma IV.4/IV.6]
    AMBIGUOUS   iff d_cap > 0
    SEPARATED   iff d_cap == 0
This is the sole criterion. The generic dimension-counting lower bound
    floor = max(0, rank(D_p) + rank(D_s) - N)     (N = residual ambient dim)
is retained ONLY as an explanatory/diagnostic annotation of *why* an
intersection is forced by dimension-counting alone -- it must NEVER be
subtracted from d_cap to redefine the verdict. This was a real error,
explicitly corrected by adversarial review; the
correction is enforced here structurally (the verdict function takes only
d_cap, not floor, as its argument) so the error cannot silently recur.

Two mechanism labels, purely explanatory, computed from d_cap and floor:
    DIMENSION_FORCED : d_cap == floor  (ambiguity is generic; adding
                        sensors that raise the ambient dimension N
                        relative to rank(D_p)+rank(D_s) can eliminate it)
    EXCESS_FAULT_SPECIFIC : d_cap > floor  (ambiguity exceeds what generic
                        dimension-counting alone would force; reflects a
                        genuine geometric alignment between these two
                        specific fault directions, not just dimension
                        starvation -- not generically eliminable by sensor
                        count alone)
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .dictionaries import (
    RANK_TOL, observability_matrix_L, nullspace_annihilator,
    toeplitz_dictionary, principal_angles, d_cap_via_rank_identity,
)


class StructuralVerdict(Enum):
    AMBIGUOUS = "AMBIGUOUS"
    SEPARATED = "SEPARATED"


class AmbiguityMechanism(Enum):
    DIMENSION_FORCED = "DIMENSION_FORCED"
    EXCESS_FAULT_SPECIFIC = "EXCESS_FAULT_SPECIFIC"
    NOT_APPLICABLE = "NOT_APPLICABLE"  # separated pairs have no mechanism


@dataclass(frozen=True)
class StructuralResult:
    r1: int              # rank(D_p)
    r2: int              # rank(D_s)
    ambient_dim: int      # N = L*ny - n
    generic_floor: int    # max(0, r1+r2-N), diagnostic only
    d_cap: int            # THE criterion (principal-angle route)
    d_cap_rank: int       # independent cross-check: rank(D_p)+rank(D_s)-rank([D_p D_s])
    excess: int           # d_cap - generic_floor, diagnostic only
    verdict: StructuralVerdict
    mechanism: AmbiguityMechanism
    L: int
    gamma: float
    tol: float = RANK_TOL


def compute_pair_structural(Acl, Ep, Es_col, Ds_col, C, L, gamma, tol=RANK_TOL):
    """Compute the frozen structural result for one (process-direction,
    sensor-direction) pair at horizon L. Ep is the process-fault direction
    (state-entry, e.g. a B_d column or a leak e_i); Es_col/Ds_col are the
    sensor-fault channel's state-entry (zero) and feedthrough (-e_idx)
    columns as returned by diff_system_matrices."""
    ny = C.shape[0]
    O_L = observability_matrix_L(Acl, C, L)
    K, obs_rank, resid = nullspace_annihilator(O_L, tol=tol)
    Tp = toeplitz_dictionary(Acl, Ep, C, np.zeros(ny), L)
    Ts = toeplitz_dictionary(Acl, Es_col, C, Ds_col, L)
    Dp = K @ Tp
    Ds = K @ Ts
    angles, d_cap, r1, r2 = principal_angles(Dp, Ds, tol=tol)
    d_cap_rank, r1_rank, r2_rank, stacked_rank = d_cap_via_rank_identity(Dp, Ds, tol=tol)
    if r1 != r1_rank or r2 != r2_rank:
        raise AssertionError(
            f"rank mismatch between principal_angles and rank_and_basis: "
            f"r1={r1} vs {r1_rank}, r2={r2} vs {r2_rank} -- both routes must use the "
            f"same RANK_TOL convention (rank_and_basis in dictionaries.py); this should "
            f"be structurally impossible since both call the same helper, so a mismatch "
            f"here indicates a real bug, not just numerical looseness.")
    if d_cap != d_cap_rank:
        raise AssertionError(
            f"d_cap disagreement: principal-angle route gives d_cap={d_cap}, rank-identity "
            f"route (Sheikhi Lemma IV.4: rank(D_p)+rank(D_s)-rank([D_p D_s])) gives "
            f"d_cap_rank={d_cap_rank} (r1={r1_rank}, r2={r2_rank}, rank([D_p D_s])={stacked_rank}). "
            f"These two independently-derived routes "
            f"must agree; a disagreement here must stop execution and be investigated, not "
            f"be silently reported as-is.")
    ambient = K.shape[0]
    floor = max(0, r1 + r2 - ambient)
    excess = d_cap - floor
    verdict = StructuralVerdict.AMBIGUOUS if d_cap > 0 else StructuralVerdict.SEPARATED
    if verdict == StructuralVerdict.SEPARATED:
        mechanism = AmbiguityMechanism.NOT_APPLICABLE
    elif d_cap == floor:
        mechanism = AmbiguityMechanism.DIMENSION_FORCED
    else:
        mechanism = AmbiguityMechanism.EXCESS_FAULT_SPECIFIC
    return StructuralResult(r1=r1, r2=r2, ambient_dim=ambient, generic_floor=floor,
                             d_cap=d_cap, d_cap_rank=d_cap_rank, excess=excess, verdict=verdict,
                             mechanism=mechanism, L=L, gamma=gamma, tol=tol)


def result_as_dict(res: StructuralResult):
    return dict(r1=res.r1, r2=res.r2, ambient_dim=res.ambient_dim,
                generic_floor=res.generic_floor, d_cap=res.d_cap, d_cap_rank=res.d_cap_rank,
                excess=res.excess, verdict=res.verdict.value, mechanism=res.mechanism.value,
                L=res.L, gamma=res.gamma, tol=res.tol)
