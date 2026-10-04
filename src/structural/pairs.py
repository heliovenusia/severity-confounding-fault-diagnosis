"""
Thin convenience layer tying the plant, fault registry, and structural
criterion together for one (process-fault, sensor-fault) pair -- a
tested, reusable function with an explicit primary horizon and a
separate robustness-horizon argument, which are never mixed into one row
(an earlier ad hoc implementation conflated the two and was corrected).
"""
from dataclasses import dataclass

from ..plant import build_model, diff_system_matrices, DEFAULT_KGAIN
from ..faults import PROCESS_FAULTS
from .criterion import compute_pair_structural, StructuralResult


def structural_pair(process_fault_id: str, sensor_idx: int, ny: int, gamma: float,
                     L: int, Kgain: float = DEFAULT_KGAIN) -> StructuralResult:
    """Compute the structural result for one named process fault vs. one
    sensor-fault channel, at one EXPLICIT primary horizon L. Callers that
    also want a robustness sequence must call this again at each L in that
    sequence and keep the results in separate fields -- this function
    itself never conflates horizons."""
    model = build_model(gamma=gamma, Kgain=Kgain)
    rec = PROCESS_FAULTS[process_fault_id]
    Ep = rec.direction_fn(model)
    Acl, Edelta, C, Ddelta = diff_system_matrices(model, 0.0, sensor_idx, ny)
    Es_col = Edelta[:, 1]
    Ds_col = Ddelta[:, 1]
    return compute_pair_structural(Acl, Ep, Es_col, Ds_col, C, L, gamma)


def worst_case_gamma(process_fault_id: str, sensor_idx: int, ny: int, L: int,
                      gamma_grid, Kgain: float = DEFAULT_KGAIN):
    """Scan gamma_grid and return (gamma_star, d_cap_curve) where gamma_star
    maximizes d_cap -- the worst case for a 'separated' claim, matching the
    convention used for every reported structural table."""
    curve = [structural_pair(process_fault_id, sensor_idx, ny, g, L, Kgain).d_cap
             for g in gamma_grid]
    max_dcap = max(curve)
    if max_dcap > 0:
        g_star = float(gamma_grid[int(curve.index(max_dcap))])
    else:
        g_star = float(gamma_grid[len(gamma_grid) // 2])  # any gamma; all separated
    return g_star, curve
