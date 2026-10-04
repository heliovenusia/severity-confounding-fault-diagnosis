"""
Dimensionally consistent point-realization Bayes-separation engine.

Numerically unchanged from the original derivation.
Separate, physically anchored fault-severity scales (alpha_p, alpha_s /
alpha_leak) and separate noise covariances (Q in state space [cm^2], R in
output space [V^2]) -- never a shared numerical epsilon, never q=r.

  alpha_p    = (pct_p/100) * v0                       [V]   pump faults
  alpha_leak = (pct_leak/100) * h_i0(gamma)            [cm]  leak faults (Type C)
  alpha_s    = (pct_s/100) * kc * h_i0(gamma)          [V]   sensor faults, anchored
                                                               to THAT sensor's own
                                                               nominal reading
  Q = sigma_h^2 * I_4      [cm^2]   state/process noise
  R = sigma_v^2 * I_ny     [V^2]    measurement noise
  Sigma = G Q G^T + R      where G is the state-noise Toeplitz map (contains
                            C-multiplications, so G Q G^T is correctly in V^2)

  Delta = sqrt( (mu_p - mu_s)^T Sigma^-1 (mu_p - mu_s) )
  A*    = Phi(Delta / 2)     (equal priors, two-hypothesis Gaussian Bayes accuracy)
"""
import numpy as np
from scipy.stats import norm

from ..plant import build_model, diff_system_matrices, step_response_shape, DEFAULT_KGAIN, kc as KC_GAIN, DEFAULT_V0
from ..faults import PROCESS_FAULTS, FaultKind


def state_noise_toeplitz(Acl, C, L):
    """G: block-Toeplitz map from a unit STATE disturbance (cm) injected at
    each step k=0..L-1 into the stacked output window, shape (L*ny, L*n).
    Because it contains C-multiplications (V/cm), G @ Q @ G.T is correctly
    expressed in V^2, matching R -- this is what keeps Sigma dimensionally
    consistent."""
    n = Acl.shape[0]
    ny = C.shape[0]
    G = np.zeros((L * ny, L * n))
    for k in range(L):
        block = np.zeros((L * ny, n))
        x = np.eye(n)
        for t in range(k + 1, L):
            block[t * ny:(t + 1) * ny, :] = C @ x
            x = Acl @ x
        G[:, k * n:(k + 1) * n] = block
    return G


def _severity_anchor(process_fault_id, model, sensor_idx):
    """Returns (alpha_p_per_unit_pct, alpha_s_per_unit_pct) scaling factors
    i.e. alpha = pct * (this anchor / 100)."""
    rec = PROCESS_FAULTS[process_fault_id]
    h0 = model["h0"]
    if rec.kind == FaultKind.PROCESS_ACTUATOR:
        p_anchor = DEFAULT_V0
    else:
        tank_idx = int(process_fault_id[-1]) - 1  # 'leak_tank3' -> index 2
        p_anchor = h0[tank_idx]
    s_anchor = KC_GAIN * h0[sensor_idx]
    return p_anchor, s_anchor


def delta_astar_physical(process_fault_id, sensor_idx, ny, gamma, pct_p, pct_s,
                          sigma_h, sigma_v, L, Kgain=DEFAULT_KGAIN):
    """Compute Delta and A* for one pair at one declared (severity, noise)
    operating point. Never hard-codes equal numerical magnitudes for
    dimensionally different fault types -- pct_p/pct_s are both PERCENTAGES
    of their own physical anchors, not shared raw magnitudes."""
    model = build_model(gamma=gamma, Kgain=Kgain)
    rec = PROCESS_FAULTS[process_fault_id]
    p_anchor, s_anchor = _severity_anchor(process_fault_id, model, sensor_idx)
    alpha_p = (pct_p / 100.0) * p_anchor
    alpha_s = (pct_s / 100.0) * s_anchor

    Ep = rec.direction_fn(model)
    Acl, Edelta, C, Ddelta = diff_system_matrices(model, 0.0, sensor_idx, ny)
    s_p = step_response_shape(Acl, Ep, C, np.zeros(ny), L)
    s_s = step_response_shape(Acl, Edelta[:, 1], C, Ddelta[:, 1], L)
    mu_p = alpha_p * s_p
    mu_s = alpha_s * s_s

    G = state_noise_toeplitz(Acl, C, L)
    Sigma = (sigma_h ** 2) * (G @ G.T) + (sigma_v ** 2) * np.eye(L * C.shape[0])
    diff = mu_p - mu_s
    Delta = float(np.sqrt(diff @ np.linalg.solve(Sigma, diff)))
    Astar = float(norm.cdf(Delta / 2))
    return Delta, Astar, dict(alpha_p=float(alpha_p), alpha_s=float(alpha_s),
                               p_anchor=float(p_anchor), s_anchor=float(s_anchor),
                               h0=model["h0"].tolist())
