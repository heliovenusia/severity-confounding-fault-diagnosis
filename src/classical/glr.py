"""
Classical, genuinely implementable model-based FDI baseline: a
Generalized Likelihood Ratio (GLR) test between the two composite
hypotheses H_p (process fault) and H_s (sensor fault), for one
experiment-matrix condition at a time.

Why GLR, and why it is NOT the Bayes reference in disguise
-------------------------------------------------------------------------
The Bayes-optimal reference (src/statistical/ensemble_astar.py) decides by
comparing MARGINAL (mixture) likelihoods -- it integrates the Gaussian
likelihood over the full declared severity/onset prior for each
hypothesis and compares the two integrals. That requires knowing and
using the exact declared prior weights, which is the ground-truth
information-theoretic ceiling, not something a practitioner would
implement as a "diagnostic algorithm" (it IS the quantity the Bayes gap
measures a real method against).

GLR instead PLUGS IN the maximum-likelihood estimate of each hypothesis's
unknown nuisance parameters (severity, onset) and compares the two
best-fit likelihoods:

    Lambda_p = max_{alpha in [alpha_p_lo, alpha_p_hi], onset in {0,1,2,3}} p(y | H_p, alpha, onset)
    Lambda_s = max_{alpha in [alpha_s_lo, alpha_s_hi], onset in {0,1,2,3}} p(y | H_s, alpha, onset)
    decide H_p  iff  log Lambda_p - log Lambda_s > threshold

This is a standard, textbook composite-hypothesis GLR/matched-filter test
(the plug-in/max-likelihood decision rule), a DIFFERENT classical
principle from Bayes marginalization (max vs. integral over the nuisance
parameter), and is what a practitioner with the known plant model but NO
access to the true per-run severity/onset (both are latent, estimated
from the single observed window) could actually run.

What this method KNOWS (all model-level, public information about the
fault family and plant -- never the true per-run values):
  - the closed-loop plant model Acl, C for this condition (gamma, ny)
  - the two fault direction step-response shapes s_p(onset), s_s(onset)
    for onset in {0,1,2,3} -- i.e. the same delayed_step_response used to
    build the declared priors' quadrature mixture, NOT the true onset of
    any specific test run
  - the declared ADMISSIBLE severity RANGE [pct_lo, pct_hi] = [1,30]% for
    both fault families (the prior's SUPPORT, a property of the fault
    family, not of any one run)
  - the noise covariance Sigma at the reference (sigma_h, sigma_v)
    operating point used for this condition's dataset

What this method does NOT know (checked in tests/test_classical.py):
  - the true severity_pct of the observed run
  - the true onset of the observed run
  - the true fault_class label (that is what it predicts)
  - any structural quantity (d_cap, verdict, mechanism) or condition_id
"""
from dataclasses import dataclass

import numpy as np

from ..plant import build_model, diff_system_matrices, DEFAULT_KGAIN, kc as KC_GAIN, DEFAULT_V0
from ..faults import PROCESS_FAULTS, FaultKind
from ..statistical.point_astar import state_noise_toeplitz
from ..statistical.ensemble_astar import delayed_step_response

ONSET_GRID = (0, 1, 2, 3)       # matches the declared onset prior's support, exactly
SEVERITY_RANGE_PCT = (1.0, 30.0)  # matches the declared severity prior's support, exactly


@dataclass(frozen=True)
class GLRModel:
    """Everything the GLR test needs, precomputed once per condition --
    the model-level quantities listed in the module docstring, and
    nothing about any specific observed run."""
    condition_id: str
    Acl: np.ndarray
    C: np.ndarray
    Sigma: np.ndarray
    Sigma_inv: np.ndarray
    log_det_Sigma: float
    s_p_by_onset: dict          # onset -> unit-severity process-fault response shape
    s_s_by_onset: dict          # onset -> unit-severity sensor-fault response shape
    alpha_p_lo: float
    alpha_p_hi: float
    alpha_s_lo: float
    alpha_s_hi: float
    ny_L: int


def build_glr_model(cond, sigma_h: float, sigma_v: float, L: int = 8,
                     severity_range_pct=SEVERITY_RANGE_PCT, onset_grid=ONSET_GRID,
                     Kgain: float = DEFAULT_KGAIN) -> GLRModel:
    model = build_model(gamma=cond.gamma, Kgain=Kgain)
    rec = PROCESS_FAULTS[cond.process_fault_id]
    h0 = model["h0"]
    Acl, Edelta, C, Ddelta = diff_system_matrices(model, 0.0, cond.sensor_idx, cond.ny)

    if rec.kind == FaultKind.PROCESS_ACTUATOR:
        p_anchor = DEFAULT_V0
    else:
        tank_idx = int(cond.process_fault_id[-1]) - 1
        p_anchor = h0[tank_idx]
    s_anchor = KC_GAIN * h0[cond.sensor_idx]

    Bp, Dp = rec.direction_fn(model), np.zeros(C.shape[0])
    Bs, Ds = Edelta[:, 1], Ddelta[:, 1]

    s_p_by_onset = {o: delayed_step_response(Acl, Bp, C, Dp, L, o) for o in onset_grid}
    s_s_by_onset = {o: delayed_step_response(Acl, Bs, C, Ds, L, o) for o in onset_grid}

    G = state_noise_toeplitz(Acl, C, L)
    Sigma = (sigma_h ** 2) * (G @ G.T) + (sigma_v ** 2) * np.eye(L * C.shape[0])
    Sigma_inv = np.linalg.inv(Sigma)
    _, log_det = np.linalg.slogdet(Sigma)

    lo, hi = severity_range_pct
    return GLRModel(
        condition_id=cond.condition_id, Acl=Acl, C=C, Sigma=Sigma, Sigma_inv=Sigma_inv,
        log_det_Sigma=float(log_det), s_p_by_onset=s_p_by_onset, s_s_by_onset=s_s_by_onset,
        alpha_p_lo=(lo / 100.0) * p_anchor, alpha_p_hi=(hi / 100.0) * p_anchor,
        alpha_s_lo=(lo / 100.0) * s_anchor, alpha_s_hi=(hi / 100.0) * s_anchor,
        ny_L=L * C.shape[0],
    )


def _gaussian_loglik(y, mu, Sigma_inv, log_det_Sigma, ny_L):
    diff = y - mu
    quad = diff @ Sigma_inv @ diff
    return -0.5 * (ny_L * np.log(2 * np.pi) + log_det_Sigma + quad)


def _profile_max_loglik(y, s_by_onset, alpha_lo, alpha_hi, Sigma_inv, log_det_Sigma, ny_L):
    """max over onset in the grid and alpha in [alpha_lo, alpha_hi] of the
    Gaussian log-likelihood, via the closed-form per-onset alpha MLE
    (unconstrained MLE clipped to the declared admissible range -- valid
    because the Gaussian log-likelihood is concave/quadratic in alpha for
    fixed onset, so the constrained maximizer is exactly the projection
    of the unconstrained maximizer onto the interval)."""
    best_ll, best_onset, best_alpha = -np.inf, None, None
    for onset, s in s_by_onset.items():
        denom = s @ Sigma_inv @ s
        if denom < 1e-14:
            alpha_hat = 0.0
        else:
            alpha_hat_unc = float((s @ Sigma_inv @ y) / denom)
            alpha_hat = min(max(alpha_hat_unc, alpha_lo), alpha_hi)
        ll = _gaussian_loglik(y, alpha_hat * s, Sigma_inv, log_det_Sigma, ny_L)
        if ll > best_ll:
            best_ll, best_onset, best_alpha = ll, onset, alpha_hat
    return best_ll, best_onset, best_alpha


def glr_scores(y: np.ndarray, glr_model: GLRModel):
    """Return (loglik_p_max, loglik_s_max, glr_statistic, details). The
    GLR statistic is loglik_p_max - loglik_s_max; positive favors H_p.
    `y` must be the RAW observed trajectory only -- no other argument to
    this function carries any per-run ground truth."""
    ll_p, onset_p, alpha_p = _profile_max_loglik(
        y, glr_model.s_p_by_onset, glr_model.alpha_p_lo, glr_model.alpha_p_hi,
        glr_model.Sigma_inv, glr_model.log_det_Sigma, glr_model.ny_L)
    ll_s, onset_s, alpha_s = _profile_max_loglik(
        y, glr_model.s_s_by_onset, glr_model.alpha_s_lo, glr_model.alpha_s_hi,
        glr_model.Sigma_inv, glr_model.log_det_Sigma, glr_model.ny_L)
    stat = ll_p - ll_s
    details = dict(loglik_p=ll_p, loglik_s=ll_s, onset_hat_p=onset_p, alpha_hat_p=alpha_p,
                    onset_hat_s=onset_s, alpha_hat_s=alpha_s)
    return ll_p, ll_s, stat, details


def glr_predict(y: np.ndarray, glr_model: GLRModel, threshold: float = 0.0):
    """0 = predict process, 1 = predict sensor (matching the RunMetadata
    convention fault_class in {'process','sensor'} -> {0,1} used
    elsewhere in this codebase's ML pipeline)."""
    _, _, stat, details = glr_scores(y, glr_model)
    pred = 0 if stat > threshold else 1
    return pred, stat, details


def fit_threshold(stats: np.ndarray, labels: np.ndarray, candidate_grid=None):
    """Select the decision threshold on GLR statistics that maximizes
    balanced accuracy, evaluated ONLY on the data passed in (the caller
    must pass TRAIN or VAL statistics, never TEST -- enforced by the
    calling convention in experiments/run_classical_fdi.py, not by this
    function, which is data-agnostic)."""
    if candidate_grid is None:
        lo, hi = float(np.min(stats)), float(np.max(stats))
        candidate_grid = np.linspace(lo, hi, 401)
    best_thr, best_bal_acc = 0.0, -1.0
    for thr in candidate_grid:
        pred = (stats <= thr).astype(int)  # stat>thr -> 0 (process), else 1 (sensor)
        tp0 = np.mean(pred[labels == 0] == 0) if np.any(labels == 0) else np.nan
        tp1 = np.mean(pred[labels == 1] == 1) if np.any(labels == 1) else np.nan
        bal_acc = np.nanmean([tp0, tp1])
        if bal_acc > best_bal_acc:
            best_bal_acc, best_thr = bal_acc, float(thr)
    return best_thr, best_bal_acc
