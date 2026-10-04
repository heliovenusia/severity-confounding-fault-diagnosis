"""
Ensemble Bayes-accuracy engine under DECLARED priors over fault severity
and onset.

An earlier implementation approximated a stated-CONTINUOUS log-uniform
severity prior with only 7 discrete log-spaced nodes, weighted by
`np.gradient(log_grid)`, and called this "trapezoidal quadrature." It was
not: on a uniform grid np.gradient returns the constant spacing h at
every node including the two endpoints (a linear sequence's finite
difference is exactly h everywhere), whereas true composite trapezoidal
quadrature requires half-weight at the two endpoints. That earlier rule
was actually an unweighted equal-node rule, not what it was labeled -- a
real, if minor, mislabeling error, corrected here.

Fix, done properly:
  - The severity prior is explicitly declared CONTINUOUS:
        ln(severity_pct) ~ Uniform(ln(pct_lo), ln(pct_hi))
    (log-uniform in the original percentage variable).
  - It is approximated by GAUSS-LEGENDRE quadrature in the log-severity
    variable u=ln(pct) -- a proper numerical-integration rule (exact for
    polynomials up to degree 2*n_nodes-1 in u, not just a labeling fix),
    with weights correctly normalized so they sum to 1 (they approximate
    the probability-weighted average under the Uniform(a,b) density in u,
    not merely a bare Gauss-Legendre weight on [-1,1]).
  - A genuine composite-trapezoidal rule (with correct half-weights at the
    two endpoints) is also implemented, purely as a cross-check against
    the corrected Gauss-Legendre rule and against the earlier, mislabeled
    equal-weight rule -- all three are reported in the convergence table
    so the discrepancy is visible, not hidden.
  - The onset prior is explicitly declared DISCRETE (Discrete-uniform over
    {0,1,2,3} sample indices within the L=8 window) -- it is exact, not a
    quadrature approximation of anything continuous, and is stated as such.
  - Resolution convergence is checked at n_severity in {7,15,31,63,127}
    (the exact sequence specified). The reported "converged" resolution is
    the smallest one after which every larger resolution's ensemble A*
    VALUE stays within DECLARED_CONVERGENCE_TOL of the final (127-node)
    value.

Audit correction: an earlier write-up of this result claimed "zero
decisions flip between n=7 and n=127" for every condition. That is false
and was corrected. The common-fixed-test-set
convergence check (resolution_convergence(), below) DOES record individual
per-point decision flips (`decisions_flipped_since_prev_resolution`), and
several conditions show a handful of flips between n=7 and n=15 (observed
range across the 9 conditions: 0-4 flipped decisions out of 6,000
fixed test points, i.e. at most 0.07% of points). What is true, and is the
actual basis for "converged," is narrower and weaker: the AGGREGATE
accuracy (ensemble A*) stays within the declared 1e-3 tolerance from
n_severity=7 onward, because the individual flips that do occur are
roughly balanced between newly-correct and newly-incorrect and so
largely cancel in the aggregate. This is a materially different claim
from "the decision rule is unchanged," and only the aggregate-accuracy
claim is asserted anywhere in this codebase now. Because 7-node
convergence turned out to rest on this weaker (if still practically
adequate) property rather than genuine per-point stability, the
PRODUCTION DEFAULT resolution used by this module is n_severity=31 (not
7) -- see DEFAULT_N_SEVERITY below -- retained as a safety margin, not
because a rigorous argument requires exactly 31. The n=7 finding is kept and reported (as evidence, with its
actual flip counts alongside it), never re-asserted as "zero flips."
"""

# Conservative production default:
# do not rely on the minimal (n=7) resolution that was shown to still
# produce nonzero per-point decision flips near that boundary; use 31
# Gauss-Legendre nodes as the standard operating resolution for every
# ensemble-A* computation in this codebase unless a specific later
# computational-cost reason requires dropping to a coarser grid (in which
# case that reason must be stated at the call site, not silently assumed).
DEFAULT_N_SEVERITY = 31
import numpy as np
from scipy.special import roots_legendre

from ..plant import build_model, diff_system_matrices, DEFAULT_KGAIN, kc as KC_GAIN, DEFAULT_V0
from ..faults import PROCESS_FAULTS, FaultKind
from .point_astar import state_noise_toeplitz

DECLARED_CONVERGENCE_TOL = 1e-3   # declared tolerance on ensemble A*, stated explicitly
RESOLUTION_SEQUENCE = (7, 15, 31, 63, 127)


def delayed_step_response(Acl, Bcol, Ccol, Dcol, L, onset):
    n = Acl.shape[0]
    x = np.zeros(n)
    ys = []
    for t in range(L):
        active = 1.0 if t >= onset else 0.0
        ys.append(Ccol @ x + Dcol * active)
        x = Acl @ x + Bcol * active
    return np.concatenate(ys)


def gauss_legendre_log_uniform_nodes(n_nodes, lo, hi):
    """Proper Gauss-Legendre quadrature for E_u~Uniform(a,b)[f(u)], a=ln(lo),
    b=ln(hi), mapped from the canonical [-1,1] rule. Returns (pct_nodes,
    weights) with weights summing to exactly 1 (verified below), pct_nodes
    = exp(u_i) so callers work directly in the original percentage units."""
    a, b = np.log(lo), np.log(hi)
    x, w = roots_legendre(n_nodes)           # nodes/weights on [-1,1], sum(w)=2
    u = 0.5 * (b - a) * x + 0.5 * (a + b)     # mapped nodes in [a,b]
    weights = w / 2.0                         # normalized: sum(weights) = 1 exactly
    assert abs(weights.sum() - 1.0) < 1e-10
    return np.exp(u), weights


def composite_trapezoidal_log_uniform_nodes(n_nodes, lo, hi):
    """Genuine composite trapezoidal rule (correct half-weights at the two
    endpoints), for cross-checking against Gauss-Legendre. NOT what Round
    14 actually implemented (see module docstring) despite the label used
    there -- this is the real thing, included only as a comparison."""
    a, b = np.log(lo), np.log(hi)
    u = np.linspace(a, b, n_nodes)
    h = (b - a) / (n_nodes - 1)
    w = np.full(n_nodes, h)
    w[0] *= 0.5
    w[-1] *= 0.5
    w = w / w.sum()  # normalize to a probability-weighted average (divides by (b-a))
    return np.exp(u), w


def mislabeled_equal_weight_nodes(n_nodes, lo, hi):
    """Reproduces exactly what an earlier version of declared_priors()
    actually computed (np.gradient on a uniform grid -> equal weight h at every
    node, including endpoints), kept only so the convergence table can
    show, quantitatively, how much this mislabeling changed the answer."""
    a, b = np.log(lo), np.log(hi)
    u = np.linspace(a, b, n_nodes)
    w = np.gradient(u)
    w = w / w.sum()
    return np.exp(u), w


def declared_priors(n_severity=DEFAULT_N_SEVERITY, pct_lo=1.0, pct_hi=30.0, onsets=(0, 1, 2, 3),
                     rule="gauss_legendre"):
    """severity ~ CONTINUOUS Log-Uniform(pct_lo, pct_hi), approximated by
    the requested quadrature `rule` ('gauss_legendre' [default, proper],
    'trapezoidal' [genuine composite trapezoidal, comparison only],
    'mislabeled_equal_weight' [the earlier, mislabeled rule, comparison only]).
    onset ~ DISCRETE Uniform{onsets}, exact (not a quadrature approximation
    of anything continuous)."""
    if rule == "gauss_legendre":
        pct_nodes, w = gauss_legendre_log_uniform_nodes(n_severity, pct_lo, pct_hi)
    elif rule == "trapezoidal":
        pct_nodes, w = composite_trapezoidal_log_uniform_nodes(n_severity, pct_lo, pct_hi)
    elif rule == "mislabeled_equal_weight":
        pct_nodes, w = mislabeled_equal_weight_nodes(n_severity, pct_lo, pct_hi)
    else:
        raise ValueError(f"unknown quadrature rule: {rule}")
    onset_w = np.ones(len(onsets)) / len(onsets)
    return list(zip(pct_nodes, w)), list(zip(onsets, onset_w))


def _severity_anchor(process_fault_id, model, sensor_idx, is_process):
    h0 = model["h0"]
    if is_process:
        rec = PROCESS_FAULTS[process_fault_id]
        if rec.kind == FaultKind.PROCESS_ACTUATOR:
            return DEFAULT_V0
        tank_idx = int(process_fault_id[-1]) - 1
        return h0[tank_idx]
    return KC_GAIN * h0[sensor_idx]


def build_mixture(process_fault_id, sidx, ny, gamma, is_process, sev_nodes, onset_nodes,
                   L, Kgain=DEFAULT_KGAIN):
    model = build_model(gamma=gamma, Kgain=Kgain)
    Acl, Edelta, C, Ddelta = diff_system_matrices(model, 0.0, sidx, ny)
    if is_process:
        rec = PROCESS_FAULTS[process_fault_id]
        Bcol = rec.direction_fn(model)
        Dcol = np.zeros(C.shape[0])
    else:
        Bcol = Edelta[:, 1]
        Dcol = Ddelta[:, 1]
    anchor = _severity_anchor(process_fault_id, model, sidx, is_process)
    mus, weights = [], []
    for pct, w_s in sev_nodes:
        alpha = (pct / 100.0) * anchor
        for onset, w_o in onset_nodes:
            s = delayed_step_response(Acl, Bcol, Ccol=C, Dcol=Dcol, L=L, onset=onset)
            mus.append(alpha * s)
            weights.append(w_s * w_o)
    weights = np.array(weights)
    weights = weights / weights.sum()
    return np.array(mus), weights, Acl, C


def log_mixture_density(y, mus, weights, Sigma_inv, log_det_Sigma, ny_L):
    diffs = y[None, :] - mus
    quad = np.einsum("ki,ij,kj->k", diffs, Sigma_inv, diffs)
    log_normconst = -0.5 * (ny_L * np.log(2 * np.pi) + log_det_Sigma)
    log_comp = log_normconst - 0.5 * quad + np.log(weights)
    m = log_comp.max()
    return m + np.log(np.sum(np.exp(log_comp - m)))


def ensemble_bayes_accuracy(process_fault_id, sidx, ny, gamma, sigma_h, sigma_v,
                             L=8, n_mc=8000, seed=0, n_severity=DEFAULT_N_SEVERITY,
                             pct_lo=1.0, pct_hi=30.0, rule="gauss_legendre",
                             Kgain=DEFAULT_KGAIN):
    """Quadrature (severity, `rule`) x discrete (onset) builds a closed-form
    finite Gaussian-mixture representation of each class's output law;
    Monte Carlo evaluates the exact max-density (Bayes-optimal) decision
    between the two closed-form mixtures, with an explicit standard error."""
    rng = np.random.default_rng(seed)
    sev_nodes, onset_nodes = declared_priors(n_severity=n_severity, pct_lo=pct_lo,
                                              pct_hi=pct_hi, rule=rule)
    mus_p, w_p, Acl, C = build_mixture(process_fault_id, sidx, ny, gamma, True,
                                        sev_nodes, onset_nodes, L, Kgain)
    mus_s, w_s, _, _ = build_mixture(process_fault_id, sidx, ny, gamma, False,
                                      sev_nodes, onset_nodes, L, Kgain)
    ny_L = mus_p.shape[1]
    G = state_noise_toeplitz(Acl, C, L)
    Sigma = (sigma_h ** 2) * (G @ G.T) + (sigma_v ** 2) * np.eye(ny_L)
    Sigma_inv = np.linalg.inv(Sigma)
    _, log_det = np.linalg.slogdet(Sigma)

    correct = 0
    for _ in range(n_mc):
        cls = rng.integers(0, 2)
        if cls == 0:
            k = rng.choice(len(w_p), p=w_p)
            y = rng.multivariate_normal(mus_p[k], Sigma)
        else:
            k = rng.choice(len(w_s), p=w_s)
            y = rng.multivariate_normal(mus_s[k], Sigma)
        lp = log_mixture_density(y, mus_p, w_p, Sigma_inv, log_det, ny_L)
        ls = log_mixture_density(y, mus_s, w_s, Sigma_inv, log_det, ny_L)
        pred = 0 if lp >= ls else 1
        correct += int(pred == cls)
    acc = correct / n_mc
    se = float(np.sqrt(acc * (1 - acc) / n_mc))
    return acc, se


def true_continuous_test_set(process_fault_id, sidx, ny, gamma, sigma_h, sigma_v,
                              L=8, n_mc=8000, seed=0, pct_lo=1.0, pct_hi=30.0,
                              onsets=(0, 1, 2, 3), Kgain=DEFAULT_KGAIN):
    """Draw n_mc test points directly from the TRUE continuous-severity prior
    (severity ~ continuous Log-Uniform(pct_lo,pct_hi) in the original
    percentage variable, onset ~ discrete-uniform, exact), with NO
    quadrature approximation anywhere in this function. This fixed sample
    (same seed) is then reused by every resolution tested in
    resolution_convergence, so that resolution-to-resolution comparisons
    are common-random-number comparisons of a DECISION RULE evaluated on a
    shared test set, not comparisons of two independently-sampled Monte
    Carlo accuracy estimates -- this isolates quadrature error from Monte
    Carlo sampling noise, which independent-sample comparisons at
    realistic n_mc cannot resolve at the 1e-3 declared tolerance (checked:
    naive independent-sample SE at n_mc=2000 is ~0.008, an order of
    magnitude larger than the tolerance)."""
    rng = np.random.default_rng(seed)
    model = build_model(gamma=gamma, Kgain=Kgain)
    Acl, Edelta, C, Ddelta = diff_system_matrices(model, 0.0, sidx, ny)
    rec = PROCESS_FAULTS[process_fault_id]
    p_anchor = _severity_anchor(process_fault_id, model, sidx, True)
    s_anchor = _severity_anchor(process_fault_id, model, sidx, False)
    G = state_noise_toeplitz(Acl, C, L)
    Sigma = (sigma_h ** 2) * (G @ G.T) + (sigma_v ** 2) * np.eye(C.shape[0] * L)
    a, b = np.log(pct_lo), np.log(pct_hi)

    Y, labels = [], []
    for _ in range(n_mc):
        cls = int(rng.integers(0, 2))  # 0 = process, 1 = sensor, equal priors
        pct = float(np.exp(rng.uniform(a, b)))
        onset = int(rng.choice(onsets))
        if cls == 0:
            alpha = (pct / 100.0) * p_anchor
            Bcol, Dcol = rec.direction_fn(model), np.zeros(C.shape[0])
        else:
            alpha = (pct / 100.0) * s_anchor
            Bcol, Dcol = Edelta[:, 1], Ddelta[:, 1]
        mu = alpha * delayed_step_response(Acl, Bcol, C, Dcol, L, onset)
        y = rng.multivariate_normal(mu, Sigma)
        Y.append(y)
        labels.append(cls)
    return np.array(Y), np.array(labels), Acl, C, Sigma


def resolution_convergence(process_fault_id, sidx, ny, gamma, sigma_h, sigma_v,
                            L=8, n_mc=8000, seed=0, resolutions=RESOLUTION_SEQUENCE,
                            tol=DECLARED_CONVERGENCE_TOL, rule="gauss_legendre",
                            pct_lo=1.0, pct_hi=30.0, Kgain=DEFAULT_KGAIN):
    """Fix ONE common test set drawn from the true continuous prior
    (true_continuous_test_set, same seed for every resolution), then for
    each resolution N in `resolutions` build the N-node quadrature
    mixture's Bayes decision rule and evaluate its accuracy on that same
    fixed test set. Because the test points are identical across N, the
    resolution-to-resolution accuracy differences are a paired comparison
    (most points do not flip their decision as N changes), which is far
    tighter than comparing independent Monte Carlo draws at each N -- this
    is what makes the declared 1e-3 tolerance actually checkable. Returns
    A* vs. resolution plus the smallest resolution after which every
    subsequent estimate stays within `tol` of the final (highest-
    resolution) value."""
    Y, labels, Acl, C, Sigma = true_continuous_test_set(
        process_fault_id, sidx, ny, gamma, sigma_h, sigma_v, L=L, n_mc=n_mc,
        seed=seed, pct_lo=pct_lo, pct_hi=pct_hi, Kgain=Kgain)
    Sigma_inv = np.linalg.inv(Sigma)
    _, log_det = np.linalg.slogdet(Sigma)
    ny_L = Y.shape[1]

    rows = []
    prev_preds = None
    for n in resolutions:
        sev_nodes, onset_nodes = declared_priors(n_severity=n, pct_lo=pct_lo,
                                                  pct_hi=pct_hi, rule=rule)
        mus_p, w_p, _, _ = build_mixture(process_fault_id, sidx, ny, gamma, True,
                                          sev_nodes, onset_nodes, L, Kgain)
        mus_s, w_s, _, _ = build_mixture(process_fault_id, sidx, ny, gamma, False,
                                          sev_nodes, onset_nodes, L, Kgain)
        preds = np.empty(len(Y), dtype=int)
        for i, y in enumerate(Y):
            lp = log_mixture_density(y, mus_p, w_p, Sigma_inv, log_det, ny_L)
            ls = log_mixture_density(y, mus_s, w_s, Sigma_inv, log_det, ny_L)
            preds[i] = 0 if lp >= ls else 1
        acc = float(np.mean(preds == labels))
        se = float(np.sqrt(acc * (1 - acc) / len(Y)))
        flips_from_prev = int(np.sum(preds != prev_preds)) if prev_preds is not None else None
        rows.append(dict(n_severity=n, ensemble_astar=acc, naive_se=se,
                          decisions_flipped_since_prev_resolution=flips_from_prev))
        prev_preds = preds

    final = rows[-1]["ensemble_astar"]
    converged_at = None
    for i, row in enumerate(rows):
        if all(abs(r["ensemble_astar"] - final) < tol for r in rows[i:]):
            converged_at = row["n_severity"]
            break
    return dict(rows=rows, tol=tol, final_value=final,
                converged_at_n_severity=converged_at,
                declared_convergence_tol=tol,
                method="common fixed test set (n_mc points, true continuous prior), "
                       "paired decision-rule comparison across resolutions")
