"""
Shared helpers for the supporting shortcut/leakage audit scripts (window
leakage, fixed onset, regime confound). These scripts need direct access
to simulate_run() to generate naive/deliberately-bad protocol variants
and controlled/matched-distribution variants that are not present in any
pre-generated dataset.

Representative conditions: pump1_h1_partial (AMBIGUOUS, A*=0.8177, plenty
of headroom below 1.0 for an inflation effect to be visible) is the
primary condition used across the audits; leak_tank3_h1_partial
(AMBIGUOUS, A*=0.8650) is added as a second condition for some audits, to
check the effect is not a one-off artifact of a single condition. All
audit datasets are generated fresh via simulate_run() with deterministic,
logged seeds -- never reusing the main severity-matrix experiment's
run_ids or trajectories, so there is no risk of interfering with the
primary results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from src.config import EXPERIMENT_MATRIX
from src.simulation import simulate_run

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "results"
SHORTCUT_DIR = ART / "shortcut_audit"
SHORTCUT_DIR.mkdir(parents=True, exist_ok=True)

COND_BY_ID = {c.condition_id: c for c in EXPERIMENT_MATRIX}
PRIMARY_CONDITION = "pump1_h1_partial"
SECONDARY_CONDITION = "leak_tank3_h1_partial"

# One fixed, modest XGBoost config used throughout the audit (not the
# grid search used elsewhere for model selection) -- the
# audit's question is protocol validity, not squeezing out the last bit of
# XGBoost accuracy, and a single fixed config keeps every audit comparison
# a controlled, deterministic, apples-to-apples one.
AUDIT_XGB_HPARAMS = dict(max_depth=3, learning_rate=0.10, n_estimators=300,
                          subsample=0.8, colsample_bytree=0.8)
AUDIT_SEED = 42


def fit_xgb(X_train, y_train, X_test, y_test, hparams=AUDIT_XGB_HPARAMS, seed=AUDIT_SEED):
    clf = XGBClassifier(max_depth=hparams["max_depth"], learning_rate=hparams["learning_rate"],
                         n_estimators=hparams["n_estimators"], subsample=hparams["subsample"],
                         colsample_bytree=hparams["colsample_bytree"], eval_metric="logloss",
                         random_state=seed, n_jobs=4, verbosity=0)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, pred))
    return clf, acc, proba, pred


def gen_run(cond, run_id, seed, fault_class, severity_pct=None, onset=None, L=8,
            sigma_h=0.0, sigma_v=0.3, gamma_override=None):
    """Thin wrapper around simulate_run that optionally overrides gamma
    (for E4) by substituting a modified condition object -- everything
    else is passed straight through. Returns (Y, meta)."""
    if gamma_override is not None:
        from dataclasses import replace
        cond = replace(cond, gamma=gamma_override)
    return simulate_run(cond, run_id=run_id, seed=seed, fault_class=fault_class,
                         severity_pct=severity_pct, onset=onset, L=L,
                         sigma_h=sigma_h, sigma_v=sigma_v)


def flatten(Y):
    return np.asarray(Y).reshape(-1)


def class_conditional_bayes_accuracy(process_fault_id, sidx, ny, sigma_h, sigma_v,
                                      gamma_p, sev_nodes_p, onset_nodes_p,
                                      gamma_s, sev_nodes_s, onset_nodes_s,
                                      L=8, n_mc=6000, seed=0):
    """The TRUE Bayes-optimal decision accuracy when the two classes are
    allowed DIFFERENT (gamma, severity-prior, onset-prior) generating
    distributions -- i.e. a legitimate closed-form Bayes ceiling for a
    confounded/asymmetric scenario, recomputed for the altered
    distribution rather than reused from the matched case, used by the
    shortcut audits below.

    This only works in closed form because this codebase's declared
    NOISE_REFERENCE always has sigma_h=0 (see src/config/experiment_matrix.py),
    so the observation covariance Sigma = sigma_v^2 * I is IDENTICAL
    regardless of gamma/Acl -- the state-noise term that would otherwise
    make Sigma class-dependent vanishes identically. This is a genuine
    fact about the model, not an audit-specific assumption; the guard below
    makes that assumption explicit and fails loudly rather than silently if
    sigma_h is ever nonzero when this is called.
    """
    from src.statistical.ensemble_astar import build_mixture, log_mixture_density
    if sigma_h != 0.0:
        raise NotImplementedError(
            "class_conditional_bayes_accuracy's shared-Sigma shortcut requires sigma_h=0 "
            "(true throughout this codebase's NOISE_REFERENCE); got sigma_h="
            f"{sigma_h} -- would need a class-dependent Sigma, not implemented here.")
    rng = np.random.default_rng(seed)
    mus_p, w_p, _, _ = build_mixture(process_fault_id, sidx, ny, gamma_p, True,
                                      sev_nodes_p, onset_nodes_p, L)
    mus_s, w_s, _, _ = build_mixture(process_fault_id, sidx, ny, gamma_s, False,
                                      sev_nodes_s, onset_nodes_s, L)
    ny_L = mus_p.shape[1]
    Sigma = (sigma_v ** 2) * np.eye(ny_L)
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


def single_point_nodes(value):
    """A degenerate (Dirac) 'prior' with all weight on one value -- used to
    build the NAIVE fixed-onset / fixed-severity shortcut distributions in
    closed form for class_conditional_bayes_accuracy."""
    return [(value, 1.0)]
