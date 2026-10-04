"""
Operating-regime (gamma) confound audit (one of the supporting
shortcut/leakage controls).

NAIVE (confounded) protocol: process-fault runs simulated ALWAYS at
gamma=0.05 (this condition's worst-case-for-ambiguity operating
point), sensor-fault runs simulated ALWAYS at gamma=0.95 (the opposite
extreme of the 19-point gamma grid used throughout this release) -- a
class-correlated operating-REGIME shortcut. gamma is NEVER given to any
model as an explicit input feature anywhere in this release (only the
observed trajectory Y is model-visible) -- the audit question is whether
Y's SHAPE leaks which regime generated it, not whether gamma itself is a
feature.

REFERENCE / regime-BALANCED protocol: for both classes, gamma is drawn
per run uniformly from {0.05, 0.95} (the same two extreme regimes used in
the naive protocol), independent of class label -- so the marginal
regime distribution is IDENTICAL between classes, only the fault
mechanism differs.

This script generates the data and reports whatever it finds, including
if the manipulation turns out uninformative or numerically unstable --
no result is forced.

Bayes ceiling: the NAIVE (confounded) protocol uses SINGLE fixed gamma
values per class (0.05 for process, 0.95 for sensor), so its exact Bayes
ceiling is recomputed via class_conditional_bayes_accuracy, the same
mechanism used for the other shortcut audits. The REFERENCE/balanced
protocol mixes TWO gamma regimes per class with equal probability --
computing its exact mixture-of-mixtures ceiling in closed form would be a
real extension of the existing machinery, and is out of scope here; we
explicitly do not claim an exact ceiling for the balanced
protocol here and restrict interpretation to the empirical accuracy
comparison (naive in-distribution vs naive-trained-model on balanced
data), which is what actually answers the audit's question (does Y leak
regime) without needing that ceiling.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shortcut_common import COND_BY_ID, PRIMARY_CONDITION, gen_run, fit_xgb, flatten, \
    SHORTCUT_DIR, AUDIT_SEED, class_conditional_bayes_accuracy
from src.statistical.ensemble_astar import declared_priors

N_RUNS_PER_CLASS = 300
GAMMA_LOW, GAMMA_HIGH = 0.05, 0.95


def _stable_seed(run_id, base):
    import hashlib
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2 ** 31 - 1)


def gen_dataset(cond, n_runs_per_class, gamma_policy, base_seed, tag):
    """gamma_policy: dict fault_class -> fixed gamma, OR "balanced" to draw
    from {GAMMA_LOW, GAMMA_HIGH} uniformly per run regardless of class."""
    X, y = [], []
    for cls, label in (("process", 0), ("sensor", 1)):
        rng = np.random.default_rng(base_seed + (0 if cls == "process" else 1))
        for idx in range(n_runs_per_class):
            run_id = f"{tag}__{cond.condition_id}__{cls}__{idx:05d}"
            seed = _stable_seed(run_id, base_seed)
            if gamma_policy == "balanced":
                gamma_override = float(rng.choice([GAMMA_LOW, GAMMA_HIGH]))
            else:
                gamma_override = gamma_policy[cls]
            Y, meta = gen_run(cond, run_id, seed, cls, gamma_override=gamma_override)
            X.append(Y)
            y.append(label)
    return X, np.array(y)


def run_for_condition(cond_id):
    cond = COND_BY_ID[cond_id]

    naive_policy = dict(process=GAMMA_LOW, sensor=GAMMA_HIGH)
    Xtr_naive, ytr_naive = gen_dataset(cond, N_RUNS_PER_CLASS, naive_policy, 93001, "e4naive_train")
    Xte_naive, yte_naive = gen_dataset(cond, N_RUNS_PER_CLASS, naive_policy, 93002, "e4naive_test")

    Xte_balanced, yte_balanced = gen_dataset(cond, N_RUNS_PER_CLASS, "balanced", 93003,
                                              "e4balanced_test")
    Xtr_ref, ytr_ref = gen_dataset(cond, N_RUNS_PER_CLASS, "balanced", 93004, "e4ref_train")

    def flat(X):
        return np.stack([flatten(x) for x in X])

    naive_clf, naive_indist_acc, _, _ = fit_xgb(flat(Xtr_naive), ytr_naive,
                                                 flat(Xte_naive), yte_naive)
    proba_bal = naive_clf.predict_proba(flat(Xte_balanced))[:, 1]
    pred_bal = (proba_bal >= 0.5).astype(int)
    naive_on_balanced_acc = float(np.mean(pred_bal == yte_balanced))

    _, ref_acc, _, _ = fit_xgb(flat(Xtr_ref), ytr_ref, flat(Xte_balanced), yte_balanced)

    sev_nodes, onset_nodes = declared_priors(pct_lo=1.0, pct_hi=30.0)
    astar_naive, astar_naive_se = class_conditional_bayes_accuracy(
        cond.process_fault_id, cond.sensor_idx, cond.ny, sigma_h=0.0, sigma_v=0.3,
        gamma_p=GAMMA_LOW, sev_nodes_p=sev_nodes, onset_nodes_p=onset_nodes,
        gamma_s=GAMMA_HIGH, sev_nodes_s=sev_nodes, onset_nodes_s=onset_nodes,
        seed=AUDIT_SEED,
    )

    return dict(
        condition_id=cond_id, n_runs_per_class=N_RUNS_PER_CLASS,
        gamma_low=GAMMA_LOW, gamma_high=GAMMA_HIGH,
        naive=dict(in_distribution_test_accuracy=naive_indist_acc,
                   on_balanced_test_accuracy=naive_on_balanced_acc,
                   collapse=naive_indist_acc - naive_on_balanced_acc,
                   bayes_ceiling_for_naive_distribution=astar_naive,
                   bayes_ceiling_se=astar_naive_se,
                   naive_indist_exceeds_its_own_ceiling=bool(
                       naive_indist_acc > astar_naive + 2 * astar_naive_se)),
        reference_balanced=dict(test_accuracy=ref_acc,
                                 note="trained AND tested on the regime-balanced distribution; "
                                      "exact Bayes ceiling for this 2-regime mixture NOT "
                                      "recomputed (out of scope, see module docstring), so only the empirical "
                                      "in-distribution-vs-balanced comparison is interpreted"),
        interpretation="gamma is never a model input feature; any collapse from naive "
                       "in-distribution accuracy to accuracy-on-balanced-regime data would "
                       "indicate the trajectory SHAPE itself leaks regime, which a model can "
                       "exploit as an unintended shortcut correlated with class.",
    )


def main():
    out = {}
    cond_id = PRIMARY_CONDITION
    print(f"=== E4 regime-confound audit: {cond_id} ===")
    try:
        res = run_for_condition(cond_id)
    except Exception as e:
        print(f"  E4 manipulation produced an error unrelated to the intended question: {e}")
        print("  Bayes ceiling not recomputed for this mixture; omitting rather than forcing a result.")
        out[cond_id] = dict(condition_id=cond_id, omitted=True, reason=str(e))
        out_path = SHORTCUT_DIR / "e4_regime_confound.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"saved {out_path} (omitted)")
        return

    out[cond_id] = res
    print(f"  NAIVE in-distribution test acc:  {res['naive']['in_distribution_test_accuracy']:.4f}  "
          f"(its own Bayes ceiling: {res['naive']['bayes_ceiling_for_naive_distribution']:.4f}"
          f"+-{res['naive']['bayes_ceiling_se']:.4f})")
    print(f"  NAIVE model on BALANCED test acc: {res['naive']['on_balanced_test_accuracy']:.4f}")
    print(f"  collapse (in-dist - balanced):    {res['naive']['collapse']:+.4f}")
    print(f"  REFERENCE (trained+tested on balanced): {res['reference_balanced']['test_accuracy']:.4f}")

    out_path = SHORTCUT_DIR / "e4_regime_confound.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
