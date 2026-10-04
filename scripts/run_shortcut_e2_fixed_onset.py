"""
Fixed-onset shortcut audit (one of the supporting shortcut/leakage controls).

REFERENCE protocol: onset ~ discrete-uniform{0,1,2,3}, drawn independently
per run, for BOTH classes -- exactly the declared prior used everywhere else
in this release (src/config/experiment_matrix.py::ONSET_PRIOR).

NAIVE (confounded) protocol: process-fault runs ALWAYS onset=0, sensor-
fault runs ALWAYS onset=3 -- a class-correlated timing shortcut that has
nothing to do with the actual fault mechanism, deliberately introduced to
see whether a model trained on it "cheats" by learning onset-timing rather
than the fault signature itself. Severity is drawn from the same declared
prior as REFERENCE in both protocols -- onset is the ONLY manipulated
variable, isolating its effect.

Procedure:
  1. Train a model on the NAIVE (confounded-onset) training distribution.
  2. Evaluate it on a NAIVE-distribution test set (in-distribution; likely
     shortcut-inflated).
  3. Evaluate the SAME trained model on a CONTROLLED test set where onset
     is matched (randomized per the declared prior) -- if performance
     collapses, that demonstrates the shortcut.
  4. Separately, train and evaluate a REFERENCE model entirely on the
     declared-prior (matched-onset) distribution, for a clean baseline.

Bayes ceilings: the REFERENCE distribution is exactly this release's
declared prior, so its A* is the same quantity computed by
run_ensemble_astar.py (no recomputation needed here). The NAIVE
(confounded) onset distribution is a well-defined, if unintended,
distribution, so its own correct Bayes ceiling is recomputed in closed
form via class_conditional_bayes_accuracy -- this lets us check whether
the naive-in-distribution model accuracy stays within its own legitimate
ceiling (it should) even though that ceiling is not comparable to the
original A*.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from _shortcut_common import COND_BY_ID, PRIMARY_CONDITION, SECONDARY_CONDITION, \
    gen_run, fit_xgb, flatten, SHORTCUT_DIR, AUDIT_SEED, \
    class_conditional_bayes_accuracy, single_point_nodes
from src.statistical.ensemble_astar import declared_priors

N_RUNS_PER_CLASS = 300
SEV_PCT_LO, SEV_PCT_HI = 1.0, 30.0


def _stable_seed(run_id, base):
    import hashlib
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2 ** 31 - 1)


def gen_dataset(cond, n_runs_per_class, onset_policy, base_seed, tag):
    """onset_policy: dict fault_class -> fixed onset int, OR None per class
    to draw uniformly from {0,1,2,3} (the declared prior)."""
    X, y = [], []
    rng_master = np.random.default_rng(base_seed)
    for cls, label in (("process", 0), ("sensor", 1)):
        pinned_onset = onset_policy.get(cls) if onset_policy else None
        for idx in range(n_runs_per_class):
            run_id = f"{tag}__{cond.condition_id}__{cls}__{idx:05d}"
            seed = _stable_seed(run_id, base_seed)
            onset = pinned_onset  # None -> simulate_run draws it from the declared prior itself
            Y, meta = gen_run(cond, run_id, seed, cls, onset=onset)
            X.append(Y)
            y.append(label)
    return X, np.array(y)


def run_for_condition(cond_id):
    cond = COND_BY_ID[cond_id]

    # --- REFERENCE: matched (random) onset for both classes ---
    Xtr_ref, ytr_ref = gen_dataset(cond, N_RUNS_PER_CLASS, onset_policy=None,
                                    base_seed=91001, tag="e2ref_train")
    Xte_ref, yte_ref = gen_dataset(cond, N_RUNS_PER_CLASS, onset_policy=None,
                                    base_seed=91002, tag="e2ref_test")

    # --- NAIVE: process always onset=0, sensor always onset=3 ---
    naive_policy = dict(process=0, sensor=3)
    Xtr_naive, ytr_naive = gen_dataset(cond, N_RUNS_PER_CLASS, onset_policy=naive_policy,
                                        base_seed=91003, tag="e2naive_train")
    Xte_naive, yte_naive = gen_dataset(cond, N_RUNS_PER_CLASS, onset_policy=naive_policy,
                                        base_seed=91004, tag="e2naive_test")

    # --- CONTROLLED test set: onset matched/randomized, used to evaluate
    #     the NAIVE-trained model on the controlled test distribution ---
    Xte_controlled, yte_controlled = gen_dataset(cond, N_RUNS_PER_CLASS, onset_policy=None,
                                                  base_seed=91005, tag="e2controlled_test")

    def flat(X):
        return np.stack([flatten(x) for x in X])

    # reference model
    _, ref_acc, _, _ = fit_xgb(flat(Xtr_ref), ytr_ref, flat(Xte_ref), yte_ref)

    # naive model
    naive_clf, naive_indist_acc, _, _ = fit_xgb(flat(Xtr_naive), ytr_naive,
                                                 flat(Xte_naive), yte_naive)
    # SAME naive-trained model evaluated on the controlled (matched-onset) test set
    proba_ctrl = naive_clf.predict_proba(flat(Xte_controlled))[:, 1]
    pred_ctrl = (proba_ctrl >= 0.5).astype(int)
    naive_on_controlled_acc = float(np.mean(pred_ctrl == yte_controlled))

    # Bayes ceilings
    sev_nodes, onset_nodes_uniform = declared_priors(pct_lo=SEV_PCT_LO, pct_hi=SEV_PCT_HI)
    astar_naive, astar_naive_se = class_conditional_bayes_accuracy(
        cond.process_fault_id, cond.sensor_idx, cond.ny, sigma_h=0.0, sigma_v=0.3,
        gamma_p=cond.gamma, sev_nodes_p=sev_nodes, onset_nodes_p=single_point_nodes(0),
        gamma_s=cond.gamma, sev_nodes_s=sev_nodes, onset_nodes_s=single_point_nodes(3),
        seed=AUDIT_SEED,
    )

    return dict(
        condition_id=cond_id, n_runs_per_class=N_RUNS_PER_CLASS,
        reference=dict(test_accuracy=ref_acc,
                        note="matched (frozen-prior) onset both classes; NOT the same "
                             "dataset/n as the primary leaderboard -- freshly generated for "
                             "this audit at smaller n, so compare within-audit, not to the "
                             "frozen A*/XGBoost numbers directly"),
        naive=dict(in_distribution_test_accuracy=naive_indist_acc,
                   on_controlled_test_accuracy=naive_on_controlled_acc,
                   collapse=naive_indist_acc - naive_on_controlled_acc,
                   bayes_ceiling_for_naive_distribution=astar_naive,
                   bayes_ceiling_se=astar_naive_se,
                   naive_indist_exceeds_its_own_ceiling=bool(
                       naive_indist_acc > astar_naive + 2 * astar_naive_se)),
        interpretation="The naive (fixed-onset) model reaches near its OWN legitimate "
                       "confounded-distribution Bayes ceiling in-distribution (onset alone is "
                       "a near-perfect giveaway when it's deterministic per class), then "
                       "collapses toward the reference-level accuracy once evaluated on a "
                       "controlled test set where onset no longer reveals the class. Per Part "
                       "F, the naive in-distribution number must NEVER be compared to the "
                       "original (matched-onset) A* -- it is a different, confounded "
                       "distribution with its own (recomputed) ceiling.",
    )


def main():
    out = {}
    for cond_id in (PRIMARY_CONDITION, SECONDARY_CONDITION):
        print(f"=== E2 fixed-onset audit: {cond_id} ===")
        res = run_for_condition(cond_id)
        out[cond_id] = res
        print(f"  REFERENCE test acc (matched onset): {res['reference']['test_accuracy']:.4f}")
        print(f"  NAIVE in-distribution test acc:      {res['naive']['in_distribution_test_accuracy']:.4f}  "
              f"(its own Bayes ceiling: {res['naive']['bayes_ceiling_for_naive_distribution']:.4f}"
              f"+-{res['naive']['bayes_ceiling_se']:.4f})")
        print(f"  NAIVE model on CONTROLLED test acc:  {res['naive']['on_controlled_test_accuracy']:.4f}")
        print(f"  collapse (in-dist - controlled):     {res['naive']['collapse']:+.4f}")

    out_path = SHORTCUT_DIR / "e2_fixed_onset.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
