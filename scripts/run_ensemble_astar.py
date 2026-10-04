"""
Ensemble Bayes accuracy (declared log-uniform severity x discrete-uniform
onset priors) for every experiment-matrix condition, plus the
resolution-convergence check (Gauss-Legendre quadrature, common fixed
test-set design) that determines which n_severity is actually adequate,
using a controlled, convergence-checked quadrature in place of an
uncontrolled 7-node approximation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EXPERIMENT_MATRIX, L_PRIMARY, NOISE_REFERENCE, SEVERITY_PRIOR
from src.statistical import ensemble_bayes_accuracy, resolution_convergence


def main():
    n_severity_frozen = SEVERITY_PRIOR["n_severity_frozen"]
    results = {}

    print("=" * 100)
    print(f"Resolution convergence check (Gauss-Legendre, common fixed test set, n_mc=6000)")
    print("=" * 100)
    convergence_results = {}
    for cond in EXPERIMENT_MATRIX:
        conv = resolution_convergence(cond.process_fault_id, cond.sensor_idx, cond.ny, cond.gamma,
                                       sigma_h=NOISE_REFERENCE["sigma_h"], sigma_v=NOISE_REFERENCE["sigma_v"],
                                       L=L_PRIMARY, n_mc=6000, seed=0)
        convergence_results[cond.condition_id] = conv
        row_strs = [f"{r['n_severity']}:{r['ensemble_astar']:.4f}" for r in conv["rows"]]
        print(f"{cond.condition_id:28s} converged_at_n_severity={conv['converged_at_n_severity']:>4} "
              f"final_A*={conv['final_value']:.4f}  [{', '.join(row_strs)}]")

    print()
    print("=" * 100)
    print(f"Ensemble A* at the operating resolution n_severity={n_severity_frozen}, "
          f"sigma_v={NOISE_REFERENCE['sigma_v']}V")
    print("=" * 100)
    for cond in EXPERIMENT_MATRIX:
        acc, se = ensemble_bayes_accuracy(cond.process_fault_id, cond.sensor_idx, cond.ny, cond.gamma,
                                           sigma_h=NOISE_REFERENCE["sigma_h"], sigma_v=NOISE_REFERENCE["sigma_v"],
                                           L=L_PRIMARY, n_mc=8000, seed=0, n_severity=n_severity_frozen)
        results[cond.condition_id] = dict(role=cond.role, verdict=cond.expected_verdict,
                                           mechanism=cond.expected_mechanism,
                                           ensemble_astar=acc, mc_se=se,
                                           convergence=convergence_results[cond.condition_id])
        print(f"{cond.condition_id:28s} [{cond.role:18s} {cond.expected_verdict:10s}] "
              f"ensemble A* = {acc:.4f} +/- {se:.4f}")

    out_path = Path(__file__).resolve().parents[1] / "results" / "ensemble_astar.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved {out_path}")
    return results


if __name__ == "__main__":
    main()
