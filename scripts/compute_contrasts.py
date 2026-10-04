"""
Recomputes the six headline severity-confounding contrasts directly from
the released result table and checks them against the values reported in
the paper.

A_MM, A_MC, A_CM, A_CC are the four cells of the matched/confounded
train-test intervention matrix for one (condition, model) pair:
  A_MM: matched train,    matched test
  A_MC: matched train,    confounded test
  A_CM: confounded train, matched test
  A_CC: confounded train, confounded test

The six contrasts, each averaged over all 27 (condition, model) rows:
  competence_loss      = A_MM - A_CM   (cost of confounded training on the
                                         matched task)
  confounded_test_ease = A_MC - A_MM   (how much easier the confounded
                                         test distribution is for a
                                         matched-trained model)
  factorial_interaction = A_CC - A_CM - A_MC + A_MM
                                        (the train-by-test interaction;
                                         this is the quantity to report as
                                         "the factorial interaction", not
                                         A_CC - A_MC)
  retained_confound_benefit = A_CC - A_MC
                                        (simple training-arm effect on the
                                         confounded test; this is a
                                         different quantity from the
                                         factorial interaction above and
                                         the two must not be conflated)
  benchmark_inflation  = A_CC - A_MM   (apparent accuracy gain from using
                                         a conventional confounded
                                         benchmark instead of the matched
                                         reference)
  direct_collapse      = A_CC - A_CM   (same confound-trained model,
                                         confounded vs matched test)

Usage:
    python scripts/compute_contrasts.py
    python scripts/compute_contrasts.py --csv path/to/other_table.csv
"""
import argparse
import csv
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "severity_matrix_table.csv"

# Reference means reported in the paper, in percentage points (pp).
REPORTED_PP = {
    "competence_loss": 16.2716,
    "confounded_test_ease": 6.0206,
    "factorial_interaction": 24.0000,
    "retained_confound_benefit": 7.7284,
    "benchmark_inflation": 13.7490,
    "direct_collapse": 30.0206,
}
TOLERANCE_PP = 0.01


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [
            {
                "condition": row["condition"],
                "model": row["model"],
                "A_MM": float(row["A_MM"]),
                "A_MC": float(row["A_MC"]),
                "A_CM": float(row["A_CM"]),
                "A_CC": float(row["A_CC"]),
            }
            for row in reader
        ]


def compute_contrasts(rows):
    def col(name):
        return [row[name] for row in rows]

    mm, mc, cm, cc = col("A_MM"), col("A_MC"), col("A_CM"), col("A_CC")
    n = len(rows)

    competence_loss = [mm[i] - cm[i] for i in range(n)]
    confounded_test_ease = [mc[i] - mm[i] for i in range(n)]
    factorial_interaction = [cc[i] - cm[i] - mc[i] + mm[i] for i in range(n)]
    retained_confound_benefit = [cc[i] - mc[i] for i in range(n)]
    benchmark_inflation = [cc[i] - mm[i] for i in range(n)]
    direct_collapse = [cc[i] - cm[i] for i in range(n)]

    return {
        "competence_loss": competence_loss,
        "confounded_test_ease": confounded_test_ease,
        "factorial_interaction": factorial_interaction,
        "retained_confound_benefit": retained_confound_benefit,
        "benchmark_inflation": benchmark_inflation,
        "direct_collapse": direct_collapse,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                         help="path to a severity_matrix_table.csv-shaped file")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    print(f"loaded {len(rows)} (condition, model) rows from {args.csv}")

    contrasts = compute_contrasts(rows)

    print()
    print(f"{'contrast':28s} {'mean (pp)':>12s} {'reported (pp)':>14s} {'min':>8s} {'max':>8s} {'n positive':>11s}")
    all_ok = True
    for name, values in contrasts.items():
        mean_pp = mean(values) * 100
        reported_pp = REPORTED_PP[name]
        ok = abs(mean_pp - reported_pp) <= TOLERANCE_PP
        all_ok = all_ok and ok
        n_positive = sum(1 for v in values if v > 0)
        flag = "" if ok else "  MISMATCH"
        print(f"{name:28s} {mean_pp:12.4f} {reported_pp:14.4f} {min(values)*100:8.2f} "
              f"{max(values)*100:8.2f} {n_positive:4d}/{len(values):<4d}{flag}")

    print()
    if all_ok:
        print("All six contrasts match the values reported in the paper "
              f"within {TOLERANCE_PP} pp.")
    else:
        print("At least one contrast did not match. Check the input table.")
        sys.exit(1)


if __name__ == "__main__":
    main()
