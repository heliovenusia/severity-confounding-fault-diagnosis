"""
Window-leakage audit (one of the supporting shortcut/leakage controls).

REFERENCE (clean) protocol: one L=8 window per run (the standard protocol
used everywhere else in this release), split RUN-LEVEL (every run
assigned whole to train or test).

NAIVE (deliberately bad) protocol: simulate each run at L=16, slice 5
overlapping L=8 windows per run (offsets 0,2,4,6,8, stride=2), pool ALL
windows from ALL runs, then split WINDOW-LEVEL at random -- so windows
from the same underlying L=16 trajectory can land in both train and test.
This is a textbook data-leakage protocol, included here only to quantify
how much apparent accuracy it manufactures. It is deliberately a bad
protocol, used only to quantify leakage.

Both protocols are evaluated with XGBoost and TCN, on two representative
conditions (pump1_h1_partial, leak_tank3_h1_partial). Explicit run-ID
overlap between the naive protocol's train and test windows is recorded.

Leaked windows violate the independent-test-set assumption A* is built
on, not a distributional shift -- so the correct statement is
"the naive protocol is not comparable to the clean-distribution Bayes
reference," not a recomputed ceiling. Both protocols use the same
declared priors (severity, onset) as every other primary result, so this is
a pure PROCEDURAL manipulation, isolated from any distributional one.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from _shortcut_common import COND_BY_ID, PRIMARY_CONDITION, SECONDARY_CONDITION, \
    gen_run, fit_xgb, flatten, SHORTCUT_DIR, AUDIT_SEED
from src.deep import TCNClassifier, train_binary_classifier, tcn_input

N_RUNS_PER_CLASS = 300
L_LONG = 16
L_WINDOW = 8
OFFSETS = (0, 2, 4, 6, 8)
TRAIN_FRAC = 0.7


def _stable_seed(run_id, base):
    import hashlib
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2 ** 31 - 1)


def generate_long_runs(cond, n_runs_per_class, base_seed=90001):
    """Simulate each run at L=16; return dict run_id -> (Y16, fault_class)."""
    out = {}
    for cls in ("process", "sensor"):
        for idx in range(n_runs_per_class):
            run_id = f"e1__{cond.condition_id}__{cls}__{idx:05d}"
            seed = _stable_seed(run_id, base_seed)
            Y, meta = gen_run(cond, run_id, seed, cls, L=L_LONG)
            out[run_id] = dict(Y=Y, cls=cls, label=0 if cls == "process" else 1)
    return out


def reference_protocol(long_runs, seed=AUDIT_SEED):
    """One window (offset 0) per run; RUN-LEVEL train/test split."""
    run_ids = sorted(long_runs.keys())
    rng = np.random.default_rng(seed)
    ids = list(run_ids)
    rng.shuffle(ids)
    n_train = int(round(TRAIN_FRAC * len(ids)))
    train_ids, test_ids = ids[:n_train], ids[n_train:]

    def windows(ids_subset):
        X, y, rid = [], [], []
        for rid_ in ids_subset:
            Y16 = long_runs[rid_]["Y"]
            X.append(Y16[0:L_WINDOW])
            y.append(long_runs[rid_]["label"])
            rid.append(rid_)
        return X, np.array(y), rid

    Xtr, ytr, ridtr = windows(train_ids)
    Xte, yte, ridte = windows(test_ids)
    return Xtr, ytr, ridtr, Xte, yte, ridte


def naive_protocol(long_runs, seed=AUDIT_SEED):
    """5 overlapping windows per run, pooled, WINDOW-LEVEL random split
    (ignoring run identity) -- so the same run's windows can appear on
    both sides."""
    all_windows = []  # (X, y, run_id)
    for rid_, rec in long_runs.items():
        Y16 = rec["Y"]
        for off in OFFSETS:
            all_windows.append((Y16[off:off + L_WINDOW], rec["label"], rid_))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(all_windows))
    rng.shuffle(idx)
    n_train = int(round(TRAIN_FRAC * len(idx)))
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    def build(subset_idx):
        X = [all_windows[i][0] for i in subset_idx]
        y = np.array([all_windows[i][1] for i in subset_idx])
        rid = [all_windows[i][2] for i in subset_idx]
        return X, y, rid

    Xtr, ytr, ridtr = build(train_idx)
    Xte, yte, ridte = build(test_idx)
    return Xtr, ytr, ridtr, Xte, yte, ridte


def train_eval_xgb(Xtr, ytr, Xte, yte):
    Xtr_flat = np.stack([flatten(x) for x in Xtr])
    Xte_flat = np.stack([flatten(x) for x in Xte])
    _, acc, _, _ = fit_xgb(Xtr_flat, ytr, Xte_flat, yte)
    return acc


def train_eval_tcn(Xtr, ytr, Xte, yte, ny, seed=AUDIT_SEED):
    Xtr_t = torch.tensor(np.stack([tcn_input(x) for x in Xtr]), dtype=torch.float32)
    Xte_t = torch.tensor(np.stack([tcn_input(x) for x in Xte]), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.float32)
    # use test tensors as the val set too for this quick audit fit (early
    # stopping on a held-out slice of train instead would also be valid;
    # this is a bounded audit, not the primary leaderboard -- to avoid
    # leaking the audit's own test set into early stopping, carve a val
    # split OUT of train instead)
    n_val = max(1, int(0.15 * len(ytr_t)))
    Xva_t, yva_t = Xtr_t[:n_val], ytr_t[:n_val]
    Xtr2_t, ytr2_t = Xtr_t[n_val:], ytr_t[n_val:]
    model = TCNClassifier(ny=ny)
    res = train_binary_classifier(model, Xtr2_t, ytr2_t, Xva_t, yva_t, Xte_t, yte_t,
                                   seed=seed, max_epochs=150, patience=15)
    acc = float(np.mean(np.array(res["test_preds"]) == np.array(res["test_labels"])))
    return acc


def run_for_condition(cond_id):
    cond = COND_BY_ID[cond_id]
    long_runs = generate_long_runs(cond, N_RUNS_PER_CLASS)

    Xtr_r, ytr_r, ridtr_r, Xte_r, yte_r, ridte_r = reference_protocol(long_runs)
    Xtr_n, ytr_n, ridtr_n, Xte_n, yte_n, ridte_n = naive_protocol(long_runs)

    overlap_run_ids = set(ridtr_n) & set(ridte_n)
    overlap_frac_of_test = len(overlap_run_ids) / len(set(ridte_n))

    ref_xgb_acc = train_eval_xgb(Xtr_r, ytr_r, Xte_r, yte_r)
    naive_xgb_acc = train_eval_xgb(Xtr_n, ytr_n, Xte_n, yte_n)
    ref_tcn_acc = train_eval_tcn(Xtr_r, ytr_r, Xte_r, yte_r, ny=cond.ny)
    naive_tcn_acc = train_eval_tcn(Xtr_n, ytr_n, Xte_n, yte_n, ny=cond.ny)

    return dict(
        condition_id=cond_id,
        n_runs_per_class=N_RUNS_PER_CLASS,
        reference=dict(n_train=len(ytr_r), n_test=len(yte_r),
                        xgb_accuracy=ref_xgb_acc, tcn_accuracy=ref_tcn_acc),
        naive=dict(n_train=len(ytr_n), n_test=len(yte_n),
                    xgb_accuracy=naive_xgb_acc, tcn_accuracy=naive_tcn_acc,
                    run_id_overlap_train_test=len(overlap_run_ids),
                    run_id_overlap_fraction_of_test_run_ids=overlap_frac_of_test),
        inflation=dict(xgb=naive_xgb_acc - ref_xgb_acc, tcn=naive_tcn_acc - ref_tcn_acc),
        interpretation="NAIVE protocol accuracy is NOT comparable to the clean-distribution "
                       "Bayes reference (A*) for this condition -- overlapping windows from "
                       "the same underlying trajectory in both train and test violate the "
                       "independent-test-set assumption A* is built on. We do not "
                       "recompute a 'ceiling' for this protocol; the protocol itself is invalid.",
    )


def main():
    out = {}
    for cond_id in (PRIMARY_CONDITION, SECONDARY_CONDITION):
        print(f"=== E1 window-leakage audit: {cond_id} ===")
        res = run_for_condition(cond_id)
        out[cond_id] = res
        print(f"  REFERENCE (run-level split): XGB={res['reference']['xgb_accuracy']:.4f}  "
              f"TCN={res['reference']['tcn_accuracy']:.4f}")
        print(f"  NAIVE (window-level split):  XGB={res['naive']['xgb_accuracy']:.4f}  "
              f"TCN={res['naive']['tcn_accuracy']:.4f}  "
              f"run-ID overlap frac(test)={res['naive']['run_id_overlap_fraction_of_test_run_ids']:.3f}")
        print(f"  apparent inflation: XGB={res['inflation']['xgb']:+.4f}  "
              f"TCN={res['inflation']['tcn']:+.4f}")

    out_path = SHORTCUT_DIR / "e1_window_leakage.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
