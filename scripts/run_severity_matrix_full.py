"""
Extended 2x2 severity-confounding intervention matrix (A_MM, A_MC, A_CM,
A_CC) across all 9 Layer-A conditions, three model classes (GLR,
XGBoost, TCN x3 seeds). This is the core benchmark-validity-auditing
experiment for the paper. It reuses simulate_run, the GLR classical model, the
existing XGBoost audit config, and the existing TCN training harness
unchanged -- only the dataset-generation and matrix-bookkeeping logic here
is new.

Confound definition (process severity Log-Uniform(1,8)%, sensor severity
Log-Uniform(15,30)%; onset always drawn from the declared matched prior in
both arms -- severity range is the only manipulated variable).

Design: per condition, ONE matched test set and ONE confounded test set
are generated (not regenerated per training arm), so every matrix cell
contrast is a paired comparison on identical test runs. Two models are
trained per model-class (one on matched train, one on confounded train)
and each is evaluated on both test sets, yielding all four matrix cells
per (condition, model-class).
"""
import json
import sys
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from src.config import EXPERIMENT_MATRIX, NOISE_REFERENCE
from src.simulation import simulate_run
from src.classical import build_glr_model, glr_scores, fit_threshold
from src.deep import TCNClassifier, train_binary_classifier, tcn_input

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "results" / "severity_matrix_full"
ART.mkdir(parents=True, exist_ok=True)

COND_BY_ID = {c.condition_id: c for c in EXPERIMENT_MATRIX}
N_RUNS_PER_CLASS = 300
REF_LO, REF_HI = 1.0, 30.0
NAIVE_PROCESS_LO, NAIVE_PROCESS_HI = 1.0, 8.0
NAIVE_SENSOR_LO, NAIVE_SENSOR_HI = 15.0, 30.0
SEEDS_TCN = (42, 43, 44)
AUDIT_XGB_HPARAMS = dict(max_depth=3, learning_rate=0.10, n_estimators=300,
                          subsample=0.8, colsample_bytree=0.8)
CONFOUND_POLICY = dict(process=(NAIVE_PROCESS_LO, NAIVE_PROCESS_HI),
                        sensor=(NAIVE_SENSOR_LO, NAIVE_SENSOR_HI))


def stable_seed(run_id, base):
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2 ** 31 - 1)


def gen_dataset(cond, n_per_class, policy, base_seed, tag):
    X, y = [], []
    for cls, label in (("process", 0), ("sensor", 1)):
        rng = np.random.default_rng(base_seed + (0 if cls == "process" else 1))
        lo, hi = policy.get(cls) if policy else (REF_LO, REF_HI)
        a, b = np.log(lo), np.log(hi)
        for idx in range(n_per_class):
            run_id = f"{tag}__{cond.condition_id}__{cls}__{idx:05d}"
            seed = stable_seed(run_id, base_seed)
            sev = float(np.exp(rng.uniform(a, b)))
            Y, meta = simulate_run(cond, run_id=run_id, seed=seed, fault_class=cls,
                                    severity_pct=sev, sigma_h=NOISE_REFERENCE["sigma_h"],
                                    sigma_v=NOISE_REFERENCE["sigma_v"])
            X.append(Y)
            y.append(label)
    return X, np.array(y)


def flatten_all(X):
    return np.stack([np.asarray(x).reshape(-1) for x in X])


def binom_se(acc, n):
    return float(np.sqrt(max(acc * (1 - acc), 1e-12) / n))


def split_val(T, y, frac=0.15, seed=0):
    n = T.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return T[tr_idx], y[tr_idx], T[val_idx], y[val_idx]


def run_condition(cond_id, seed_base_offset):
    cond = COND_BY_ID[cond_id]
    b = seed_base_offset
    Xtr_M, ytr_M = gen_dataset(cond, N_RUNS_PER_CLASS, None, b + 1, "train_matched")
    Xtr_C, ytr_C = gen_dataset(cond, N_RUNS_PER_CLASS, CONFOUND_POLICY, b + 2, "train_confound")
    Xte_M, yte_M = gen_dataset(cond, N_RUNS_PER_CLASS, None, b + 3, "test_matched")
    Xte_C, yte_C = gen_dataset(cond, N_RUNS_PER_CLASS, CONFOUND_POLICY, b + 4, "test_confound")

    n_test_M, n_test_C = len(yte_M), len(yte_C)
    result = dict(condition_id=cond_id, n_runs_per_class=N_RUNS_PER_CLASS,
                  n_test_matched=n_test_M, n_test_confound=n_test_C)

    # ---------------- XGBoost ----------------
    Ftr_M, Ftr_C = flatten_all(Xtr_M), flatten_all(Xtr_C)
    Fte_M, Fte_C = flatten_all(Xte_M), flatten_all(Xte_C)
    clf_M = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=4, verbosity=0,
                           **AUDIT_XGB_HPARAMS)
    clf_M.fit(Ftr_M, ytr_M)
    clf_C = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=4, verbosity=0,
                           **AUDIT_XGB_HPARAMS)
    clf_C.fit(Ftr_C, ytr_C)

    def xgb_acc(clf, F, y):
        pred = (clf.predict_proba(F)[:, 1] >= 0.5).astype(int)
        return float(accuracy_score(y, pred))

    xgb = dict(A_MM=xgb_acc(clf_M, Fte_M, yte_M), A_MC=xgb_acc(clf_M, Fte_C, yte_C),
               A_CM=xgb_acc(clf_C, Fte_M, yte_M), A_CC=xgb_acc(clf_C, Fte_C, yte_C))
    for k in list(xgb.keys()):
        n = n_test_M if k in ("A_MM", "A_CM") else n_test_C
        xgb[k + "_se"] = binom_se(xgb[k], n)
    result["xgboost"] = xgb

    # ---------------- GLR ----------------
    glr_model = build_glr_model(cond, sigma_h=NOISE_REFERENCE["sigma_h"],
                                 sigma_v=NOISE_REFERENCE["sigma_v"])

    def glr_stats(X):
        return np.array([glr_scores(np.asarray(x).reshape(-1), glr_model)[2] for x in X])

    stat_tr_M, stat_tr_C = glr_stats(Xtr_M), glr_stats(Xtr_C)
    stat_te_M, stat_te_C = glr_stats(Xte_M), glr_stats(Xte_C)
    thr_M, balacc_fit_M = fit_threshold(stat_tr_M, ytr_M)
    thr_C, balacc_fit_C = fit_threshold(stat_tr_C, ytr_C)

    def glr_acc(stat, y, thr):
        pred = (stat <= thr).astype(int)
        return float(accuracy_score(y, pred))

    glr = dict(A_MM=glr_acc(stat_te_M, yte_M, thr_M), A_MC=glr_acc(stat_te_C, yte_C, thr_M),
               A_CM=glr_acc(stat_te_M, yte_M, thr_C), A_CC=glr_acc(stat_te_C, yte_C, thr_C),
               threshold_matched=thr_M, threshold_confound=thr_C,
               fit_balanced_accuracy_matched=balacc_fit_M,
               fit_balanced_accuracy_confound=balacc_fit_C)
    for k in ("A_MM", "A_MC", "A_CM", "A_CC"):
        n = n_test_M if k in ("A_MM", "A_CM") else n_test_C
        glr[k + "_se"] = binom_se(glr[k], n)
    result["glr"] = glr

    # ---------------- TCN ----------------
    def to_tensor(X):
        return torch.tensor(np.stack([tcn_input(np.asarray(x)) for x in X]), dtype=torch.float32)

    Ttr_M, Ttr_C = to_tensor(Xtr_M), to_tensor(Xtr_C)
    Tte_M, Tte_C = to_tensor(Xte_M), to_tensor(Xte_C)
    ytr_M_t, ytr_C_t = torch.tensor(ytr_M, dtype=torch.float32), torch.tensor(ytr_C, dtype=torch.float32)
    yte_M_t, yte_C_t = torch.tensor(yte_M, dtype=torch.float32), torch.tensor(yte_C, dtype=torch.float32)

    tcn_seed_detail = {"M": {}, "C": {}}
    accs = {"A_MM": [], "A_MC": [], "A_CM": [], "A_CC": []}
    for seed in SEEDS_TCN:
        Ttr_M_s, ytr_M_s, Tval_M_s, yval_M_s = split_val(Ttr_M, ytr_M_t, seed=seed)
        model_M = TCNClassifier(ny=cond.ny)
        res_M = train_binary_classifier(model_M, Ttr_M_s, ytr_M_s, Tval_M_s, yval_M_s,
                                         Tte_M, yte_M_t, seed=seed)
        model_M.eval()
        with torch.no_grad():
            probs_MC = torch.sigmoid(model_M(Tte_C)).numpy()
        preds_MC = (probs_MC >= 0.5).astype(int)
        acc_MM = float(np.mean(np.array(res_M["test_preds"]) == np.array(res_M["test_labels"])))
        acc_MC = float(np.mean(preds_MC == yte_C))

        Ttr_C_s, ytr_C_s, Tval_C_s, yval_C_s = split_val(Ttr_C, ytr_C_t, seed=seed)
        model_C = TCNClassifier(ny=cond.ny)
        res_C = train_binary_classifier(model_C, Ttr_C_s, ytr_C_s, Tval_C_s, yval_C_s,
                                         Tte_C, yte_C_t, seed=seed)
        model_C.eval()
        with torch.no_grad():
            probs_CM = torch.sigmoid(model_C(Tte_M)).numpy()
        preds_CM = (probs_CM >= 0.5).astype(int)
        acc_CC = float(np.mean(np.array(res_C["test_preds"]) == np.array(res_C["test_labels"])))
        acc_CM = float(np.mean(preds_CM == yte_M))

        accs["A_MM"].append(acc_MM); accs["A_MC"].append(acc_MC)
        accs["A_CM"].append(acc_CM); accs["A_CC"].append(acc_CC)
        tcn_seed_detail["M"][str(seed)] = dict(A_MM=acc_MM, A_MC=acc_MC, best_epoch=res_M["best_epoch"])
        tcn_seed_detail["C"][str(seed)] = dict(A_CC=acc_CC, A_CM=acc_CM, best_epoch=res_C["best_epoch"])

    tcn = {k: float(np.mean(v)) for k, v in accs.items()}
    tcn.update({k + "_sd": float(np.std(v)) for k, v in accs.items()})
    tcn["seed_detail"] = tcn_seed_detail
    result["tcn"] = tcn

    return result


def main():
    out = {}
    t0 = time.time()
    for i, cond in enumerate(EXPERIMENT_MATRIX):
        tC0 = time.time()
        print(f"=== {cond.condition_id} ===", flush=True)
        out[cond.condition_id] = run_condition(cond.condition_id, seed_base_offset=93000 + i * 100)
        r = out[cond.condition_id]
        print(f"  XGB  MM={r['xgboost']['A_MM']:.4f} MC={r['xgboost']['A_MC']:.4f} "
              f"CM={r['xgboost']['A_CM']:.4f} CC={r['xgboost']['A_CC']:.4f}", flush=True)
        print(f"  GLR  MM={r['glr']['A_MM']:.4f} MC={r['glr']['A_MC']:.4f} "
              f"CM={r['glr']['A_CM']:.4f} CC={r['glr']['A_CC']:.4f}", flush=True)
        print(f"  TCN  MM={r['tcn']['A_MM']:.4f} MC={r['tcn']['A_MC']:.4f} "
              f"CM={r['tcn']['A_CM']:.4f} CC={r['tcn']['A_CC']:.4f}  "
              f"(cond wall {time.time()-tC0:.1f}s)", flush=True)
        # Incremental save after every condition so a timeout/interruption
        # does not lose completed conditions.
        with open(ART / "severity_matrix_full.json", "w") as f:
            json.dump(out, f, indent=1)

    out["_meta"] = dict(
        n_runs_per_class=N_RUNS_PER_CLASS, confound_policy=CONFOUND_POLICY,
        reference_policy=dict(process=[REF_LO, REF_HI], sensor=[REF_LO, REF_HI]),
        tcn_seeds=list(SEEDS_TCN), xgb_hparams=AUDIT_XGB_HPARAMS,
        total_wallclock_s=time.time() - t0,
        design_note="Paired-test design: one matched and one confounded test set per "
                    "condition, shared across the matched-trained and confound-trained "
                    "models of each model class, so every matrix contrast (A_MM vs A_CM; "
                    "A_MC vs A_CC; A_MM vs A_MC; A_CM vs A_CC) is evaluated on identical "
                    "test runs within that model class.",
    )
    with open(ART / "severity_matrix_full.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved", ART / "severity_matrix_full.json")
    print(f"TOTAL WALLCLOCK: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
