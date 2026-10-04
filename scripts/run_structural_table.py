"""
Regenerate the full 36-row structural table (6 process channels x
6 (process,sensor,ny) combinations) at the single explicit primary
horizon L=8, with the L in {6,12,24} robustness sequence reported
separately. Built entirely from src.structural / src.faults /
src.config rather than duplicating any matrix construction inline.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.faults import PROCESS_FAULTS
from src.structural import structural_pair, worst_case_gamma, result_as_dict
from src.config import L_PRIMARY, L_ROBUST, GAMMA_GRID_19PT


def main():
    rows = []
    header = (f"{'pair':32s} | L={L_PRIMARY} primary: {'r1':>3s} {'r2':>3s} {'N':>4s} "
              f"{'floor':>5s} {'d_cap':>5s} {'excess':>6s} {'gamma*':>6s} | "
              f"d_cap robustness @ L={list(L_ROBUST)} | verdict")
    print(header)
    print("-" * len(header))
    for pname in PROCESS_FAULTS:
        for ny, sidx_list in [(2, [0, 1]), (4, [0, 1, 2, 3])]:
            for sidx in sidx_list:
                g_star, dcap_curve = worst_case_gamma(pname, sidx, ny, L_PRIMARY, GAMMA_GRID_19PT)
                primary = structural_pair(pname, sidx, ny, g_star, L_PRIMARY)
                robust = [structural_pair(pname, sidx, ny, g_star, L).d_cap for L in L_ROBUST]
                row = dict(process=pname, sensor_idx=sidx, ny=ny, gamma_star=g_star,
                           L_robust=list(L_ROBUST), d_cap_robustness=robust,
                           **result_as_dict(primary))
                rows.append(row)
                label = f"{pname} sidx={sidx} ny={ny}"
                print(f"{label:32s} |            {primary.r1:3d} {primary.r2:3d} "
                      f"{primary.ambient_dim:4d} {primary.generic_floor:5d} {primary.d_cap:5d} "
                      f"{primary.excess:6d} {g_star:6.2f} | {robust}  {primary.verdict.value}")

    out_path = Path(__file__).resolve().parents[1] / "results" / "structural_table.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nsaved {out_path}  ({len(rows)} rows)")
    return rows


if __name__ == "__main__":
    main()
