#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for Step8 Hill integration across multiple jobids.
Collects per-gene MAE (99D) for each model into one CSV.
"""

import os, glob, argparse, subprocess
import numpy as np
import pandas as pd

# ---------------- Pick imputed CSV by job family ----------------
def _pick_imputed(jobid: str):
    if jobid.startswith("250915"):
        root = "ImputedOut_clean"
    elif jobid.startswith("250916"):
        root = "ImputedOut_250916"
    elif jobid.startswith("250917"):
        root = "ImputedOut_250917"
    else:
        return None

    pats = [
        os.path.join(root, f"Imputed_{jobid}_valid_ldim*.csv"),
        os.path.join(root, f"Imputed_*_{jobid}_valid_ldim*.csv"),
    ]
    hits = []
    for pat in pats:
        hits.extend(glob.glob(pat))
    if not hits:
        return None
    hits.sort(key=os.path.getmtime, reverse=True)  # latest
    return hits[0]

# ---------------- Run one job ----------------
def run_one(jobid, test_cells, method, nseg, window, v_prune_thresh, l1_str):
    # Hill model (code2 CSV)
    
    
    patt = f"HillModels/Hill_fromVAEonly_sub3_rk45_G99_{jobid}hill_l1{l1_str}.csv"

    hits = glob.glob(patt)
    if not hits:
        print(f"[skip] no Hill model for {jobid}")
        return None
    code2 = hits[0]

    # Imputed CSV
    data_csv = _pick_imputed(jobid)
    if data_csv is None:
        print(f"[skip] no imputed CSV for {jobid}")
        return None

    # Run Step8 integration
    cmd = [
        "python", "Step8_integrateHill_zerosmallVij_code2_window.py",
        "--code2", code2,
        "--data_csv", data_csv,
        "--test_cells", test_cells,
        "--method", method,
        "--nseg", str(nseg),
        "--window", str(window),
        "--v_prune_thresh", str(v_prune_thresh),
    ]
    print("[run]", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True)

    # Parse per-gene MAE
    lines = out.splitlines()
    mae_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Per-gene MAE"):
            mae_line = lines[i+1 : i+1+99]
            break
    if mae_line is None:
        print(f"[error] Could not parse MAE for {jobid}")
        return None

    txt = " ".join(mae_line)
    vals = np.fromstring(txt.replace("[","").replace("]",""), sep=" ")
    if vals.size != 99:
        print(f"[warn] Expected 99 genes, got {vals.size} for {jobid}")
        return None
    return vals

# ---------------- Main ----------------
def main(args):
    all_rows = {}
    for j in range(25, 37):   # jobid 025–036
        for prefix in ["250915", "250916"]:
            jobid = f"{prefix}{j:03d}"
            mae = run_one(jobid, args.test_cells,
              args.method, args.nseg, args.window, args.v_prune_thresh,
              args.l1_str)

            if mae is not None:
                all_rows[jobid] = mae

    # Save
    df = pd.DataFrame.from_dict(all_rows, orient="index")
    df.columns = [f"gene{i+1}" for i in range(99)]

    # Compute mean over cols 2–100 (gene2 … gene100) per row
    df["meanAE"] = df.mean(axis=1)


    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"HillIntegration_perGeneMAE_l1{args.l1_str}.csv")

    df.to_csv(out_csv, index_label="jobid")
    print(f"[done] Saved {out_csv} with shape {df.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_cells", type=str, default="test_cells.csv")
    ap.add_argument("--method", type=str, default="rk4")
    ap.add_argument("--nseg", type=int, default=5)
    ap.add_argument("--l1_str", type=str, required=True,
                help="L1 string to match in Hill model filename (e.g. 1en4 or 1en5)")

    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--v_prune_thresh", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, default="HillIntegrationResults")
    args = ap.parse_args()
    main(args)
