#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter ImputedOut CSVs to only keep training cell IDs for each job.
Saves all results into universality/ImputedOut_universality/
"""

import os, re, pickle
import numpy as np
import pandas as pd

ROOT = "Sec_universality"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

# Base files used in Step2_trainVAE
df0_fn = os.path.join(DATA_DIR, "all_ctgxyz_99genes_fillrand_fillzero_t012345_noshuffle.csv")
test_pkl = os.path.join(DATA_DIR, "test_cells.pkl")
with open(test_pkl, "rb") as f:
    test_idx = pickle.load(f)
df0 = pd.read_csv(df0_fn, header=None).values

ncells = 6078
rows_t0 = df0[:ncells]
rows_t0 = rows_t0[~np.isin(rows_t0[:, 0], test_idx)]
xyz_t0 = rows_t0[:, 101:104]
cell_ids_remaining = np.setdiff1d(np.arange(1, ncells + 1), test_idx)

# --- Helpers ---
def parse_swarm_line(line):
    parts = line.strip().split()
    if len(parts) < 17:
        return None
    ldim = int(parts[10])
    regiontype, subregion = parts[13], parts[14]
    job_id = int(parts[15])
    seed_select = int(parts[16])
    return job_id, regiontype, subregion, ldim, seed_select


def get_selected_cells(regiontype, subregion, seed_select=1):
    x, z = xyz_t0[:, 0], xyz_t0[:, 2]
    n_part = int(1/3 * len(x))
    n_half = int(0.5 * len(x))
    sorted_idx_x, sorted_idx_z = np.argsort(x), np.argsort(z)

    if regiontype == "amp":
        a_cells = cell_ids_remaining[sorted_idx_x[:n_part]]
        m_cells = cell_ids_remaining[sorted_idx_x[n_part:2*n_part]]
        p_cells = cell_ids_remaining[sorted_idx_x[2*n_part:]]
        if subregion == "a": return a_cells
        if subregion == "m": return m_cells
        if subregion == "p": return p_cells
    elif regiontype == "ap":
        a_cells = cell_ids_remaining[sorted_idx_x[:n_half]]
        p_cells = cell_ids_remaining[sorted_idx_x[-n_half:]]
        return a_cells if subregion == "a" else p_cells
    elif regiontype == "dmv":
        v_cells = cell_ids_remaining[sorted_idx_z[:n_part]]
        m_cells = cell_ids_remaining[sorted_idx_z[n_part:2*n_part]]
        d_cells = cell_ids_remaining[sorted_idx_z[2*n_part:]]
        if subregion == "d": return d_cells
        if subregion == "m": return m_cells
        if subregion == "v": return v_cells
    elif regiontype == "dv":
        v_cells = cell_ids_remaining[sorted_idx_z[:n_half]]
        d_cells = cell_ids_remaining[sorted_idx_z[-n_half:]]
        return d_cells if subregion == "d" else v_cells
    elif regiontype == "all":
        rng = np.random.RandomState(seed_select)
        pool = cell_ids_remaining
        if subregion == "all33":
            n = len(pool) // 3
        elif subregion == "all50":
            n = len(pool) // 2
        else:
            raise ValueError("Invalid subregion for all")
        return np.sort(rng.choice(pool, size=n, replace=False))
    else:
        raise ValueError("Unknown regiontype")

# --- Main ---
def main():
    swarm_files = [
        os.path.join(ROOT, "Step2_trainVAE_universality01.swarm"),
        os.path.join(ROOT, "Step2_trainVAE_universality02.swarm"),
    ]
    jobs = {}
    for swarm in swarm_files:
        with open(swarm) as f:
            for line in f:
                if not line.strip(): continue
                job = parse_swarm_line(line)
                if job: jobs[job[0]] = job  # job_id → (job_id, region, subregion, ldim)

    out_dir = os.path.join(ROOT, "ImputedOut_universality")
    os.makedirs(out_dir, exist_ok=True)

    for job_id, region, subregion, ldim, seed in jobs.values():
        print(f"[job {job_id}] Filtering {region}-{subregion}, ldim={ldim}, seed={seed}")

        # Cell IDs
        selected_cells = get_selected_cells(region, subregion, seed_select=seed)

        # Load imputed file
        imp_fn = os.path.join(ROOT, f"ImputedOut/ImputedOut_{job_id}/Imputed_{job_id}_ldim{ldim}.csv")
        if not os.path.exists(imp_fn):
            print(f"  Skipping (missing {imp_fn})")
            continue
        df = pd.read_csv(imp_fn, header=None)

        # Filter rows by cell ID
        mask = df.iloc[:,0].isin(selected_cells)
        df_filtered = df[mask]

        # Save new file
        out_fn = os.path.join(out_dir, f"Imputed_{job_id}_ldim{ldim}_trainonly.csv")
        df_filtered.to_csv(out_fn, header=False, index=False)
        print(f"  Saved {out_fn} with {len(df_filtered)} rows")


if __name__ == "__main__":
    main()
