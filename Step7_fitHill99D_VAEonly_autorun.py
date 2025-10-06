#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit Hill model using a VAE-only teacher (no latent ODE).
Auto-searches VAE checkpoint and imputed CSV from job_id.
"""

import os, glob, argparse
from pathlib import Path

# (everything else from your previous Step7 code remains unchanged,
# I'm only showing the new main() section with auto-search logic)

# ====== Auto-locate helpers ======

def find_ckpt(job_id, ckdir="checkpoints"):
    pats = [
        os.path.join(ckdir, f"VAE*{job_id}*.ckpt"),
    ]
    hits = []
    for pat in pats:
        hits.extend(glob.glob(pat))
    # filter: must start with VAE and not contain SMOOTH
    hits = [h for h in hits if os.path.basename(h).startswith("VAE") and "SMOOTH" not in os.path.basename(h)]
    if not hits:
        raise FileNotFoundError(f"No valid VAE ckpt found for job_id={job_id}")
    # choose shortest filename for determinism
    hits.sort(key=lambda x: (len(x), x))
    return hits[0]

def find_imputed_csv(job_id: str) -> str:
    prefix = job_id[:6]
    if prefix == "250915":
        root = "ImputedOut_clean"
    elif prefix == "250916":
        root = "ImputedOut_250916"
    elif prefix == "250917":
        root = "ImputedOut_250917"
    else:
        raise ValueError(f"Unknown job family prefix={prefix}")
    pats = [
        os.path.join(root, f"Imputed_{job_id}_valid_ldim*.csv"),
        os.path.join(root, f"Imputed_*_{job_id}_valid_ldim*.csv"),
    ]
    hits = []
    for pat in pats:
        hits.extend(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"No imputed CSV found for job_id={job_id} in {root}")
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]

# ========== Main ==========
def main():
    ap = argparse.ArgumentParser("Fit Hill using VAE-only pushforward (no ODE)")
    ap.add_argument("--job_id", required=True, type=str, help="Job ID (e.g. 250916032)")
    ap.add_argument("--n_sub", type=int, default=3, help="Subdivisions per interval for teacher derivatives")
    ap.add_argument("--int_substeps", type=int, default=5, help="RK4 integration substeps")
    ap.add_argument("--bsize", type=int, default=1024)
    ap.add_argument("--max_epochs", type=int, default=600)
    ap.add_argument("--ep_maxlr", type=int, default=150)
    ap.add_argument("--maxlr", type=float, default=3e-3)
    ap.add_argument("--l1_lambda", type=float, default=1e-4)
    ap.add_argument("--sched_start_deriv", type=int, default=30)
    ap.add_argument("--sched_full_int", type=int, default=80)
    ap.add_argument("--deriv_only", action="store_true",
                    help="Train only on derivative + L1 (skip state/collocation)")
    args = ap.parse_args()

    job_id = args.job_id
    vae_class = "Step2_trainVAE_importable_ref:VAE"

    vae_ckpt = find_ckpt(job_id)
    imputed_csv = find_imputed_csv(job_id)

    print(f"[info] job_id={job_id}")
    print(f"[info] VAE ckpt: {vae_ckpt}")
    print(f"[info] Imputed CSV: {imputed_csv}")

    # reuse your previous pipeline
    from Step7_fitHill99D_VAEonly import (
        load_vae, build_pushfwd_from_vae,
        MixedDerivStateDataset, HillTrainerTorch
    )

    encode_mu, decode, expects_xyz, out_dim = load_vae(vae_class, vae_ckpt)
    G = out_dim

    Xd, Yd, Xn, Xn1, T = build_pushfwd_from_vae(
        imputed_csv, encode_mu, decode,
        n_sub=args.n_sub, expects_xyz=expects_xyz, G=G, bsize_pairs=4096
    )

    import numpy as np, torch
    med = np.median(Yd, axis=0)
    mad = np.median(np.abs(Yd - med), axis=0)
    sigma_vec = np.maximum(1.4826 * mad, 1e-6)

    ds = MixedDerivStateDataset(Xd, Yd, Xn, Xn1, T)

    l1_str = f"{args.l1_lambda:.0e}".replace("e-0", "en")
    save_job_id = f"{job_id}hill_l1{l1_str}"

    trainer = HillTrainerTorch(
        G=G, max_lr=args.maxlr, ep_maxlr=args.ep_maxlr, max_epochs=args.max_epochs,
        l1_lambda=args.l1_lambda, int_substeps=args.int_substeps,
        sched_epochs=(args.sched_start_deriv, args.sched_full_int),
        sigma_vec=sigma_vec, wd=0.0, deriv_only=args.deriv_only
    )

    params_csv = trainer.train(
        ds, bsize=args.bsize, ckpt_dir="HillModels",
        name_prefix=f"Hill_fromVAEonly_sub{args.n_sub}_rk4{args.int_substeps}",
        job_id=save_job_id
    )

    print(f"[done] best params at: {params_csv}")


if __name__ == "__main__":
    main()
