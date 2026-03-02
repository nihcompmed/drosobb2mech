#!/usr/bin/env python3
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 1.0,
    "axes.labelsize": 11.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLORS = ["#0072B2", "#E69F00", "#CC79A7"]


def _load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def _get_target_epochs(indir: Path):
    cfg_path = indir / "Step02_run_config.json"
    if cfg_path.exists():
        cfg = _load_json(cfg_path)
        if "eval_epochs" in cfg and isinstance(cfg["eval_epochs"], list) and len(cfg["eval_epochs"]) > 0:
            te = sorted({float(x) for x in cfg["eval_epochs"]})
            if te[0] != 0.0:
                te = [0.0] + te
            if "epochs_max" in cfg:
                te.append(float(cfg["epochs_max"]))
                te = sorted(set(te))
            return np.array(te, dtype=float), cfg_path
    return None, None


def _extract_epoch_recon(metrics_by_epoch: dict):
    epochs = []
    recon = []
    for k, v in metrics_by_epoch.items():
        try:
            e = float(k)
        except Exception:
            continue
        if isinstance(v, dict) and "recon" in v:
            epochs.append(e)
            recon.append(float(v["recon"]))
    if len(epochs) == 0:
        raise ValueError("No valid epoch/recon entries found.")
    order = np.argsort(np.array(epochs))
    epochs = np.array(epochs, dtype=float)[order]
    recon = np.array(recon, dtype=float)[order]
    return epochs, recon


def _interp_to_grid(x_src, y_src, x_tgt):
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_tgt = np.asarray(x_tgt, dtype=float)

    uniq_x, uniq_idx = np.unique(x_src, return_index=True)
    uniq_y = y_src[uniq_idx]

    if uniq_x.size == 1:
        return np.full_like(x_tgt, uniq_y[0], dtype=float)

    return np.interp(x_tgt, uniq_x, uniq_y)


def _format_k_label(k, capped=False):
    return f"{k:,}*" if capped else f"{k:,}"


def _pick_epoch_index(target_epochs, epoch_pick, tol=1e-9):
    target_epochs = np.asarray(target_epochs, dtype=float)
    idx_exact = np.where(np.isclose(target_epochs, epoch_pick, atol=tol, rtol=0.0))[0]
    if idx_exact.size > 0:
        return int(idx_exact[0]), float(target_epochs[idx_exact[0]])
    idx_near = int(np.argmin(np.abs(target_epochs - epoch_pick)))
    return idx_near, float(target_epochs[idx_near])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True,
                    help="Step02 outdir containing Step02_metrics_subject=*.json")
    ap.add_argument("--outpng", default=None)
    ap.add_argument("--outpdf", default=None)
    ap.add_argument("--epoch-pick", type=float, default=3.0,
                    help="Epoch used for panel b comparison (default: 3.0)")
    ap.add_argument("--dpi", type=int, default=600,
                    help="Raster export dpi (default: 600)")
    ap.add_argument("--print-files", action="store_true",
                    help="Print every imported file path during the run")
    args = ap.parse_args()

    indir = Path(args.indir)
    files = sorted(indir.glob("Step02_metrics_subject=*_kreq=*_seedSplit=*_seedTorch=*.json"))
    if len(files) == 0:
        raise RuntimeError(
            f"No metrics json found in {indir}. Expected files like "
            f"Step02_metrics_subject=*_kreq=*_seedSplit=*_seedTorch=*.json"
        )

    target_epochs, cfg_path = _get_target_epochs(indir)
    imported_paths = []
    if cfg_path is not None:
        imported_paths.append(cfg_path)

    if target_epochs is None:
        union = set()
        for fp in files:
            imported_paths.append(fp)
            d = _load_json(fp)
            ep, _ = _extract_epoch_recon(d["metrics_by_epoch"])
            union |= set(ep.tolist())
        target_epochs = np.array(sorted(union), dtype=float)
    
    by_k = defaultdict(list)
    capped_flag = defaultdict(bool)

    # Avoid duplicate listing if files were already appended in the fallback branch.
    already_listed = {str(p.resolve()) for p in imported_paths}
    for fp in files:
        rp = str(fp.resolve())
        if rp not in already_listed:
            imported_paths.append(fp)
            already_listed.add(rp)
        d = _load_json(fp)
        kreq = int(d.get("k_requested", d.get("kreq", -1)))
        if kreq < 0:
            raise KeyError(f"Cannot find k_requested in {fp}")

        k_used = d.get("k_used", d.get("k_train", d.get("k_used_train", kreq)))
        k_used = int(k_used)
        if k_used < kreq:
            capped_flag[kreq] = True

        ep_src, r_src = _extract_epoch_recon(d["metrics_by_epoch"])
        r_grid = _interp_to_grid(ep_src, r_src, target_epochs)
        by_k[kreq].append(r_grid)

    print("[Imported files]")
    for p in imported_paths:
        print(str(p))

    ks = sorted(by_k.keys())
    n_subj = {k: len(by_k[k]) for k in ks}
    unique_n = sorted(set(n_subj.values()))
    if len(unique_n) != 1:
        print(f"[Warning] Unequal subject counts by k: {n_subj}")
    n_text = f"n = {max(unique_n)}" if len(unique_n) > 0 else ""

    improvement_by_k = {}
    for k in ks:
        recon_mat = np.stack(by_k[k], axis=0)
        # Positive values mean lower reconstruction error after fine-tuning.
        improvement = (recon_mat[:, [0]] - recon_mat) / recon_mat[:, [0]] * 100.0
        improvement_by_k[k] = improvement

    ep_idx, epoch_used = _pick_epoch_index(target_epochs, args.epoch_pick)
    if not np.isclose(epoch_used, args.epoch_pick):
        print(f"[Info] Requested epoch {args.epoch_pick:g} not found exactly; using nearest epoch {epoch_used:g}.")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(11.2, 4.85),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.14, top=0.92, wspace=0.28)

    for i, k in enumerate(ks):
        imp_mat = improvement_by_k[k]
        mean = imp_mat.mean(axis=0)
        se = imp_mat.std(axis=0, ddof=1) / np.sqrt(imp_mat.shape[0]) if imp_mat.shape[0] > 1 else np.zeros_like(mean)
        color = COLORS[i % len(COLORS)]
        label = _format_k_label(k, capped_flag.get(k, False))

        axA.plot(
            target_epochs, mean,
            marker="o", markersize=4.8,
            linewidth=2.0,
            color=color,
            label=label,
            zorder=3,
        )
        axA.fill_between(target_epochs, mean - se, mean + se, color=color, alpha=0.18, zorder=2)

    axA.axhline(0.0, linestyle="--", linewidth=1.0, color="0.45", zorder=1)
    axA.set_xlabel("Fine-tuning epoch")
    axA.set_ylabel("Test reconstruction improvement (%)")
    axA.grid(True, alpha=0.22, linewidth=0.8)
    axA.legend(title="Adaptation budget", frameon=False, loc="upper left",
               bbox_to_anchor=(0.105, 0.985), handlelength=2.0,
               title_fontsize=9.5, fontsize=9.3, borderaxespad=0.0)
    if n_text:
        axA.text(0.98, 0.98, n_text, transform=axA.transAxes, ha="right", va="top", fontsize=9.5)
    axA.text(0.008, 0.988, "(a)", transform=axA.transAxes, fontweight="bold", fontsize=13.0, va="top", ha="left",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.2))

    data = [improvement_by_k[k][:, ep_idx] for k in ks]
    labels = [_format_k_label(k, capped_flag.get(k, False)) for k in ks]

    bp = axB.boxplot(
        data,
        labels=labels,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(linewidth=1.8, color="black"),
        whiskerprops=dict(linewidth=1.2, color="black"),
        capprops=dict(linewidth=1.2, color="black"),
        boxprops=dict(linewidth=1.2, color="black"),
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.18)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        x = i + rng.normal(0.0, 0.045, size=len(vals))
        axB.scatter(
            x, vals,
            s=22,
            facecolors="white",
            edgecolors="black",
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
        )

    axB.axhline(0.0, linestyle="--", linewidth=1.0, color="0.45", zorder=1)
    axB.set_xlabel("Adaptation budget k (windows)")
    axB.set_ylabel(f"Improvement at epoch {epoch_used:g} (%)")
    axB.grid(True, axis="y", alpha=0.22, linewidth=0.8)
    axB.text(0.015, 0.985, "(b)", transform=axB.transAxes, fontweight="bold", fontsize=13.0, va="top", ha="left",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.15))
    axB.text(0.98, -0.145, "* capped by available training windows for some subjects",
             transform=axB.transAxes, ha="right", va="top", fontsize=8.8, clip_on=False)

    for ax in (axA, axB):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if args.outpng is None:
        args.outpng = str(indir / "Figure3_finetune_efficiency_npjsba.png")
    if args.outpdf is None:
        args.outpdf = str(indir / "Figure3_finetune_efficiency_npjsba.pdf")

    fig.savefig(args.outpng, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.outpdf, bbox_inches="tight")
    print(f"[Saved] {args.outpng}")
    print(f"[Saved] {args.outpdf}")


if __name__ == "__main__":
    main()
