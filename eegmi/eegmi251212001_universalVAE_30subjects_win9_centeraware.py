#!/usr/bin/env python3
"""
Universal EEG VAE over 30 subjects (EEGMMI):

- Uses 64-channel EEG
- window_points = 9, stride = 1
- Flattens each window to 9*64 = 576-dim vector

Data:
  - Loads multiple subjects (default: the 30 chosen subjects)
  - For each subject, loads all 14 .npy runs in EEGMI_numpy/Sxxx/
  - For each (T,64) run:
        - build windows (N, 9, 64)
        - randomly sample approx 1/downsample_factor of windows
  - Concatenates sampled windows from ALL subjects
  - Shuffles them

Training:
  - VAE(x) with:
        input_dim = 9*64
        hidden_dim = 512
        latent_dim = 8
        beta = 3e-4
  - Reconstruction loss:
        Huber, center-aware:
        center time-slice (index 4) weighted by center_weight (default 5.0)
  - KL term multiplied by beta
  - Adam optimizer
  - Train/val split (e.g. 90%/10%)

Usage example:

  python eegmi_universalVAE_30subj.py \
      --data-dir EEGMI_numpy \
      --output checkpoints_vae_251210_universal/vae_universal30_win9_hd512_ld8_beta3e-04_251210_univ.pth \
      --epochs 100 \
      --batch-size 512 \
      --device cuda

"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset


# -----------------------------------------------------
# Default subject list (30 chosen subjects)
# -----------------------------------------------------

DEFAULT_SUBJECTS_30 = [
    "S008", "S009", "S012", "S022", "S026",
    "S030", "S031", "S033", "S038", "S047",
    "S050", "S052", "S054", "S067", "S071",
    "S072", "S077", "S079", "S081", "S086",
    "S089", "S092", "S093", "S095", "S098",
    "S100", "S103", "S105", "S106", "S109",
]


# -----------------------------------------------------
# Data loading helpers
# -----------------------------------------------------

def safe_load_npy(path: Path) -> np.ndarray:
    """Robust .npy loading. Skips macOS '._' sidecar files."""

    if path.name.startswith("._"):
        raise ValueError("Sidecar file")

    try:
        arr = np.load(path)
    except ValueError as e:
        msg = str(e)
        if "pickled data" in msg:
            # trusted local data only
            arr = np.load(path, allow_pickle=True)
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
                arr = arr.item()
        else:
            raise

    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{path} did not load as ndarray (got {type(arr)})")

    return arr


def make_windows(X: np.ndarray, window_points: int = 9, stride: int = 1):
    """
    X: (T, 64)
    Returns:
        wins: (N, window_points, 64)
    """
    T, C = X.shape
    W = window_points
    if C != 64:
        raise ValueError(f"Expected 64 channels, got {C}")

    if T < W:
        return np.zeros((0, W, C), dtype=X.dtype)  # no windows

    starts = list(range(0, T - W + 1, stride))
    wins = np.stack([X[s:s+W] for s in starts], axis=0)
    return wins  # (N,W,64)


def collect_sampled_windows(
    data_dir: str,
    subjects,
    window_points: int = 9,
    stride: int = 1,
    downsample_factor: int = 30,
    seed: int = 123,
):
    """
    For each subject in `subjects`, load all runs, make windows, then
    randomly keep approx 1/downsample_factor of the windows for that run.

    Returns:
      X_all: (M, W, 64) numpy array of sampled windows
    """
    rng = np.random.RandomState(seed)
    data_root = Path(data_dir)

    all_wins = []

    for subj in subjects:
        subj_dir = data_root / subj
        if not subj_dir.exists():
            print(f"[WARN] Subject dir not found: {subj_dir}, skipping.")
            continue

        run_files = sorted(subj_dir.glob("*.npy"))
        if not run_files:
            print(f"[WARN] No .npy under {subj_dir}, skipping subject.")
            continue

        for p in run_files:
            try:
                X = safe_load_npy(p)
                if X.ndim != 2 or X.shape[1] != 64:
                    print(f"[Skip] {p.name}: shape {X.shape} (expected (T,64))")
                    continue
            except Exception as e:
                print(f"[Skip] {p.name}: load failed ({e})")
                continue

            wins = make_windows(X, window_points=window_points, stride=stride)
            N = wins.shape[0]
            if N == 0:
                print(f"[Skip] {p.name}: not enough length for windows.")
                continue

            # randomly subsample ≈ N/downsample_factor
            n_keep = max(1, int(round(N / float(downsample_factor))))
            idx = rng.permutation(N)[:n_keep]
            wins_sub = wins[idx]

            all_wins.append(wins_sub)
            print(f"  subject={subj} file={p.name}: "
                  f"windows={N}, keep={wins_sub.shape[0]}")

    if not all_wins:
        raise RuntimeError("No windows collected. Check data_dir/subjects.")

    X_all = np.concatenate(all_wins, axis=0)
    return X_all  # (M,W,64)


# -----------------------------------------------------
# VAE model
# -----------------------------------------------------

class EEGWindowVAE(nn.Module):
    """
    Simple MLP VAE for flattened EEG windows (W*64).
    - encode(x) -> mu, logvar
    - decode(z) -> recon_x
    - forward(x) -> recon_x, mu, logvar
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        h = self.enc(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# -----------------------------------------------------
# Loss: center-aware Huber + beta-KL
# -----------------------------------------------------

def huber_center_weighted_loss(
    recon_x, x,
    window_points: int,
    huber_delta: float = 1.0,
    center_weight: float = 5.0,
):
    """
    recon_x, x: (B, D=window_points*64)
    Reshapes to (B, W, 64), applies Huber loss with extra weight on center slice.
    """
    B, D = x.shape
    W = window_points
    C = 64
    assert D == W * C, f"Expected D={W*C}, got {D}"

    recon = recon_x.view(B, W, C)
    target = x.view(B, W, C)

    diff = recon - target
    abs_diff = diff.abs()

    # Huber
    delta = huber_delta
    loss = torch.where(
        abs_diff <= delta,
        0.5 * diff.pow(2) / delta,
        abs_diff - 0.5 * delta,
    )  # (B,W,C)

    # center weighting
    w = torch.ones(W, device=loss.device)
    center = W // 2
    w[center] = center_weight
    w = w.view(1, W, 1)  # (1,W,1)

    loss = loss * w  # (B,W,C)
    loss = loss.mean()  # average over B,W,C

    return loss


def kl_divergence(mu, logvar):
    """
    Standard VAE KL divergence to N(0, I).
    Returns mean KL over batch.
    """
    # per-sample KL: sum over latent dims
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


# -----------------------------------------------------
# Main training script
# -----------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-dir", required=True,
                    help="EEGMI root folder containing subject subfolders Sxxx/")
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS_30,
                    help="List of subject IDs (e.g. S008 S009 ...). "
                         "Default is 30 chosen subjects.")

    ap.add_argument("--window-points", type=int, default=9)
    ap.add_argument("--stride-points", type=int, default=1)

    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--beta", type=float, default=3e-4)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--center-weight", type=float, default=5.0)

    ap.add_argument("--downsample-factor", type=int, default=30,
                    help="Approximate factor to downsample windows per run. "
                         "downsample_factor=30 → keep ~1/30 windows.")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-fraction", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--output", required=True,
                    help="Path to save VAE checkpoint (.pth). "
                         "Directory will be created if needed.")

    args = ap.parse_args()

    # device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # collect sampled windows
    print("[Data] Collecting sampled windows from subjects:")
    print("       ", " ".join(args.subjects))

    X_all = collect_sampled_windows(
        args.data_dir,
        subjects=args.subjects,
        window_points=args.window_points,
        stride=args.stride_points,
        downsample_factor=args.downsample_factor,
        seed=args.seed,
    )  # (M,W,64)

    M, W, C = X_all.shape
    print(f"[Data] Total sampled windows: {M} (each {W}x{C})")

    # flatten
    D = W * C
    X_flat = X_all.reshape(M, D).astype(np.float32)

    # shuffle + train/val split
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(M)
    M_val = int(round(M * args.val_fraction))
    M_train = M - M_val

    train_idx = idx[:M_train]
    val_idx = idx[M_train:]

    X_train = X_flat[train_idx]
    X_val = X_flat[val_idx]

    print(f"[Split] train={M_train}, val={M_val}")

    train_tensor = torch.from_numpy(X_train)
    val_tensor = torch.from_numpy(X_val)

    train_ds = TensorDataset(train_tensor)
    val_ds = TensorDataset(val_tensor)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # model
    vae = EEGWindowVAE(
        input_dim=D,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)

    print(f"[Model] VAE input_dim={D}, hidden_dim={args.hidden_dim}, "
          f"latent_dim={args.latent_dim}, beta={args.beta}")
    print(f"[Train] epochs={args.epochs}, batch_size={args.batch_size}, "
          f"huber_delta={args.huber_delta}, center_weight={args.center_weight}")

    best_val = float("inf")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        vae.train()
        total_train = 0.0
        n_train = 0

        for (xb,) in train_loader:
            xb = xb.to(device)
            recon, mu, logvar = vae(xb)

            recon_loss = huber_center_weighted_loss(
                recon, xb,
                window_points=args.window_points,
                huber_delta=args.huber_delta,
                center_weight=args.center_weight,
            )
            kl = kl_divergence(mu, logvar)
            loss = recon_loss + args.beta * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
            optimizer.step()

            total_train += loss.item() * xb.size(0)
            n_train += xb.size(0)

        train_loss = total_train / max(1, n_train)

        # validation
        vae.eval()
        total_val = 0.0
        n_val = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, mu, logvar = vae(xb)
                recon_loss = huber_center_weighted_loss(
                    recon, xb,
                    window_points=args.window_points,
                    huber_delta=args.huber_delta,
                    center_weight=args.center_weight,
                )
                kl = kl_divergence(mu, logvar)
                loss = recon_loss + args.beta * kl

                total_val += loss.item() * xb.size(0)
                n_val += xb.size(0)

        val_loss = total_val / max(1, n_val)

        print(f"Epoch {ep:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "input_dim": D,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "beta": args.beta,
                "window_points": args.window_points,
                "stride_points": args.stride_points,
                "huber_delta": args.huber_delta,
                "center_weight": args.center_weight,
                "subjects": args.subjects,
                "downsample_factor": args.downsample_factor,
                "state_dict": vae.state_dict(),
                "best_val": best_val,
            }
            torch.save(ckpt, out_path)
            print(f"  [Best] val -> {best_val:.6f} | saved to {out_path}")

    print(f"[Done] Best val={best_val:.6f}")
    print(f"[Saved best] {out_path}")


if __name__ == "__main__":
    main()
