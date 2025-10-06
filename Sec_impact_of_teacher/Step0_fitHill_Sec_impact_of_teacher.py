#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit Hill model *directly from data* (no VAE), with fixed gene dimension.

Key change vs prior version:
- We explicitly slice the first `gene_count` columns after [cid,time] as genes (default 27),
  ignoring any extra columns (xyz and/or appended g_next). The derivative target is Δg in that
  fixed gene space, so G will be exactly `gene_count` regardless of CSV width.

Loss, curriculum, logging, and lasso settings are identical to the with-BB run to enable fair A/B.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

# ===== Globals =====
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42


# ===== Utils =====
def erf_smooth_l1(residual: torch.Tensor, sigma_vec: torch.Tensor) -> torch.Tensor:
    """Robust smooth |.| via E|r + N(0,sigma^2)| (per-gene sigma)."""
    sigma = torch.clamp(sigma_vec, min=1e-8)
    z = residual / (sigma * np.sqrt(2.0))
    term1 = residual * torch.erf(z)
    term2 = sigma * np.sqrt(2.0/np.pi) * torch.exp(-z*z)
    return (term1 + term2).mean()


# ===== Data-only supervision (fixed gene_count) =====
def build_data_only_supervision(imputed_csv: str, n_sub: int, gene_count: int):
    """
    imputed_csv columns may be:
      [cid, time, g_now(>=gene_count), ...maybe xyz(3) ...maybe g_next(gene_count) ...]
    We ONLY take the first `gene_count` genes after [cid,time] and ignore the rest.
    Δg is computed from paired rows (cid,t)->(cid,t+1) using those 27 genes.

    Returns:
      Xd, Yd:   sub-step states and constant derivative targets (Δg) for derivative fit
      Xn, Xn1:  one-step pairs for state loss
      T:        Δt column of ones
      G:        == gene_count
    """
    arr  = pd.read_csv(imputed_csv, header=None).to_numpy().astype(np.float64)
    cid  = arr[:, 0].astype(np.int64)
    time = arr[:, 1].astype(np.int64)

    G = int(gene_count)
    genes = arr[:, 2:2+G]  # <-- FIXED slice: first G columns only (ignore extras)
    if genes.shape[1] != G:
        raise RuntimeError(
            f"Requested gene_count={G}, but CSV has only {genes.shape[1]} gene columns after [cid,time]."
        )

    # index pairs (cid,t)->(cid,t+1)
    key2idx = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(cid, time))}
    pairs = [(int(c), int(t)) for c, t in zip(cid, time) if (int(c), int(t)+1) in key2idx]
    if not pairs:
        raise RuntimeError("No (cid,t)->(cid,t+1) pairs found in imputed_csv")

    K = n_sub + 1
    s_grid = np.linspace(0.0, 1.0, K, dtype=np.float64)

    Xd_list, Yd_list = [], []
    Xn_list, Xn1_list, T_list = [], [], []

    for (c, t) in pairs:
        i  = key2idx[(c, t)]
        j  = key2idx[(c, t+1)]
        g0 = genes[i]
        g1 = genes[j]
        dg = (g1 - g0)  # Δt assumed 1

        # sub-steps (linear path in gene space), derivative is constant Δg
        g_stack = np.stack([g0 + s * dg for s in s_grid], axis=0)   # [K, G]
        d_stack = np.tile(dg[None, :], (K, 1))                      # [K, G]

        Xd_list.append(g_stack)
        Yd_list.append(d_stack)

        Xn_list.append(g0[None, :])
        Xn1_list.append(g1[None, :])
        T_list.append(np.ones((1, 1), dtype=np.float64))

    Xd  = np.concatenate(Xd_list, axis=0)
    Yd  = np.concatenate(Yd_list, axis=0)
    Xn  = np.concatenate(Xn_list, axis=0)
    Xn1 = np.concatenate(Xn1_list, axis=0)
    T   = np.concatenate(T_list, axis=0)  # shape [Npairs,1]
    return Xd, Yd, Xn, Xn1, T, G


# ===== Dataset =====
class MixedDerivStateDataset(Dataset):
    """Returns derivative sample i and a random state pair j."""
    def __init__(self, Xd, Yd, Xn, Xn1, T):
        self.Xd  = torch.as_tensor(Xd,  dtype=torch.float64)
        self.Yd  = torch.as_tensor(Yd,  dtype=torch.float64)
        self.Xn  = torch.as_tensor(Xn,  dtype=torch.float64)
        self.Xn1 = torch.as_tensor(Xn1, dtype=torch.float64)
        self.T   = torch.as_tensor(T,   dtype=torch.float64)

    def __len__(self): return self.Xd.shape[0]
    def __getitem__(self, i):
        j = np.random.randint(0, self.Xn.shape[0])
        return (self.Xd[i], self.Yd[i], self.Xn[j], self.Xn1[j], self.T[j])


# ===== Hill model & integrator =====
class HillAll(nn.Module):
    """
    dg/dt = production(g) - gamma * g
    production_i(g) = b0_i + sum_j V_{i,j} * (g_j^n / (K^n + g_j^n))
    """
    def __init__(self, G: int):
        super().__init__()
        self.b0 = nn.Parameter(torch.zeros(G, dtype=torch.float64))
        self.V  = nn.Parameter(torch.randn(G, G, dtype=torch.float64) * 0.1)
        self.K  = nn.Parameter(torch.ones(G, G,  dtype=torch.float64) * 0.5)
        self.n  = nn.Parameter(torch.ones(G, G,  dtype=torch.float64) * 1.0)
        self.log_gamma = nn.Parameter(torch.zeros(G, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, G = x.shape
        x_safe = torch.clamp(x, min=1e-12)
        n_eff  = torch.clamp(self.n, min=1.0, max=3.0)
#        n_eff  = torch.clamp(self.n, min=1e-2, max=10.0)
        K_eff  = torch.clamp(self.K, min=0.2, max=2.5)
   #      K_eff  = torch.clamp(self.K, min=1e-4, max=1e4)
        gamma  = torch.exp(torch.clamp(self.log_gamma, -20.0, 20.0))

        xj  = x_safe.unsqueeze(1).expand(B, G, G)
        n_b = n_eff.unsqueeze(0).expand(B, G, G)
        K_b = K_eff.unsqueeze(0).expand(B, G, G)

        xn  = torch.pow(xj, n_b)
        den = torch.pow(K_b, n_b) + xn
        h   = xn / (den + x_safe.new_tensor(1e-12))
        h   = torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=0.0)

        production = (h * self.V.unsqueeze(0)).sum(dim=2) + self.b0
        out = production - gamma * x_safe
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        return out


def rk4(field: nn.Module, x0: torch.Tensor, T: torch.Tensor, n_sub: int) -> torch.Tensor:
    if T.ndim == 1:
        T = T.unsqueeze(1)
    steps = max(1, n_sub)
    h = T / float(steps)
    x = x0
    for _ in range(steps):
        k1 = field(x)
        k2 = field(x + 0.5 * h * k1)
        k3 = field(x + 0.5 * h * k2)
        k4 = field(x + h * k3)
        x  = x + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x


# ===== Trainer (plain PyTorch) =====
class HillTrainerTorch:
    def __init__(self, G, max_lr, ep_maxlr, max_epochs, l1_lambda,
                 int_substeps=5, sched_epochs=(100, 300),
                 sigma_vec=None, wd=0.0):
        self.field       = HillAll(G).to(DEVICE)
        self.loss_mse    = nn.MSELoss()
        self.max_lr      = max_lr
        self.ep_maxlr    = ep_maxlr
        self.max_epochs  = max_epochs
        self.l1_lambda   = l1_lambda
        self.int_sub     = int(int_substeps)
        self.sigma_vec   = torch.as_tensor(
            sigma_vec if sigma_vec is not None else np.ones(G),
            dtype=torch.float64, device=DEVICE
        )
        self.start_deriv, self.full_int = sched_epochs

        self.optimizer   = optim.Adam(self.field.parameters(), lr=max_lr, weight_decay=wd, foreach=False)
        self.scheduler   = LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, epoch):
        steep1 = 10.0 / float(self.ep_maxlr)
        steep2 = 10.0 / float(self.max_epochs)
        if epoch <= self.ep_maxlr:
            c1 = 1 + np.exp(-steep1 * self.ep_maxlr / 2.0)
            return c1 / (1 + np.exp(-steep1 * (epoch - self.ep_maxlr / 2.0)))
        else:
            c2 = 1 + np.exp(-steep2 * (3 * self.max_epochs / 4.0 - self.ep_maxlr))
            return c2 / (1 + np.exp(-steep2 * (3 * self.max_epochs / 4.0 - epoch)))

    def _weights(self, epoch):
        """
        Cosine curriculum from derivative->integration.
        Returns: w_deriv, w_int, alpha_for_monitor, coll_w
        """
        e0, e1 = self.start_deriv, self.full_int
        # endpoints
        wD_0, wI_0, alpha_0, coll_0 = 1.0, 0.10, 0.30, 0.30
        wD_1, wI_1, alpha_1, coll_1 = 0.20, 1.00, 0.10, 0.20

        if epoch <= e0:
            return wD_0, wI_0, alpha_0, coll_0
        if epoch >= e1:
            return wD_1, wI_1, alpha_1, coll_1

        # cosine interpolation
        a = (epoch - e0) / max(1, (e1 - e0))
        c = 0.5 - 0.5 * np.cos(np.pi * a)
        wD   = (1-c)*wD_0 + c*wD_1
        wI   = (1-c)*wI_0 + c*wI_1
        alpha= (1-c)*alpha_0 + c*alpha_1
        coll = (1-c)*coll_0 + c*coll_1
        return float(wD), float(wI), float(alpha), float(coll)

    def train(self, ds: MixedDerivStateDataset, bsize: int, val_frac=0.1,
              ckpt_dir="HillModels", name_prefix="Hill_dataonly", job_id="000"):
        torch.manual_seed(SEED); np.random.seed(SEED)

        n_val = max(1, int(val_frac * len(ds)))
        train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
                                        generator=torch.Generator().manual_seed(SEED))
        train_loader = DataLoader(train_ds, batch_size=bsize, shuffle=True,
                                  num_workers=4, pin_memory=torch.cuda.is_available())
        val_loader   = DataLoader(val_ds, batch_size=bsize, shuffle=False,
                                  num_workers=2, pin_memory=torch.cuda.is_available())

        best_val = np.inf
        os.makedirs(ckpt_dir, exist_ok=True)
        params_csv = os.path.join(
            ckpt_dir, f"{name_prefix}_sub{self.int_sub}_rk4{self.int_sub}_G{self.field.b0.numel()}_{job_id}.csv"
        )

        for epoch in range(self.max_epochs):
            self.field.train()
            w_deriv, w_int, alpha, coll_w = self._weights(epoch)
            logd=logs=logc=logl=0.0; nstep=0

            for batch in train_loader:
                xd, yd, xn, xn1, T = [b.to(DEVICE) for b in batch]

                # derivative loss (robust)
                y_hat = self.field(xd)
                loss_deriv = erf_smooth_l1(y_hat - yd, self.sigma_vec)

                # 1-step RK4 state loss
                x_pred = rk4(self.field, xn, T, n_sub=self.int_sub)
                loss_state = self.loss_mse(x_pred, xn1)

                # collocation (Euler defect) between xn and x_pred along straight path
                K  = max(2, self.int_sub)
                dt = T / float(K - 1)
                xs = torch.stack([xn + (i/(K-1))*(x_pred - xn) for i in range(K)], dim=0)  # [K,B,G]
                fxs = torch.stack([self.field(xs[i]) for i in range(K-1)], dim=0)
                defects = xs[1:] - xs[:-1] - dt.unsqueeze(0) * fxs
                loss_coll = (defects**2).mean()

                # lasso: L1 only on V
                l1_pen = self.l1_lambda * torch.abs(self.field.V).sum()

                loss = w_deriv*loss_deriv + w_int*loss_state + coll_w*loss_coll + l1_pen

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.field.parameters(), max_norm=5.0)
                self.optimizer.step()

                logd += float(loss_deriv.detach()); logs += float(loss_state.detach())
                logc += float(loss_coll.detach());  logl += float(l1_pen.detach()); nstep += 1

            self.scheduler.step()

            # validation
            self.field.eval()
            with torch.no_grad():
                v_deriv=v_state=v_coll=0.0; vc=0
                for batch in val_loader:
                    xd, yd, xn, xn1, T = [b.to(DEVICE) for b in batch]
                    v_deriv += float(erf_smooth_l1(self.field(xd)-yd, self.sigma_vec))
                    x_pred  = rk4(self.field, xn, T, n_sub=self.int_sub)
                    v_state += float(self.loss_mse(x_pred, xn1))
                    # collocation
                    K  = max(2, self.int_sub)
                    dt = T / float(K - 1)
                    xs = torch.stack([xn + (i/(K-1))*(x_pred - xn) for i in range(K)], dim=0)
                    fxs = torch.stack([self.field(xs[i]) for i in range(K-1)], dim=0)
                    defects = xs[1:] - xs[:-1] - dt.unsqueeze(0) * fxs
                    v_coll += float((defects**2).mean())
                    vc += 1
                v_deriv/=max(1,vc); v_state/=max(1,vc); v_coll/=max(1,vc)

                # monitor: α·val_d + (1-α)·val_i   (α printed as wC for continuity)
                monitor = alpha*v_deriv + (1.0-alpha)*v_state

            print(f"[ep {epoch:03d}] wD={w_deriv:.2f} wI={w_int:.2f} wC={alpha:.2f}  "
                  f"train d={logd/max(1,nstep):.4e} i={logs/max(1,nstep):.4e} c={logc/max(1,nstep):.4e} l1={logl/max(1,nstep):.4e}  "
                  f"val d={v_deriv:.4e} i={v_state:.4e} c={v_coll:.4e}  mon={monitor:.4e}")

            if monitor < best_val:
                best_val = monitor
                # save stacked params csv: [b0; V; K; n; gamma]
                with torch.no_grad():
                    b0 = self.field.b0.detach().cpu().numpy()[None, :]
                    V  = self.field.V.detach().cpu().numpy()
                    K  = self.field.K.detach().cpu().numpy()
                    n  = self.field.n.detach().cpu().numpy()
                    gamma = torch.exp(torch.clamp(self.field.log_gamma, -20.0, 20.0)).detach().cpu().numpy()[None, :]
                    mat = np.vstack([b0, V, K, n, gamma])
                    pd.DataFrame(mat).to_csv(params_csv, header=False, index=False)
                torch.save(self.field.state_dict(), os.path.join(ckpt_dir, f"{name_prefix}_best_{job_id}.pt"))
                print(f"  ✅ saved best -> {params_csv}  (monitor={best_val:.4e})")

        return params_csv


# ===== Main =====
def main():
    import argparse
    ap = argparse.ArgumentParser("Fit Hill directly from data (no teacher/VAEs)")
    ap.add_argument("--imputed_csv", required=True, type=str,
                    help="CSV with [cid, time, *many columns*]. We only use the first gene_count after time.")
    ap.add_argument("--gene_count",   type=int, default=27,
                    help="How many gene columns to use immediately after [cid,time].")
    ap.add_argument("--n_sub",        type=int, default=5,  help="Subdivisions per [t,t+1] for derivative supervision")
    ap.add_argument("--int_substeps", type=int, default=5,  help="RK4 substeps for state loss")
    ap.add_argument("--bsize",        type=int, default=1024)
    ap.add_argument("--max_epochs",   type=int, default=600)
    ap.add_argument("--ep_maxlr",     type=int, default=150)
    ap.add_argument("--maxlr",        type=float, default=3e-3)
    ap.add_argument("--l1_lambda",    type=float, default=1e-4)
    ap.add_argument("--sched_start_deriv", type=int, default=100)
    ap.add_argument("--sched_full_int",    type=int, default=300)
    ap.add_argument("--job_id",       type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)

    # Build supervision from raw data (fixed gene_count slice)
    Xd, Yd, Xn, Xn1, T, G = build_data_only_supervision(
        args.imputed_csv, n_sub=args.n_sub, gene_count=args.gene_count
    )

    # Robust per-gene scales for erf_smooth_l1
    med = np.median(Yd, axis=0)
    mad = np.median(np.abs(Yd - med), axis=0)
    sigma_vec = np.maximum(1.4826 * mad, 1e-6)

    ds = MixedDerivStateDataset(Xd, Yd, Xn, Xn1, T)

    trainer = HillTrainerTorch(
        G=G, max_lr=args.maxlr, ep_maxlr=args.ep_maxlr, max_epochs=args.max_epochs,
        l1_lambda=args.l1_lambda, int_substeps=args.int_substeps,
        sched_epochs=(int(args.sched_start_deriv), int(args.sched_full_int)),
        sigma_vec=sigma_vec, wd=0.0
    )

    params_csv = trainer.train(
        ds, bsize=args.bsize, ckpt_dir="HillModels",
        name_prefix=f"Hill_dataonly", job_id=args.job_id
    )
    print(f"[done] best params at: {params_csv}")


if __name__ == "__main__":
    main()
