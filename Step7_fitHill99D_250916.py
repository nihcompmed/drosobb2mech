#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit Hill model using a VAE-only teacher (no latent ODE).

Teacher (per (cid,t)->(cid,t+1)):
  z_t  = Enc([g_t | xyz_t?]),  z_t1 = Enc([g_{t+1} | xyz_{t+1}?])
  dz   = z_t1 - z_t   (constant latent velocity)
  For s in linspace(0,1,n_sub+1):
     z(s)   = z_t + s * dz
     g(s)   = Dec(z(s))
     gdot(s)= J_dec(z(s)) @ dz     (via autograd JVP)

Loss:
  total = w_deriv(epoch)*DerivLoss + w_int(epoch)*StateLoss + w_coll(epoch)*Collocation + L1(V)

- DerivLoss: robust erf-smooth L1 on gdot(s) vs Hill(g(s))
- StateLoss: MSE on 1-step RK4 prediction from g_t to g_{t+1}
- Collocation: low-cost Euler defect between g_t and predicted next
- L1(V): same lasso (only on V) as your old trainer

Scheduling:
  --sched_start_deriv <E0>, --sched_full_int <E1>
    epochs <= E0: derivative-heavy
    epochs >= E1: integration-heavy
    epochs in (E0,E1): cosine blend

CLI example (100 epochs troubleshooting):
  python Step7_fitHill_250905.py \
    --imputed_csv all_ctgxyz_27genes_fillrand_fillzero_t012345_noshuffle.csv \
    --vae_class Step2_trainVAE_penalizeJacdec_27g_ver3_lamjacschedule:VAE \
    --vae_ckpt checkpoints/your_vae.ckpt \
    --n_sub 5 --int_substeps 5 \
    --bsize 1024 --max_epochs 100 --ep_maxlr 25 \
    --maxlr 3e-3 --l1_lambda 1e-5 \
    --sched_start_deriv 30 --sched_full_int 80 \
    --job_id 250905001
"""
import glob, re

import os
import sys
import importlib
import importlib.util
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

# --- dynamic import helper (file-or-module name) ---
def _dynamic_import(modname: str):
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError:
        # Try with CWD on sys.path
        if os.getcwd() not in sys.path:
            sys.path.append(os.getcwd())
        return importlib.import_module(modname)

# ===== VAE loader (module:Class) =====
def load_vae(vae_class: str, vae_ckpt: str):
    """
    vae_class like 'Step2_trainVAE_penalizeJacdec_27g_ver3_lamjacschedule:VAE'
    Returns:
      encode_mu(x): [B,input_dim] -> [B,zdim] (posterior mean; no grad)
      decode(z)   : [B,zdim] -> [B,output_dim] (with grad, needed for JVP)
      expects_xyz : bool (True if input_dim = output_dim + 3)
      out_dim     : int (decoder output dimension == #genes)
    """
    modname, clsname = vae_class.split(":")
    module = _dynamic_import(modname)
    VAELit = getattr(module, clsname)

    vae: nn.Module = VAELit.load_from_checkpoint(vae_ckpt, strict=False).to(DEVICE).eval()

    # Try to get dims from hparams; fall back to attributes if needed
    hp = getattr(vae, "hparams", vae)
    out_dim = int(getattr(hp, "output_dim", 0) or getattr(vae, "output_dim", 0) or 0)
    in_dim  = int(getattr(hp, "input_dim",  0) or getattr(vae, "input_dim",  0) or 0)
    expects_xyz = (in_dim > 0 and out_dim > 0 and (in_dim - out_dim) == 3)

    # support either .encode or ._encode
    def _enc(x: torch.Tensor):
        if hasattr(vae, "encode"):
            mu, _ = vae.encode(x)
        else:
            mu, _ = vae._encode(x)
        return mu

    @torch.no_grad()
    def encode_mu(x: torch.Tensor) -> torch.Tensor:
        return _enc(x)

    # DO NOT @no_grad — we need autograd for decoder JVP
    def decode(z: torch.Tensor) -> torch.Tensor:
        return vae.decoder(z)

    return encode_mu, decode, expects_xyz, out_dim

# ===== Build teacher derivatives (VAE-only) =====
def build_pushfwd_from_vae(imputed_csv: str,
                           encode_mu, decode,
                           n_sub: int,
                           expects_xyz: bool,
                           G: int,
                           bsize_pairs: int = 4096):
    """
    Build (Xd, Yd) for derivative supervision and (Xn, Xn1, T) for state supervision.

    imputed_csv columns:
      [cid, time, g1..gG, (optional x y z)]

    If VAE expects xyz but csv lacks them, zeros are used for xyz.
    """
    data = pd.read_csv(imputed_csv, header=None).to_numpy().astype(np.float64)
    cid  = data[:, 0].astype(np.int64)
    time = data[:, 1].astype(np.int64)

    ncol = data.shape[1]
    has_xyz = (ncol >= 2 + G + 3)
    genes = data[:, 2:2+G]
    xyz   = data[:, 2+G:2+G+3] if has_xyz else np.zeros((genes.shape[0], 3), dtype=np.float64)

    # index pairs (cid,t)->(cid,t+1)
    key2idx = {(int(c), int(t)): i for i, (c, t) in enumerate(zip(cid, time))}
    pairs = [(int(c), int(t)) for c, t in zip(cid, time) if (int(c), int(t)+1) in key2idx]
    if not pairs:
        raise RuntimeError("No (cid,t)->(cid,t+1) pairs found in imputed_csv")

    Xd_list, Yd_list = [], []
    Xn_list, Xn1_list, T_list = [], [], []

    prev_flag = torch.is_grad_enabled()
    torch.set_grad_enabled(True)  # ensure JVP can run

    for i in range(0, len(pairs), bsize_pairs):
        batch = pairs[i:i+bsize_pairs]
        idx_t  = [key2idx[(c, t)]   for c, t in batch]
        idx_t1 = [key2idx[(c, t+1)] for c, t in batch]

        g_t  = torch.tensor(genes[idx_t],  device=DEVICE)
        g_t1 = torch.tensor(genes[idx_t1], device=DEVICE)

        # inputs to encoder (respect VAE input layout)
        if expects_xyz:
            x_t  = torch.cat([g_t,  torch.tensor(xyz[idx_t],  device=DEVICE)], dim=1)
            x_t1 = torch.cat([g_t1, torch.tensor(xyz[idx_t1], device=DEVICE)], dim=1)
        else:
            x_t, x_t1 = g_t, g_t1

        # latent means & constant velocity
        z_t  = encode_mu(x_t)
        z_t1 = encode_mu(x_t1)
        dz   = z_t1 - z_t

        # substeps
        s_grid = torch.linspace(0.0, 1.0, n_sub+1, dtype=torch.float64, device=DEVICE)  # [K]
        z_stack = torch.stack([z_t + s * dz for s in s_grid], dim=0)                    # [K,B,Z]

        # decode each substep
        g_stack = torch.stack([decode(z_stack[k]) for k in range(n_sub+1)], dim=0)      # [K,B,G]

        # JVP: J_dec(z) @ dz
        def jvp_batch(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            outs = []
            for zi, vi in zip(z, v):
                zi = zi.detach().requires_grad_(True)
                _, jvpi = torch.autograd.functional.jvp(
                    lambda _z: decode(_z.unsqueeze(0)).squeeze(0),
                    zi, vi, create_graph=False, strict=False
                )
                outs.append(jvpi)
            return torch.stack(outs, dim=0)

        gdots = torch.stack([jvp_batch(z_stack[k], dz) for k in range(n_sub+1)], dim=0)  # [K,B,G]

        # flatten
        K, B = n_sub+1, len(batch)
        Xd_list.append(g_stack.reshape(K*B, G).detach().cpu().numpy())
        Yd_list.append(gdots.reshape(K*B, G).detach().cpu().numpy())

        # 1-step supervision
        Xn_list.append(g_t.detach().cpu().numpy())
        Xn1_list.append(g_t1.detach().cpu().numpy())
        T_list.append(np.ones((B, 1), dtype=np.float64))

    torch.set_grad_enabled(prev_flag)

    Xd  = np.vstack(Xd_list)
    Yd  = np.vstack(Yd_list)
    Xn  = np.vstack(Xn_list)
    Xn1 = np.vstack(Xn1_list)
    T   = np.vstack(T_list)
    return Xd, Yd, Xn, Xn1, T

# ===== Dataset =====
class MixedDerivStateDataset(Dataset):
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
#        K_eff  = torch.clamp(self.K, min=1e-4, max=1e4)
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
        """Cosine-blend weights from derivative -> integration."""
        e0, e1 = self.start_deriv, self.full_int
        if epoch <= e0: return 1.0, 0.1, 0.3  # w_deriv, w_int, w_coll
        if epoch >= e1: return 0.2, 1.0, 0.2
        alpha = (epoch - e0) / max(1, (e1 - e0))
        cos_a = 0.5 - 0.5 * np.cos(np.pi * alpha)
        w_deriv = 1.0 * (1 - cos_a) + 0.2 * cos_a
        w_int   = 0.1 * (1 - cos_a) + 1.0 * cos_a
        w_coll  = 0.3 * (1 - cos_a) + 0.2 * cos_a
        return float(w_deriv), float(w_int), float(w_coll)

    def train(self, ds: MixedDerivStateDataset, bsize: int, val_frac=0.1,
              ckpt_dir="HillModels", name_prefix="hill_fromVAE", job_id="000"):
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
        params_csv = os.path.join(ckpt_dir, f"{name_prefix}_G{self.field.b0.numel()}_{job_id}.csv")

        for epoch in range(self.max_epochs):
            self.field.train()
            w_deriv, w_int, w_coll = self._weights(epoch)
            logd=logs=logc=logl=0.0; nstep=0

            for batch in train_loader:
                xd, yd, xn, xn1, T = [b.to(DEVICE) for b in batch]

                # derivative loss (robust)
                y_hat = self.field(xd)
                loss_deriv = erf_smooth_l1(y_hat - yd, self.sigma_vec)

                # 1-step RK4 state loss
                x_pred = rk4(self.field, xn, T, n_sub=self.int_sub)
                loss_state = self.loss_mse(x_pred, xn1)

                # collocation (Euler defect) between xn and x_pred
                K  = max(2, self.int_sub)
                dt = T / float(K - 1)
                xs = torch.stack([xn + (i/(K-1))*(x_pred - xn) for i in range(K)], dim=0)  # [K,B,G]
                fxs = torch.stack([self.field(xs[i]) for i in range(K-1)], dim=0)
                defects = xs[1:] - xs[:-1] - dt.unsqueeze(0) * fxs
                loss_coll = (defects**2).mean()

                # lasso (exactly as requested): L1 only on V
                l1_pen = self.l1_lambda * torch.abs(self.field.V).sum()

                loss = w_deriv*loss_deriv + w_int*loss_state + w_coll*loss_coll + l1_pen

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
                # monitor favors integration later
                monitor = 0.3*v_deriv + 0.7*v_state

            print(f"[ep {epoch:03d}] wD={w_deriv:.2f} wI={w_int:.2f} wC={w_coll:.2f}  "
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


def _pick_best(paths):
    """Choose the 'best' file deterministically:
       prefer names containing 'SMOOTH', then 'best', then latest mtime.
    """
    if not paths:
        return None
    def score(p):
        name = os.path.basename(p)
        s1 = 0 if ("SMOOTH" in name or "smooth" in name) else 1
        s2 = 0 if ("_best" in name or "best" in name) else 1
        s3 = -os.path.getmtime(p)
        return (s1, s2, s3)
    return sorted(paths, key=score)[0]

def find_imputed_by_jobid(jobid: str, root="ImputedOut_clean", tag="valid"):
    """
    Find imputed CSV like:
      ImputedOut_clean/Imputed_{jobid}_{tag}_ldim*.csv
    Also tolerate prefixes like Imputed_rand_*, Imputed_avg_*, Imputed_zero_*.
    """
    pats = [
        os.path.join(root, f"Imputed_{jobid}_{tag}_ldim*.csv"),
        os.path.join(root, f"Imputed_*_{jobid}_{tag}_ldim*.csv"),
    ]
    hits = []
    for pat in pats:
        hits.extend(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(
            f"No imputed CSV found for jobid={jobid}. "
            f"Tried patterns: {pats}"
        )
    pick = _pick_best(hits)
    # try to read ldim for logging
    m = re.search(r"ldim(\d+)", os.path.basename(pick))
    ldim = int(m.group(1)) if m else None
    return pick, ldim

def find_vae_ckpt_by_jobid(jobid: str, ckdir="checkpoints"):
    """
    Pick a VAE ckpt whose basename starts with 'VAE' and contains jobid.
    Examples:
      checkpoints/VAE_KO0g_ld9_..._250915032_...ckpt
      checkpoints/VAE_KO0g_ld9_..._250915032_..._SMOOTH_best....ckpt
    """
    all_hits = glob.glob(os.path.join(ckdir, f"*{jobid}*.ckpt"))
    hits = [p for p in all_hits if os.path.basename(p).startswith("VAE")]
    if not hits:
        raise FileNotFoundError(
            f"No VAE ckpt starting with 'VAE' found for jobid={jobid} in {ckdir}. "
            f"Candidates seen: {all_hits or 'NONE'}"
        )
    pick = _pick_best(hits)
    # try to read ldim for logging
    m = re.search(r"ldim(\d+)", os.path.basename(pick))
    ldim = int(m.group(1)) if m else None
    return pick, ldim


#===========MAIN===========
def main():
    import argparse
    ap = argparse.ArgumentParser("Fit Hill using VAE-only pushforward (no ODE)")
    ap.add_argument("--imputed_csv", type=str, default=None,
                    help="If omitted, will auto-find from --job_id under ImputedOut_clean/")
    ap.add_argument("--vae_class",   required=True, type=str,
                    help="e.g., Step2_trainVAE_importable_ref:VAE")
    ap.add_argument("--vae_ckpt",    type=str, default=None,
                    help="If omitted, will auto-find from --job_id under checkpoints/ (must start with 'VAE')")
    ap.add_argument("--n_sub",       type=int, default=5, help="Teacher subdivisions per unit interval")
    ap.add_argument("--int_substeps",type=int, default=5, help="RK4 substeps for state loss")
    ap.add_argument("--bsize",       type=int, default=1024)
    ap.add_argument("--max_epochs",  type=int, default=600)
    ap.add_argument("--ep_maxlr",    type=int, default=150)
    ap.add_argument("--maxlr",       type=float, default=3e-3)
    ap.add_argument("--l1_lambda",   type=float, default=1e-4)
    ap.add_argument("--sched_start_deriv", type=int, default=100)
    ap.add_argument("--sched_full_int",    type=int, default=300)
    ap.add_argument("--job_id",      type=str, required=True,
                    help="Base jobid used to auto-find files. Outputs will save with suffix 'hill'.")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)

    # --- auto-discover files from job_id if not provided explicitly ---
    if args.imputed_csv is None:
        args.imputed_csv, ldim_imp = find_imputed_by_jobid(args.job_id, root="ImputedOut_clean", tag="valid")
    else:
        ldim_imp = None

    if args.vae_ckpt is None:
        args.vae_ckpt, ldim_ckpt = find_vae_ckpt_by_jobid(args.job_id, ckdir="checkpoints")
    else:
        ldim_ckpt = None

    print("[auto]")
    print(f"  imputed_csv = {args.imputed_csv}  (ldim={ldim_imp})")
    print(f"  vae_ckpt    = {args.vae_ckpt}      (ldim={ldim_ckpt})")

    # load VAE
    encode_mu, decode, expects_xyz, out_dim = load_vae(args.vae_class, args.vae_ckpt)
    G = out_dim
    if G <= 0:
        raise RuntimeError("Could not infer VAE decoder output_dim (G). Ensure your VAE sets hparams.output_dim.")

    # build teacher supervision
    Xd, Yd, Xn, Xn1, T = build_pushfwd_from_vae(
        args.imputed_csv, encode_mu, decode,
        n_sub=args.n_sub, expects_xyz=expects_xyz, G=G, bsize_pairs=4096
    )

    # robust per-gene scales from derivative targets
    med = np.median(Yd, axis=0)
    mad = np.median(np.abs(Yd - med), axis=0)
    sigma_vec = np.maximum(1.4826 * mad, 1e-6)

    ds = MixedDerivStateDataset(Xd, Yd, Xn, Xn1, T)

    # schedule
    start = int(args.sched_start_deriv)
    full  = int(args.sched_full_int)


    # --- save jobid = base + 'hill' + '_l1{...}' ---
    l1_str = f"{args.l1_lambda:.0e}".replace("e-0", "en")  # e.g., 1e-4 -> "1en4"
    save_job_id = f"{args.job_id}hill_l1{l1_str}"


    trainer = HillTrainerTorch(
        G=G, max_lr=args.maxlr, ep_maxlr=args.ep_maxlr, max_epochs=args.max_epochs,
        l1_lambda=args.l1_lambda, int_substeps=args.int_substeps,
        sched_epochs=(start, full), sigma_vec=sigma_vec, wd=0.0
    )

    params_csv = trainer.train(
        ds, bsize=args.bsize, ckpt_dir="HillModels",
        name_prefix=f"Hill_fromVAEonly_sub{args.n_sub}_rk4{args.int_substeps}",
        job_id=save_job_id
    )

    print(f"[done] best params at: {params_csv}")


if __name__ == "__main__":
    main()
