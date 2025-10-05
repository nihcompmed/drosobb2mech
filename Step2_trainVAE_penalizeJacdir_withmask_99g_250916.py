#!/usr/bin/env python3
"""
Train VAE with masked reconstruction, optional encoder/decoder dropout,
and a smooth sparsity penalty on hidden Linear layers only.
Also supports KL warm-up and saves best-VAL and best-TRAIN checkpoints.

Hidden activation choices (arg `actfcn`):
  tanh, gelu, silu, mish, elu, celu, softplus, softplus0, leakyrelu, relu

Usage:
  python Step2_trainVAE_moretuning.py \
      <maxlr> <bsize> <max_epochs> <ep_maxlr> <N1> <N2> <N3> <priorstd> <beta> \
      <ldim> <actfcn> <actfcndecoder> <genes_to_knock> <fillingmethod> <job_id> \
      [<dec_dropout_p>] [<lambda_sparse>] [<sparsity_weight>] [<enc_dropout_p>] [<kl_warmup_epochs>]

Examples:
  # Baseline (no sparsity), decoder dropout 0.10, no encoder dropout
  python Step2_trainVAE_moretuning.py 1e-3 500 3000 1500 1024 512 256 0.10 1e-5 6 tanh sigmoid [] rand 250900001 0.10 0 0.0 0.0 300

  # Mild sparsity, gentle threshold, small weight, KL warm-up 300 epochs
  python Step2_trainVAE_moretuning.py 1e-3 500 3000 1500 1024 512 256 0.10 1e-5 6 tanh sigmoid [] rand 250900002 0.10 1e-2 1e-6 0.0 300
"""

import os, sys, ast, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ---- global default: float64 ----
torch.set_default_dtype(torch.float64)

# ─── CLI ─────────────────────────────────────────────────────────────────────
if len(sys.argv) < 16:
    print("Usage: python script.py maxlr bsize max_epochs ep_maxlr N1 N2 N3 priorstd beta "
          "ldim actfcn actfcndecoder genes_to_knock fillingmethod job_id "
          "[dec_dropout_p] [lambda_sparse] [sparsity_weight] [enc_dropout_p] [kl_warmup_epochs]")
    sys.exit(1)

maxlr      = float(sys.argv[1])
bsize      = int(sys.argv[2])
max_epochs = int(sys.argv[3])
ep_maxlr   = int(sys.argv[4])
N1         = int(sys.argv[5])
N2         = int(sys.argv[6])
N3         = int(sys.argv[7])
priorstd   = float(sys.argv[8])
beta       = float(sys.argv[9])
ldim       = int(sys.argv[10])
actfcn     = sys.argv[11].lower()         # hidden activation
finalact   = sys.argv[12].lower()         # decoder output activation (use 'sigmoid')
genes_to_knock = ast.literal_eval(sys.argv[13])
fillingmethod  = sys.argv[14].lower()
job_id     = int(sys.argv[15])

# Optional knobs
dec_dropout_p    = float(sys.argv[16]) if len(sys.argv) >= 17 else 0.10  # recommend 0.10
lambda_sparse    = float(sys.argv[17]) if len(sys.argv) >= 18 else 1e-5
sparsity_weight  = float(sys.argv[18]) if len(sys.argv) >= 19 else 0.0
enc_dropout_p    = float(sys.argv[19]) if len(sys.argv) >= 20 else 0.0
kl_warmup_epochs = int(sys.argv[20]) if len(sys.argv) >= 21 else 100
# Optional knobs (continue after kl_warmup_epochs)
lambda_jac      = float(sys.argv[21]) if len(sys.argv) >= 22 else 1e-4   # weight on Jacobian-size penalty
jac_eps         = float(sys.argv[22]) if len(sys.argv) >= 23 else 0.10   # FD step in latent
jac_probes      = int(sys.argv[23])  if len(sys.argv) >= 24 else 1       # Hutchinson probes
lambda_dir      = float(sys.argv[24]) if len(sys.argv) >= 25 else 0.0    # weight on directional supervision

print(f"Job ID: {job_id} | actfcn={actfcn} | enc_dropout_p={enc_dropout_p} | "
      f"dec_dropout_p={dec_dropout_p} | lambda_sparse={lambda_sparse} | "
      f"sparsity_weight={sparsity_weight} | kl_warmup_epochs={kl_warmup_epochs}")

GRAD_CLIP = float(os.getenv("GRAD_CLIP_VAL", "0.0"))

# ─── Data prep ───────────────────────────────────────────────────────────────
Ngenes    = 99
NKO       = len(genes_to_knock)
G_keep    = Ngenes - NKO
setinputdim  = G_keep + 3           # genes (after KO drop) + xyz
setoutputdim = G_keep               # decoder predicts only kept genes

# file with g_next added at the end
if fillingmethod == 'zero':
    data_fn = "all_ctgxyz_99genes_fillzero_fillzero_t012345_90percent_and_shuffled_tellnextg.csv"
elif fillingmethod == 'avg':
    data_fn = "all_ctgxyz_99genes_fillavg_fillzero_t012345_90percent_and_shuffled_tellnextg.csv"
elif fillingmethod == 'rand':
    data_fn = "all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled_tellnextg.csv"
else:
    raise ValueError(f"Unknown fillingmethod '{fillingmethod}'")

alltrain_tgv = pd.read_csv(data_fn, header=None)

# Load per-row, per-gene mask for CURRENT-time genes (shape: [N_rows, 99], 1=observed, 0=imputed)
mask_df = pd.read_csv("mask_99genes_fillrand_fillzero_t012345_90percent_and_shuffled.csv", header=None)
assert mask_df.shape[0] == alltrain_tgv.shape[0], "Mask rows must match data rows"
assert mask_df.shape[1] == 99, "Mask must have 99 gene columns for current-time genes"

# ----- column layout (0-based indices) -----
# [0]=cid, [1]=time,
# [2..100]=g_now(99), [101..103]=xyz(3), [104..202]=g_next(99)
COL_CID  = 0
COL_TIME = 1
IDX_G_NOW  = list(range(2, 2+Ngenes))            # 2..100
IDX_XYZ    = list(range(2+Ngenes, 2+Ngenes+3))   # 101..103
IDX_G_NEXT = list(range(2+Ngenes+3, 2+Ngenes+3+Ngenes))  # 104..202

# knockouts: convert to 0-based gene indices (0..98)
KO0 = [g-1 for g in genes_to_knock]
keep_mask = np.ones(Ngenes, dtype=bool)
keep_mask[KO0] = False

KEEP_G_NOW  = [IDX_G_NOW[i]  for i in range(Ngenes) if keep_mask[i]]
KEEP_G_NEXT = [IDX_G_NEXT[i] for i in range(Ngenes) if keep_mask[i]]

# keep cid, time, kept g_now, xyz, kept g_next (in that order)
keep_cols = [COL_CID, COL_TIME] + KEEP_G_NOW + IDX_XYZ + KEEP_G_NEXT
df_kept = alltrain_tgv.iloc[:, keep_cols].copy()

# Apply KO to mask (current-time only)
mask_now = mask_df.iloc[:, keep_mask].astype(bool).to_numpy()  # shape [N_rows, G_keep]

# has_next: only rows with time < 5 have a real next-timestep (others were filled with zeros)
time_col = alltrain_tgv.iloc[:, COL_TIME].to_numpy()
has_next_np = (time_col < 5)

# Convert to tensors (note: X layout now: [cid, time, g_now(G_keep), xyz(3), g_next(G_keep)])
Xtrain = torch.tensor(df_kept.values, dtype=torch.float64)
Xtrain[:, 2:2+G_keep] = Xtrain[:, 2:2+G_keep].abs()  # nonneg genes at current time

# Also enforce nonnegativity for g_next
gnext_beg = 2 + G_keep + 3
gnext_end = gnext_beg + G_keep
Xtrain[:, gnext_beg:gnext_end] = Xtrain[:, gnext_beg:gnext_end].abs()

Mask   = torch.tensor(mask_now, dtype=torch.bool)     # shape [N_rows, G_keep]
HasNext = torch.tensor(has_next_np, dtype=torch.bool) # shape [N_rows]


if fillingmethod == 'zero':
    present = ~torch.isnan(Xtrain)
    Xtrain = torch.where(present, Xtrain, torch.zeros_like(Xtrain))

class DrosoDataset(Dataset):
    def __init__(self, X, mask, has_next):
        self.X = X; self.mask = mask; self.has_next = has_next
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.mask[i], self.has_next[i]

# Split (keep alignment across X, mask, has_next)
train_X, val_X         = Xtrain[:27000], Xtrain[27000:]
train_mask, val_mask   = Mask[:27000],   Mask[27000:]
train_hn,   val_hn     = HasNext[:27000],HasNext[27000:]

NUM_CPU = os.cpu_count() or 4
num_workers = min(16, max(0, NUM_CPU - 2))
if not torch.cuda.is_available():
    num_workers = 0

train_loader = DataLoader(DrosoDataset(train_X, train_mask, train_hn), batch_size=bsize, shuffle=True,  num_workers=num_workers)
val_loader   = DataLoader(DrosoDataset(val_X,   val_mask,   val_hn),   batch_size=bsize, shuffle=False, num_workers=num_workers)

# ─── Extra activations ───────────────────────────────────────────────────────
class CenteredSoftplus(nn.Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.sp = nn.Softplus(beta=beta, threshold=threshold)
    def forward(self, x):
        return self.sp(x) - math.log(2.0)

def make_activation(name: str):
    name = name.lower()
    if name == "relu":      return nn.ReLU()
    if name == "tanh":      return nn.Tanh()
    if name == "gelu":      return nn.GELU()
    if name == "silu":      return nn.SiLU()      # swish
    if name == "mish":      return nn.Mish()
    if name == "elu":       return nn.ELU()
    if name == "celu":      return nn.CELU()
    if name == "softplus":  return nn.Softplus()
    if name == "softplus0": return CenteredSoftplus()
    if name == "leakyrelu": return nn.LeakyReLU(negative_slope=0.1)
    print(f"[warn] Unknown actfcn='{name}', falling back to tanh.")
    return nn.Tanh()

class Multiply(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x):  return self.c * x

# ─── Model ───────────────────────────────────────────────────────────────────
class VAE(pl.LightningModule):
    def __init__(self, max_lr, ndense1, ndense2, ndense3, beta, latent_dim,
                 input_dim=setinputdim, output_dim=setoutputdim,
                 enc_dropout_p=0.0, dec_dropout_p=0.10,
                 lambda_sparse=1e-5, sparsity_weight=0.0,
                 act_name="tanh", final_act="sigmoid",
                 kl_warmup_epochs=0,
                 lambda_jac=1e-4, jac_eps=0.10, jac_probes=1,   # from CLI
                 lambda_dir=0.0,                                # from CLI
                 dg_thresh=0.02,                                # (5) Δg threshold for directional loss
                 jac_eval_no_dropout=True):
        super().__init__()
        self.save_hyperparameters()
        self.beta = float(beta)
        self.max_lr = float(max_lr)
        self.latent_dim = int(latent_dim)

        # penalties / knobs
        self.lambda_sparse = float(lambda_sparse)
        self.sparsity_weight = float(sparsity_weight)
        self.kl_warmup_epochs = int(kl_warmup_epochs)
        self.lambda_jac  = float(lambda_jac)
        self.jac_eps     = float(jac_eps)
        self.jac_probes  = int(jac_probes)
        self.lambda_dir  = float(lambda_dir)
        self.dg_thresh   = float(dg_thresh)          # (5)
        self.jac_eval_no_dropout = bool(jac_eval_no_dropout)

        self.automatic_optimization = False

        act = make_activation(act_name)
        self.enc_drop = nn.Dropout(p=enc_dropout_p) if enc_dropout_p > 0 else nn.Identity()

        # Encoder hidden stack
        self.enc_fc1 = nn.Linear(input_dim, ndense1, dtype=torch.float64)
        self.enc_fc2 = nn.Linear(ndense1, ndense1, dtype=torch.float64)
        self.enc_fc3 = nn.Linear(ndense1, ndense2, dtype=torch.float64)
        self.enc_fc4 = nn.Linear(ndense2, ndense3, dtype=torch.float64)
        self.act = act

        # Latent heads
        self.mu_layer      = nn.Linear(ndense3, latent_dim, dtype=torch.float64)
        self.log_var_layer = nn.Linear(ndense3, latent_dim, dtype=torch.float64)

        # Decoder hidden stack (decoder dropout; recommend 0.10)
        dec = []
        dec += [nn.Linear(latent_dim, ndense3, dtype=torch.float64), act, nn.Dropout(p=dec_dropout_p)]
        dec += [nn.Linear(ndense3, ndense2, dtype=torch.float64), act, nn.Dropout(p=dec_dropout_p)]
        dec += [nn.Linear(ndense2, ndense1, dtype=torch.float64), act, nn.Dropout(p=dec_dropout_p)]
        dec += [nn.Linear(ndense1, ndense1, dtype=torch.float64), act]
        dec += [nn.Linear(ndense1, output_dim, dtype=torch.float64)]
        if final_act == "sigmoid":
            dec += [nn.Sigmoid(), Multiply(3.18)]
        else:
            raise ValueError("Invalid final activation for decoder. Use 'sigmoid'.")
        self.decoder = nn.Sequential(*dec)

        # collect linear layers to penalize (encoder hidden + all decoder linears)
        self._sparse_layers = [self.enc_fc1, self.enc_fc2, self.enc_fc3, self.enc_fc4]
        self._sparse_layers += [m for m in self.decoder if isinstance(m, nn.Linear)]

        # Logs
        self.train_losses_ae, self.train_losses_recons = [], []
        self.train_losses_kl, self.train_losses_sparse = [], []
        self.validation_step_outputs = []

        self.double()

    def _encode(self, x):
        x = self.enc_drop(self.act(self.enc_fc1(x)))
        x = self.enc_drop(self.act(self.enc_fc2(x)))
        x = self.enc_drop(self.act(self.enc_fc3(x)))
        x = self.enc_drop(self.act(self.enc_fc4(x)))
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        log_var = torch.clamp(log_var, min=-8.0, max=8.0)  # numerical safety
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = priorstd * torch.randn_like(std, dtype=torch.float64)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self._encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z)

    @staticmethod
    def _masked_recon_loss(decoded, target_g, mask_bool):
        se = (decoded - target_g) ** 2
        m  = mask_bool.float()
        num = torch.sum(se * m)
        den = m.sum() + 1e-8
        return num / den

    def _sparsity_penalty(self):
        """Smooth sparsity over selected Linear layers:
           sparsity_weight * sum_ij [ w_ij * tanh(w_ij / lambda_sparse) ]
        """
        if self.sparsity_weight <= 0.0 or self.lambda_sparse <= 0.0:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        lam = torch.as_tensor(self.lambda_sparse, dtype=torch.float64, device=self.device)
        penalty = torch.zeros((), dtype=torch.float64, device=self.device)
        for layer in self._sparse_layers:
            W = layer.weight
            penalty = penalty + torch.sum(W * torch.tanh(W / lam))
        return self.sparsity_weight * penalty

    def _jacobian_penalty(self, z_mu, mask_bool):
        if self.lambda_jac <= 0.0:
            return z_mu.new_zeros(())
        eps    = self.jac_eps
        probes = max(1, self.jac_probes)
        z0 = z_mu.detach()
        was_training = self.decoder.training
        if self.jac_eval_no_dropout:
            self.decoder.eval()
        pen = z0.new_zeros(())
        for _ in range(probes):
            r = torch.empty_like(z0).bernoulli_(0.5).mul_(2).sub_(1)
            g_plus  = self.decoder(z0 + eps * r)
            g_minus = self.decoder(z0 - eps * r)
            Jv = (g_plus - g_minus) / (2.0 * eps)   # [B,G]
            # ⬇ mask the squared norm
            Jv = Jv * mask_bool.float()
            pen = pen + (Jv.pow(2).sum(dim=1)).mean()
        if self.jac_eval_no_dropout and was_training:
            self.decoder.train(True)
        return self.lambda_jac * (pen / probes)

    def _directional_penalty(self, mu_t, g_t, g_next, mask_bool, has_next, xyz):
        if self.lambda_dir <= 0.0:
            return mu_t.new_zeros(())

        B, G = g_t.shape
        eps = self.jac_eps
        m = mask_bool.float()
        has = has_next.float().view(-1, 1)

        # build x_next to get mu_{t+1}
        x_next = torch.cat([g_next, xyz], dim=1)
        mu_next = self._encode_mu(x_next)
        mu_curr = mu_t.detach()
        dmu = mu_next - mu_curr
        dmu_norm = dmu.norm(dim=1, keepdim=True).clamp_min(1e-12)
        r = dmu / dmu_norm

        # FD Jv at mu_t
        was_training = self.decoder.training
        if self.jac_eval_no_dropout:
            self.decoder.eval()
        g_plus  = self.decoder(mu_curr + eps * r)
        g_minus = self.decoder(mu_curr - eps * r)
        if self.jac_eval_no_dropout and was_training:
            self.decoder.train(True)
        Jv = (g_plus - g_minus) / (2.0 * eps)  # [B,G]

        # mask both Δg and Jv
        dg = (g_next - g_t) * m
        Jv_m = Jv * m

        num = (Jv_m * dg).sum(dim=1)
        den = (Jv_m.norm(dim=1) * dg.norm(dim=1)).clamp_min(1e-12)
        cos_sim = num / den

        usable = (has.squeeze(1) > 0.5) & (dg.norm(dim=1) > self.dg_thresh)
        if usable.any():
            loss = (1.0 - cos_sim[usable]).mean()
        else:
            loss = mu_t.new_zeros(())

        return self.lambda_dir * loss


    @torch.no_grad()
    def _encode_mu(self, x):
        mu, _ = self._encode(x)
        return mu.detach()



    # ---- NEW: validation-time smoothness proxy (unweighted) ----
    def _jacobian_size_eval(self, z_mu: torch.Tensor) -> torch.Tensor:
        """
        Unweighted Jacobian-size proxy on the decoder: E[||J_dec(z) v||^2]
        using central finite differences and Hutchinson probes.
        Used only for validation-time smoothness monitoring.
        """
        with torch.no_grad():
            was_training = self.decoder.training
            if self.jac_eval_no_dropout:
                self.decoder.eval()

            eps    = self.jac_eps
            probes = max(1, self.jac_probes)
            z0 = z_mu.detach()
            acc = z0.new_zeros(())

            for _ in range(probes):
                r = torch.empty_like(z0).bernoulli_(0.5).mul_(2).sub_(1)
                g_plus  = self.decoder(z0 + eps * r)
                g_minus = self.decoder(z0 - eps * r)
                Jv = (g_plus - g_minus) / (2.0 * eps)
                acc = acc + (Jv.pow(2).sum(dim=1)).mean()

            if self.jac_eval_no_dropout and was_training:
                self.decoder.train(True)

            return acc / probes

    def training_step(self, batch, batch_idx):
        # batch: tgv_row, mask_now_per_gene, has_next_row
        tgv, mask, has_next = batch
        B = tgv.size(0)
        G = self.hparams.output_dim  # == G_keep

        # Slice: [cid, time, g_now(G), xyz(3), g_next(G)]
        g_beg = 2
        g_end = g_beg + G
        xyz_beg = g_end
        xyz_end = xyz_beg + 3
        gnext_beg = xyz_end
        gnext_end = gnext_beg + G

        g   = tgv[:, g_beg:g_end]           # [B,G]
        xyz = tgv[:, xyz_beg:xyz_end]       # [B,3]
        g_n = tgv[:, gnext_beg:gnext_end]   # [B,G]

        # Build encoder input x = [g_now | xyz]
        x = torch.cat([g, xyz], dim=1)

        opt = self.optimizers()
        opt.zero_grad()

        # main VAE path
        mu, log_var = self._encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)


                # penalties (masked)
        jac_pen = self._jacobian_penalty(mu, mask)
        dir_pen = self._directional_penalty(mu, g, g_n, mask, has_next, xyz)

        # ---- Ramps (Jacobian + Directional) ----
        def ramp_factor(mult, start, end):
            if end <= start: end = start + 1
            if self.current_epoch < start:
                return 0.0
            elif self.current_epoch >= end:
                return mult
            else:
                return 1.0 + (mult - 1.0) * ((self.current_epoch - start) / float(end - start))

        jac_mult = float(os.getenv("JAC_RAMP_MULT", "1.0"))
        jac_start = int(os.getenv("JAC_RAMP_START", str(max_epochs // 2)))
        jac_end   = int(os.getenv("JAC_RAMP_END",   str(int(max_epochs * 0.9))))
        dir_mult  = float(os.getenv("DIR_RAMP_MULT", "1.0"))
        dir_start = int(os.getenv("DIR_RAMP_START", str(max_epochs // 2)))
        dir_end   = int(os.getenv("DIR_RAMP_END",   str(int(max_epochs * 0.9))))

        jac_pen = jac_pen * ramp_factor(jac_mult, jac_start, jac_end)
        dir_pen = dir_pen * ramp_factor(dir_mult, dir_start, dir_end)


        # core losses
        recon_loss = self._masked_recon_loss(decoded, g, mask)
        kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        sparse_pen = self._sparsity_penalty()

        # KL warm-up
        if self.kl_warmup_epochs and self.kl_warmup_epochs > 0:
            warm = min(1.0, (self.current_epoch + 1) / float(self.kl_warmup_epochs))
        else:
            warm = 1.0
        kl_weight = self.beta * warm

        loss = recon_loss + kl_weight * kl_div + sparse_pen + jac_pen + dir_pen
        self.log("train_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)

        self.manual_backward(loss)
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), GRAD_CLIP, error_if_nonfinite=False)
        opt.step()

        # bookkeeping
        self.train_losses_ae.append(loss.item())
        self.train_losses_recons.append(recon_loss.item())
        self.train_losses_kl.append(kl_div.item())
        self.train_losses_sparse.append(sparse_pen.item())
        return loss

    def validation_step(self, batch, batch_idx):
        tgv, mask, has_next = batch  # has_next unused here

        G = self.hparams.output_dim
        g_beg = 2
        g_end = g_beg + G
        xyz_beg = g_end
        xyz_end = xyz_beg + 3

        g = tgv[:, g_beg:g_end]
        xyz = tgv[:, xyz_beg:xyz_end]
        x = torch.cat([g, xyz], dim=1)

        mu, log_var = self._encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)

        recon_loss = self._masked_recon_loss(decoded, g, mask)
        kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        val_loss = recon_loss + self.beta * kl_div

        # ---- NEW: smoothness proxy + combined validation metric ----
        val_jac = self._jacobian_size_eval(mu)
        smooth_alpha = float(os.getenv("SMOOTH_ALPHA", "0.10"))
        val_combo = val_loss + smooth_alpha * val_jac

        self.validation_step_outputs.append(val_loss)
        self.log('val_loss',  val_loss,  on_epoch=True, prog_bar=True)
        self.log('val_jac',   val_jac,   on_epoch=True, prog_bar=False)
        self.log('val_combo', val_combo, on_epoch=True, prog_bar=True)
        # ------------------------------------------------------------

        return val_loss

    def configure_optimizers(self):
        # (4) Separate param groups with lower LR for decoder
        wd = float(os.getenv("WEIGHT_DECAY", "0.0"))
        dec_lr_mult = float(os.getenv("DEC_LR_MULT", "0.5"))  # decoder LR multiplier

        enc_params = (
            list(self.enc_fc1.parameters()) + list(self.enc_fc2.parameters()) +
            list(self.enc_fc3.parameters()) + list(self.enc_fc4.parameters()) +
            list(self.mu_layer.parameters()) + list(self.log_var_layer.parameters())
        )
        dec_params = list(self.decoder.parameters())

        optimizer = optim.AdamW(
            [
                {"params": enc_params, "lr": self.max_lr},
                {"params": dec_params, "lr": self.max_lr * dec_lr_mult},
            ],
            weight_decay=wd,
            foreach=False
        )

        # --- Warmup → Cosine Decay schedule (applies to both param groups) ---
        warmup_epochs = int(os.getenv("WARMUP_EPOCHS", str(ep_maxlr)))
        start_factor  = float(os.getenv("WARMUP_START_FACTOR", "0.01"))
        eta_min_frac  = float(os.getenv("COSINE_ETA_MIN_FRAC", "0.05"))

        warmup_epochs = max(0, min(warmup_epochs, max_epochs))
        cosine_epochs = max(1, max_epochs - warmup_epochs)

        warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.max_lr * eta_min_frac
        )
        sched = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

        # Let Lightning step per epoch (same as your original pattern)
        return [optimizer], [{"scheduler": sched, "interval": "epoch", "name": "warmup_cosine"}]

    def on_fit_start(self):
        os.makedirs("traininglosses", exist_ok=True)
        self.train_csv_path = f"traininglosses/{job_id}_train_loss.csv"
        self.val_csv_path   = f"traininglosses/{job_id}_val_loss.csv"
        # write headers once
        with open(self.train_csv_path, "w") as f:
            f.write("epoch,train_loss\n")
        with open(self.val_csv_path, "w") as f:
            f.write("epoch,val_loss\n")

    def on_validation_epoch_end(self):
        # average of validation_step outputs for this epoch
        if self.validation_step_outputs:
            mean_val = float(np.mean([float(t.detach().cpu()) for t in self.validation_step_outputs]))
        else:
            mean_val = 0.0
        with open(self.val_csv_path, "a") as f:
            f.write(f"{self.current_epoch+1},{mean_val}\n")
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        # aggregate per-epoch stats
        mean_loss_ae     = float(np.mean(self.train_losses_ae))     if self.train_losses_ae     else 0.0
        mean_loss_recons = float(np.mean(self.train_losses_recons)) if self.train_losses_recons else 0.0
        mean_loss_kl     = float(np.mean(self.train_losses_kl))     if self.train_losses_kl     else 0.0
        mean_loss_sparse = float(np.mean(self.train_losses_sparse)) if self.train_losses_sparse else 0.0

        # print
        print(f"Epoch {self.current_epoch + 1}: "
              f"Train={mean_loss_ae:.8f} (recon={mean_loss_recons:.8f}, KL={mean_loss_kl:.8f}, sparse={mean_loss_sparse:.8f})")

        # step LR scheduler once per epoch (manual optimization)
        scheds = self.lr_schedulers()
        if scheds is not None:
            if isinstance(scheds, list):
                for sch in scheds: sch.step()
            else:
                scheds.step()

        # append one line to train CSV
        with open(self.train_csv_path, "a") as f:
            f.write(f"{self.current_epoch+1},{mean_loss_ae},{mean_loss_recons},{mean_loss_kl},{mean_loss_sparse}\n")

        # clear per-step buffers
        self.train_losses_ae.clear()
        self.train_losses_recons.clear()
        self.train_losses_kl.clear()
        self.train_losses_sparse.clear()

# ─── Train ───────────────────────────────────────────────────────────────────
maxlr_str = f"{maxlr:.0e}".replace("e-0", "en")
filename_template = (
    f"VAE_KO{NKO}g_ld{ldim}_{actfcn}"
    f"_sparse{lambda_sparse}_sw{str(sparsity_weight)}"
    f"_encdo{enc_dropout_p}_decdo{dec_dropout_p}"
    f"_lr{maxlr_str}_maxep{max_epochs}_{job_id}"
    f"_best{{epoch:04d}}-{{val_loss:.5f}}"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=5000,
    verbose=True,
    mode="min",
)

# Best validation checkpoint (and last)
checkpoint_val = ModelCheckpoint(
    dirpath="checkpoints",
    filename=filename_template,
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    verbose=True,
    save_last=False,
)

# Best training checkpoint (monitors epoch-aggregated train_loss)
checkpoint_train = ModelCheckpoint(
    dirpath="checkpoints",
    filename=filename_template + "_TRAIN_{epoch:04d}-{train_loss:.5f}",
    monitor="train_loss",
    mode="min",
    save_top_k=1,
    verbose=True,
)

# ---- NEW: Smoothness-aware checkpoint (val_loss + α·val_jac) ----
checkpoint_smooth = ModelCheckpoint(
    dirpath="checkpoints",
    filename=filename_template + "_SMOOTH_best{epoch:04d}-{val_combo:.5f}",
    monitor="val_combo",
    mode="min",
    save_top_k=1,
    verbose=True,
)

vae_model = VAE(
    max_lr=maxlr, ndense1=N1, ndense2=N2, ndense3=N3, beta=beta, latent_dim=ldim,
    input_dim=setinputdim, output_dim=setoutputdim,
    enc_dropout_p=enc_dropout_p, dec_dropout_p=dec_dropout_p,
    lambda_sparse=lambda_sparse, sparsity_weight=sparsity_weight,
    act_name=actfcn, final_act=finalact,
    kl_warmup_epochs=kl_warmup_epochs,
    lambda_jac=lambda_jac, jac_eps=jac_eps, jac_probes=jac_probes,
    lambda_dir=lambda_dir,
    dg_thresh=0.02,                        # (5) default threshold (you can tweak)
    jac_eval_no_dropout=True
)

GRAD_CLIP_VAL = float(os.getenv("GRAD_CLIP_VAL", "0.0"))

trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision="64-true",
    #gradient_clip_val=GRAD_CLIP_VAL,    # manual optimization -> we clip manually in training_step
    #gradient_clip_algorithm="norm",
    log_every_n_steps=10,
    callbacks=[early_stopping, checkpoint_val, checkpoint_train, checkpoint_smooth],
    logger=False,
)
trainer.fit(vae_model, train_loader, val_loader)
