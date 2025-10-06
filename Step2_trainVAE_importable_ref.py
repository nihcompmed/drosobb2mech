#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import-safe VAE (27 genes) with masked reconstruction, optional dropout,
sparsity penalty on Linear layers, Jacobian smoothing, and directional loss.
- Top level only defines classes/helpers so it can be safely imported by other scripts.
- CLI / data / training live inside main() and run only when called as a script.

You can train from this file (same functionality as your original),
and also import `VAE` elsewhere without triggering any CLI code.

Usage example (same shape as your prior workflow):
  python Step2_trainVAE_importable_ref.py \
      1e-3 500 3000 1500 1024 512 256 0.10 1e-5 \
      12 gelu sigmoid [] rand 250903218 \
      0.10 1e-5 0.0 0.0 300 1e-4 0.10 1 0.0
"""

import os, sys, ast, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# ---- global default: float64 ----
torch.set_default_dtype(torch.float64)

# ───────────────────────────── Helpers (importable) ──────────────────────────
class CenteredSoftplus(nn.Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.sp = nn.Softplus(beta=beta, threshold=threshold)
    def forward(self, x): return self.sp(x) - math.log(2.0)

def make_activation(name: str):
    name = name.lower()
    if name == "relu":      return nn.ReLU()
    if name == "tanh":      return nn.Tanh()
    if name == "gelu":      return nn.GELU()
    if name == "silu":      return nn.SiLU()   # swish
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

# ───────────────────────────── VAE (importable) ──────────────────────────────
class VAE(pl.LightningModule):
    """
    Identical behavior to your original class, but:
      - no reliance on module-level globals;
      - optional max_epochs/ep_maxlr/priorstd stored in hparams;
      - reparameterize() uses self._priorstd (default 0.0) instead of a global.
    """
    def __init__(
        self,
        max_lr, ndense1, ndense2, ndense3, beta, latent_dim,
        input_dim, output_dim,
        enc_dropout_p=0.0, dec_dropout_p=0.10,
        lambda_sparse=1e-5, sparsity_weight=0.0,
        act_name="tanh", final_act="sigmoid",
        kl_warmup_epochs=0,
        lambda_jac=1e-4, jac_eps=0.10, jac_probes=1,
        lambda_dir=0.0, dg_thresh=0.02,
        jac_eval_no_dropout=True,
        # new optional knobs (backwards compatible with checkpoints)
        max_epochs: int = None, ep_maxlr: int = None, priorstd: float = 0.0,
        job_id: int = 0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.beta = float(beta)
        self.max_lr = float(max_lr)
        self.latent_dim = int(latent_dim)
        self._priorstd = float(priorstd)  # used by reparameterize()

        # penalties / knobs
        self.lambda_sparse = float(lambda_sparse)
        self.sparsity_weight = float(sparsity_weight)
        self.kl_warmup_epochs = int(kl_warmup_epochs)
        self.lambda_jac  = float(lambda_jac)
        self.jac_eps     = float(jac_eps)
        self.jac_probes  = int(jac_probes)
        self.lambda_dir  = float(lambda_dir)
        self.dg_thresh   = float(dg_thresh)
        self.jac_eval_no_dropout = bool(jac_eval_no_dropout)

        self.automatic_optimization = False

        act = make_activation(act_name)
        self.enc_drop = nn.Dropout(p=enc_dropout_p) if enc_dropout_p > 0 else nn.Identity()

        # Encoder
        self.enc_fc1 = nn.Linear(input_dim, ndense1, dtype=torch.float64)
        self.enc_fc2 = nn.Linear(ndense1, ndense1, dtype=torch.float64)
        self.enc_fc3 = nn.Linear(ndense1, ndense2, dtype=torch.float64)
        self.enc_fc4 = nn.Linear(ndense2, ndense3, dtype=torch.float64)
        self.act = act

        # Latent heads
        self.mu_layer      = nn.Linear(ndense3, latent_dim, dtype=torch.float64)
        self.log_var_layer = nn.Linear(ndense3, latent_dim, dtype=torch.float64)

        # Decoder
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

        # layers to sparsify
        self._sparse_layers = [self.enc_fc1, self.enc_fc2, self.enc_fc3, self.enc_fc4]
        self._sparse_layers += [m for m in self.decoder if isinstance(m, nn.Linear)]

        # logs
        self.train_losses_ae, self.train_losses_recons = [], []
        self.train_losses_kl, self.train_losses_sparse = [], []
        self.validation_step_outputs = []

        self.double()

    # ---- utilities ----
    def _hp(self, name, default):
        try:
            return getattr(self.hparams, name)
        except Exception:
            return default

    def _encode(self, x):
        x = self.enc_drop(self.act(self.enc_fc1(x)))
        x = self.enc_drop(self.act(self.enc_fc2(x)))
        x = self.enc_drop(self.act(self.enc_fc3(x)))
        x = self.enc_drop(self.act(self.enc_fc4(x)))
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        log_var = torch.clamp(log_var, min=-8.0, max=8.0)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        # sample with configured prior std (0 disables stochasticity)
        if self._priorstd and self._priorstd > 0.0:
            eps = torch.randn_like(std, dtype=torch.float64) * self._priorstd
        else:
            eps = std.new_zeros(std.shape)
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
        if self.sparsity_weight <= 0.0 or self.lambda_sparse <= 0.0:
            return torch.zeros((), dtype=torch.float64, device=self.device)
        lam = torch.as_tensor(self.lambda_sparse, dtype=torch.float64, device=self.device)
        penalty = torch.zeros((), dtype=torch.float64, device=self.device)
        for layer in self._sparse_layers:
            W = layer.weight
            penalty = penalty + torch.sum(W * torch.tanh(W / lam))
        return self.sparsity_weight * penalty

    def _jacobian_penalty(self, z_mu):
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
            Jv = (g_plus - g_minus) / (2.0 * eps)
            pen = pen + (Jv.pow(2).sum(dim=1)).mean()
        if self.jac_eval_no_dropout and was_training:
            self.decoder.train(True)
        return self.lambda_jac * (pen / probes)

    @torch.no_grad()
    def _encode_mu(self, x):
        mu, _ = self._encode(x)
        return mu.detach()

    def _directional_penalty(self, mu_t, g_t, g_next, mask_bool, has_next, xyz):
        if self.lambda_dir <= 0.0:
            return mu_t.new_zeros(())

        B, G = g_t.shape
        eps = self.jac_eps
        m = mask_bool.float()
        has = has_next.float().view(-1, 1)

        # build x_next = [g_next | xyz] to get mu_{t+1}
        x_next = torch.cat([g_next, xyz], dim=1)
        mu_next = self._encode_mu(x_next)
        mu_curr = mu_t.detach()
        dmu = mu_next - mu_curr
        dmu_norm = dmu.norm(dim=1, keepdim=True).clamp_min(1e-12)
        r = dmu / dmu_norm

        # FD Jv at mu_t (detached)
        was_training = self.decoder.training
        if self.jac_eval_no_dropout:
            self.decoder.eval()
        g_plus  = self.decoder(mu_curr + eps * r)
        g_minus = self.decoder(mu_curr - eps * r)
        if self.jac_eval_no_dropout and was_training:
            self.decoder.train(True)
        Jv = (g_plus - g_minus) / (2.0 * eps)

        # Cosine alignment with Δg (masked)
        dg = (g_next - g_t) * m
        Jv_m = Jv * m
        num = (Jv_m * dg).sum(dim=1)
        den = (Jv_m.norm(dim=1) * dg.norm(dim=1)).clamp_min(1e-12)
        cos_sim = num / den

        # rows usable: has_next & ‖Δg‖>dg_thresh
        usable = (has.squeeze(1) > 0.5) & (dg.norm(dim=1) > self.dg_thresh)
        if usable.any():
            loss = (1.0 - cos_sim[usable]).mean()
        else:
            loss = mu_t.new_zeros(())
        return self.lambda_dir * loss

    def _jacobian_size_eval(self, z_mu: torch.Tensor) -> torch.Tensor:
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

    # ---- Lightning hooks ----
    def training_step(self, batch, batch_idx):
        tgv, mask, has_next = batch
        G = self.hparams.output_dim

        # Slice: [cid, time, g_now(G), xyz(3), g_next(G)]
        g_beg = 2; g_end = g_beg + G
        xyz_beg = g_end; xyz_end = xyz_beg + 3
        gnext_beg = xyz_end; gnext_end = gnext_beg + G

        g   = tgv[:, g_beg:g_end]
        xyz = tgv[:, xyz_beg:xyz_end]
        g_n = tgv[:, gnext_beg:gnext_end]

        # Build encoder input x = [g_now | xyz]
        x = torch.cat([g, xyz], dim=1)

        opt = self.optimizers()
        opt.zero_grad()

        mu, log_var = self._encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)

        jac_pen = self._jacobian_penalty(mu)
        dir_pen = self._directional_penalty(mu, g, g_n, mask, has_next, xyz)

        # Late-stage ramp for Jacobian penalty (env overrides allowed)
        me = self._hp("max_epochs", 0)
        default_start = me // 2 if me else 0
        default_end   = int(me * 0.9) if me else 0
        ramp_start = int(os.getenv("JAC_RAMP_START", str(default_start)))
        ramp_end   = int(os.getenv("JAC_RAMP_END",   str(default_end)))
        ramp_mult  = float(os.getenv("JAC_RAMP_MULT", "1.0"))
        if ramp_end <= ramp_start: ramp_end = ramp_start + 1

        if self.current_epoch < ramp_start:
            ramp = 0.0
        elif self.current_epoch >= ramp_end:
            ramp = 1.0
        else:
            ramp = (self.current_epoch - ramp_start) / float(ramp_end - ramp_start)
        jac_pen = jac_pen * (1.0 + ramp * (ramp_mult - 1.0))

        # core losses
        recon_loss = self._masked_recon_loss(decoded, g, mask)
        kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        sparse_pen = self._sparsity_penalty()

        # KL warm-up
        warmup = self.kl_warmup_epochs if self.kl_warmup_epochs and self.kl_warmup_epochs > 0 else 0
        if warmup > 0:
            warm = min(1.0, (self.current_epoch + 1) / float(warmup))
        else:
            warm = 1.0
        kl_weight = self.beta * warm

        loss = recon_loss + kl_weight * kl_div + sparse_pen + jac_pen + dir_pen
        self.log("train_loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=True)

        self.manual_backward(loss)
        grad_clip = float(os.getenv("GRAD_CLIP_VAL", "0.0"))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip, error_if_nonfinite=False)
        opt.step()

        # bookkeeping
        self.train_losses_ae.append(loss.item())
        self.train_losses_recons.append(recon_loss.item())
        self.train_losses_kl.append(kl_div.item())
        self.train_losses_sparse.append(sparse_pen.item())
        return loss

    def validation_step(self, batch, batch_idx):
        tgv, mask, _ = batch
        G = self.hparams.output_dim
        g_beg = 2; g_end = g_beg + G
        xyz_beg = g_end; xyz_end = xyz_beg + 3

        g = tgv[:, g_beg:g_end]
        xyz = tgv[:, xyz_beg:xyz_end]
        x = torch.cat([g, xyz], dim=1)

        mu, log_var = self._encode(x)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)

        recon_loss = self._masked_recon_loss(decoded, g, mask)
        kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        val_loss = recon_loss + self.beta * kl_div

        val_jac = self._jacobian_size_eval(mu)
        smooth_alpha = float(os.getenv("SMOOTH_ALPHA", "0.10"))
        val_combo = val_loss + smooth_alpha * val_jac

        self.validation_step_outputs.append(val_loss)
        self.log('val_loss',  val_loss,  on_epoch=True, prog_bar=True)
        self.log('val_jac',   val_jac,   on_epoch=True, prog_bar=False)
        self.log('val_combo', val_combo, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        wd = float(os.getenv("WEIGHT_DECAY", "0.0"))
        dec_lr_mult = float(os.getenv("DEC_LR_MULT", "0.5"))

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

        # Warmup → Cosine Decay; defaults read from hparams if env not provided
        ep_maxlr = self._hp("ep_maxlr", 0)
        me       = self._hp("max_epochs", 1)
        warmup_epochs = int(os.getenv("WARMUP_EPOCHS", str(ep_maxlr)))
        start_factor  = float(os.getenv("WARMUP_START_FACTOR", "0.01"))
        eta_min_frac  = float(os.getenv("COSINE_ETA_MIN_FRAC", "0.05"))

        warmup_epochs = max(0, min(warmup_epochs, me))
        cosine_epochs = max(1, me - warmup_epochs)

        warmup = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=self.max_lr * eta_min_frac)
        sched = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        return [optimizer], [{"scheduler": sched, "interval": "epoch", "name": "warmup_cosine"}]

    def on_fit_start(self):
        os.makedirs("traininglosses", exist_ok=True)
        jid = self._hp("job_id", 0)
        self.train_csv_path = f"traininglosses/{jid}_train_loss.csv"
        self.val_csv_path   = f"traininglosses/{jid}_val_loss.csv"
        with open(self.train_csv_path, "w") as f:
            f.write("epoch,train_loss\n")
        with open(self.val_csv_path, "w") as f:
            f.write("epoch,val_loss\n")

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            mean_val = float(np.mean([float(t.detach().cpu()) for t in self.validation_step_outputs]))
        else:
            mean_val = 0.0
        with open(self.val_csv_path, "a") as f:
            f.write(f"{self.current_epoch+1},{mean_val}\n")
        self.validation_step_outputs.clear()

    def on_train_epoch_end(self):
        mean_loss_ae     = float(np.mean(self.train_losses_ae))     if self.train_losses_ae     else 0.0
        mean_loss_recons = float(np.mean(self.train_losses_recons)) if self.train_losses_recons else 0.0
        mean_loss_kl     = float(np.mean(self.train_losses_kl))     if self.train_losses_kl     else 0.0
        mean_loss_sparse = float(np.mean(self.train_losses_sparse)) if self.train_losses_sparse else 0.0

        print(f"Epoch {self.current_epoch + 1}: "
              f"Train={mean_loss_ae:.8f} (recon={mean_loss_recons:.8f}, KL={mean_loss_kl:.8f}, sparse={mean_loss_sparse:.8f})")

        scheds = self.lr_schedulers()
        if scheds is not None:
            if isinstance(scheds, list):
                for sch in scheds: sch.step()
            else:
                scheds.step()

        with open(self.train_csv_path, "a") as f:
            f.write(f"{self.current_epoch+1},{mean_loss_ae},{mean_loss_recons},{mean_loss_kl},{mean_loss_sparse}\n")

        self.train_losses_ae.clear()
        self.train_losses_recons.clear()
        self.train_losses_kl.clear()
        self.train_losses_sparse.clear()

# ────────────────────────────── main() (CLI) ─────────────────────────────────
def main():
    # ─── CLI ───────────────────────────────────────────────────────────────
    if len(sys.argv) < 16:
        print("Usage: python script.py maxlr bsize max_epochs ep_maxlr N1 N2 N3 priorstd beta "
              "ldim actfcn actfcndecoder genes_to_knock fillingmethod job_id "
              "[dec_dropout_p] [lambda_sparse] [sparsity_weight] [enc_dropout_p] [kl_warmup_epochs] "
              "[lambda_jac] [jac_eps] [jac_probes] [lambda_dir]")
        sys.exit(1)

    maxlr      = float(sys.argv[1]);  bsize      = int(sys.argv[2])
    max_epochs = int(sys.argv[3]);    ep_maxlr   = int(sys.argv[4])
    N1         = int(sys.argv[5]);    N2         = int(sys.argv[6]); N3 = int(sys.argv[7])
    priorstd   = float(sys.argv[8]);  beta       = float(sys.argv[9])
    ldim       = int(sys.argv[10]);   actfcn     = sys.argv[11].lower()
    finalact   = sys.argv[12].lower()
    genes_to_knock = ast.literal_eval(sys.argv[13])
    fillingmethod  = sys.argv[14].lower()
    job_id     = int(sys.argv[15])

    dec_dropout_p    = float(sys.argv[16]) if len(sys.argv) >= 17 else 0.10
    lambda_sparse    = float(sys.argv[17]) if len(sys.argv) >= 18 else 1e-5
    sparsity_weight  = float(sys.argv[18]) if len(sys.argv) >= 19 else 0.0
    enc_dropout_p    = float(sys.argv[19]) if len(sys.argv) >= 20 else 0.0
    kl_warmup_epochs = int(sys.argv[20]) if len(sys.argv) >= 21 else 100
    lambda_jac       = float(sys.argv[21]) if len(sys.argv) >= 22 else 1e-4
    jac_eps          = float(sys.argv[22]) if len(sys.argv) >= 23 else 0.10
    jac_probes       = int(sys.argv[23])  if len(sys.argv) >= 24 else 1
    lambda_dir       = float(sys.argv[24]) if len(sys.argv) >= 25 else 0.0

    print(f"Job ID: {job_id} | actfcn={actfcn} | enc_dropout_p={enc_dropout_p} | "
          f"dec_dropout_p={dec_dropout_p} | lambda_sparse={lambda_sparse} | "
          f"sparsity_weight={sparsity_weight} | kl_warmup_epochs={kl_warmup_epochs}")

    GRAD_CLIP = float(os.getenv("GRAD_CLIP_VAL", "0.0"))

    # ─── Data prep ─────────────────────────────────────────────────────────
    Ngenes    = 27
    NKO       = len(genes_to_knock)
    G_keep    = Ngenes - NKO
    setinputdim  = G_keep + 3           # genes (after KO drop) + xyz
    setoutputdim = G_keep               # decoder predicts only kept genes

    if fillingmethod == 'zero':
        data_fn = "all_tgvxyz_t012345_train5000cells_250507_shuffled.csv"
    elif fillingmethod == 'avg':
        data_fn = "all_ctgxyz_99g_fillavg_t012345_train5000cells_shuffled.csv"
    elif fillingmethod == 'rand':
        data_fn = "all_ctgxyz_27genes_fillrand_fillzero_t012345_90percent_and_shuffled_tellnextg.csv"
    else:
        raise ValueError(f"Unknown fillingmethod '{fillingmethod}'")

    alltrain_tgv = pd.read_csv(data_fn, header=None)

    # Create a 32820 x 27 matrix of ones as mask placeholder
    mask_array = np.ones((32820, 27), dtype=int)
    mask_df = pd.DataFrame(mask_array)
    print(mask_df.shape)  # should be (32820, 27)

    # column layout (0-based)
    COL_CID, COL_TIME = 0, 1
    IDX_G_NOW  = list(range(2, 2+Ngenes))
    IDX_XYZ    = list(range(2+Ngenes, 2+Ngenes+3))
    IDX_G_NEXT = list(range(2+Ngenes+3, 2+Ngenes+3+Ngenes))

    KO0 = [g-1 for g in genes_to_knock]
    keep_mask = np.ones(Ngenes, dtype=bool); keep_mask[KO0] = False

    KEEP_G_NOW  = [IDX_G_NOW[i]  for i in range(Ngenes) if keep_mask[i]]
    KEEP_G_NEXT = [IDX_G_NEXT[i] for i in range(Ngenes) if keep_mask[i]]

    keep_cols = [COL_CID, COL_TIME] + KEEP_G_NOW + IDX_XYZ + KEEP_G_NEXT
    df_kept = alltrain_tgv.iloc[:, keep_cols].copy()

    # boolean mask per-gene for CURRENT time (drop KOs)
    mask_now = mask_df.iloc[:, keep_mask].astype(bool).to_numpy()

    # has_next flag (time < 5)
    time_col = alltrain_tgv.iloc[:, COL_TIME].to_numpy()
    has_next_np = (time_col < 5)

    # X layout: [cid, time, g_now(G_keep), xyz(3), g_next(G_keep)]
    Xtrain = torch.tensor(df_kept.values, dtype=torch.float64)
    Xtrain[:, 2:2+G_keep] = Xtrain[:, 2:2+G_keep].abs()  # nonneg genes
    gnext_beg = 2 + G_keep + 3; gnext_end = gnext_beg + G_keep
    Xtrain[:, gnext_beg:gnext_end] = Xtrain[:, gnext_beg:gnext_end].abs()

    Mask    = torch.tensor(mask_now,   dtype=torch.bool)
    HasNext = torch.tensor(has_next_np,dtype=torch.bool)

    if fillingmethod == 'zero':
        present = ~torch.isnan(Xtrain)
        Xtrain = torch.where(present, Xtrain, torch.zeros_like(Xtrain))

    class DrosoDataset(Dataset):
        def __init__(self, X, mask, has_next):
            self.X = X; self.mask = mask; self.has_next = has_next
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.mask[i], self.has_next[i]

    train_X, val_X       = Xtrain[:27000], Xtrain[27000:]
    train_mask, val_mask = Mask[:27000],   Mask[27000:]
    train_hn,   val_hn   = HasNext[:27000],HasNext[27000:]

    NUM_CPU = os.cpu_count() or 4
    num_workers = min(16, max(0, NUM_CPU - 2))
    if not torch.cuda.is_available():
        num_workers = 0

    train_loader = DataLoader(DrosoDataset(train_X, train_mask, train_hn), batch_size=bsize, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(DrosoDataset(val_X,   val_mask,   val_hn),   batch_size=bsize, shuffle=False, num_workers=num_workers)

    # ─── Model & Trainer ───────────────────────────────────────────────────
    vae_model = VAE(
        max_lr=maxlr, ndense1=N1, ndense2=N2, ndense3=N3, beta=beta, latent_dim=ldim,
        input_dim=setinputdim, output_dim=setoutputdim,
        enc_dropout_p=enc_dropout_p, dec_dropout_p=dec_dropout_p,
        lambda_sparse=lambda_sparse, sparsity_weight=sparsity_weight,
        act_name=actfcn, final_act=finalact,
        kl_warmup_epochs=kl_warmup_epochs,
        lambda_jac=lambda_jac, jac_eps=jac_eps, jac_probes=jac_probes,
        lambda_dir=lambda_dir, dg_thresh=0.02, jac_eval_no_dropout=True,
        max_epochs=max_epochs, ep_maxlr=ep_maxlr, priorstd=priorstd, job_id=job_id
    )

    maxlr_str = f"{maxlr:.0e}".replace("e-0", "en")
    filename_template = (
        f"VAE_KO{len(genes_to_knock)}g_ld{ldim}_{actfcn}"
        f"_sparse{lambda_sparse}_sw{str(sparsity_weight)}"
        f"_encdo{enc_dropout_p}_decdo{dec_dropout_p}"
        f"_lr{maxlr_str}_maxep{max_epochs}_{job_id}"
        f"_best{{epoch:04d}}-{{val_loss:.5f}}"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5000,
        verbose=True, mode="min",
    )

    checkpoint_val = ModelCheckpoint(
        dirpath="checkpoints",
        filename=filename_template,
        monitor="val_loss", mode="min", save_top_k=1, verbose=True, save_last=False,
    )

    checkpoint_train = ModelCheckpoint(
        dirpath="checkpoints",
        filename=filename_template + "_TRAIN_{epoch:04d}-{train_loss:.5f}",
        monitor="train_loss", mode="min", save_top_k=1, verbose=True,
    )

    checkpoint_smooth = ModelCheckpoint(
        dirpath="checkpoints",
        filename=filename_template + "_SMOOTH_best{epoch:04d}-{val_combo:.5f}",
        monitor="val_combo", mode="min", save_top_k=1, verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="64-true",
        log_every_n_steps=10,
        callbacks=[early_stopping, checkpoint_val, checkpoint_train, checkpoint_smooth],
        logger=False,
    )

    trainer.fit(vae_model, train_loader, val_loader)

if __name__ == "__main__":
    main()
