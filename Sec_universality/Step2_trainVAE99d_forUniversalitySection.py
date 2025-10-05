#!/usr/bin/env python3
"""
Step2_trainVAE_universality.py
Train VAE for region-specific universality experiments (AMP/AP/DMV/DV/All33/All50).
- Uses masked reconstruction + KL only (no sparsity, no Jacobian, no directional loss).
- Region/subregion selection follows old universality code,
  but training dataset is the NEW fillrand/fillzero file with mask.
"""

import os, sys, pickle, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.optim.lr_scheduler as lr_scheduler

torch.set_default_dtype(torch.float64)

# ---------------- CLI args ----------------
if len(sys.argv) < 16:
    print("Usage: python script.py maxlr max_epochs ep_maxlr N1 N2 N3 priorstd beta ldim actfcn actfcndecoder regiontype subregion job_id seed")
    sys.exit(1)

maxlr      = float(sys.argv[1])
max_epochs = int(sys.argv[2])
ep_maxlr   = int(sys.argv[3])
N1, N2, N3 = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
priorstd   = float(sys.argv[7])
beta       = float(sys.argv[8])
ldim       = int(sys.argv[9])
actfcn     = sys.argv[10].lower()
finalact   = sys.argv[11].lower()
regiontype = sys.argv[12].lower()
subregion  = sys.argv[13].lower()
job_id     = int(sys.argv[14])
seed_select= int(sys.argv[15])

# ---------------- Data ----------------
ncells, ntime, Ngenes = 6078, 6, 99
setinputdim, setoutputdim = Ngenes + 3, Ngenes



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")


data_fn = os.path.join(DATA_DIR, "all_ctgxyz_99genes_fillrand_fillzero_t012345_90percent_and_shuffled.csv")
mask_fn = os.path.join(DATA_DIR, "mask_99genes_fillrand_fillzero_t012345_90percent_and_shuffled.csv")
test_pkl = os.path.join(DATA_DIR, "test_cells.pkl")

alltrain_tgv = pd.read_csv(data_fn, header=None).values
mask_df = pd.read_csv(mask_fn, header=None).values
with open(test_pkl, "rb") as f:
    test_idx = pickle.load(f)



# region selection uses t=0 rows from old-style dataset
df0_fn = os.path.join(DATA_DIR, "all_ctgxyz_99genes_fillrand_fillzero_t012345_noshuffle.csv")
df0 = pd.read_csv(df0_fn, header=None).values

cell_ids_remaining = np.setdiff1d(np.arange(1, ncells + 1), test_idx)
rows_t0 = df0[:ncells]
rows_t0 = rows_t0[~np.isin(rows_t0[:, 0], test_idx)]
xyz_t0 = rows_t0[:, 101:104]
x, z = xyz_t0[:, 0], xyz_t0[:, 2]
n_part = int(1/3 * len(x))
n_half = int(0.5 * len(x))
sorted_idx_x, sorted_idx_z = np.argsort(x), np.argsort(z)

if regiontype == "amp":
    a_cells = cell_ids_remaining[sorted_idx_x[:n_part]]
    m_cells = cell_ids_remaining[sorted_idx_x[n_part:2*n_part]]
    p_cells = cell_ids_remaining[sorted_idx_x[2*n_part:]]
elif regiontype == "ap":
    a_cells = cell_ids_remaining[sorted_idx_x[:n_half]]
    p_cells = cell_ids_remaining[sorted_idx_x[-n_half:]]
elif regiontype == "dmv":
    v_cells = cell_ids_remaining[sorted_idx_z[:n_part]]
    m_cells = cell_ids_remaining[sorted_idx_z[n_part:2*n_part]]
    d_cells = cell_ids_remaining[sorted_idx_z[2*n_part:]]
elif regiontype == "dv":
    v_cells = cell_ids_remaining[sorted_idx_z[:n_half]]
    d_cells = cell_ids_remaining[sorted_idx_z[-n_half:]]
elif regiontype == "all":
    select_rng = np.random.RandomState(seed_select)
    pool = cell_ids_remaining
    if subregion == "all33":
        n = len(pool) // 3
    elif subregion == "all50":
        n = len(pool) // 2
    else:
        raise ValueError("For regiontype='all', use subregion all33/all50")
    selected_cells = np.sort(select_rng.choice(pool, size=n, replace=False))
else:
    raise ValueError("Unknown regiontype")

# assign selected_cells
if regiontype == "amp":
    if subregion == "a": selected_cells = a_cells
    elif subregion == "m": selected_cells = m_cells
    elif subregion == "p": selected_cells = p_cells
    else: raise ValueError("Invalid subregion")
elif regiontype == "ap":
    if subregion == "a": selected_cells = a_cells
    elif subregion == "p": selected_cells = p_cells
    else: raise ValueError("Invalid subregion")
elif regiontype == "dmv":
    if subregion == "d": selected_cells = d_cells
    elif subregion == "m": selected_cells = m_cells
    elif subregion == "v": selected_cells = v_cells
    else: raise ValueError("Invalid subregion")
elif regiontype == "dv":
    if subregion == "d": selected_cells = d_cells
    elif subregion == "v": selected_cells = v_cells
    else: raise ValueError("Invalid subregion")

# filter dataset rows by selected cells
sel_mask = np.isin(alltrain_tgv[:,0].astype(int), selected_cells)
Xtrain = torch.tensor(alltrain_tgv[sel_mask], dtype=torch.float64)
Mask   = torch.tensor(mask_df[sel_mask], dtype=torch.bool)

# enforce nonnegativity
Xtrain[:, 2:2+Ngenes] = Xtrain[:, 2:2+Ngenes].abs()
Xtrain[:, 2+Ngenes+3 : 2+Ngenes+3+Ngenes] = Xtrain[:, 2+Ngenes+3 : 2+Ngenes+3+Ngenes].abs()

# train/val split
n_total = len(Xtrain)
n_train = int(0.9 * n_total)
train_X, val_X = Xtrain[:n_train], Xtrain[n_train:]
train_mask, val_mask = Mask[:n_train], Mask[n_train:]

class DrosoDataset(Dataset):
    def __init__(self, X, mask): self.X, self.mask = X, mask
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.mask[i]

num_workers = min(16, os.cpu_count()-2) if torch.cuda.is_available() else 0
train_loader = DataLoader(DrosoDataset(train_X, train_mask), batch_size=1000, shuffle=True, num_workers=num_workers)
val_loader   = DataLoader(DrosoDataset(val_X,   val_mask),   batch_size=1000, shuffle=False, num_workers=num_workers)

# ---------------- Model ----------------
class Multiply(nn.Module):
    def __init__(self, c): super().__init__(); self.c = c
    def forward(self, x): return self.c * x

def make_activation(name):
    if name == "tanh": return nn.Tanh()
    if name == "gelu": return nn.GELU()
    if name == "relu": return nn.ReLU()
    return nn.Tanh()

class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.beta = beta
        self.max_lr = maxlr
        self.latent_dim = ldim
        act = make_activation(actfcn)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(setinputdim, N1), act,
            nn.Linear(N1, N1), act,
            nn.Linear(N1, N2), act,
            nn.Linear(N2, N3), act
        )
        self.mu_layer = nn.Linear(N3, ldim)
        self.log_var_layer = nn.Linear(N3, ldim)

        # Decoder
        dec = [
            nn.Linear(ldim, N3), act,
            nn.Linear(N3, N2), act,
            nn.Linear(N2, N1), act,
            nn.Linear(N1, N1), act,
            nn.Linear(N1, setoutputdim)
        ]
        if finalact == "sigmoid":
            dec += [nn.Sigmoid(), Multiply(3.18)]
        self.decoder = nn.Sequential(*dec)

        self.double()
        self.automatic_optimization = False

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = priorstd * torch.randn_like(std)
        return mu + eps * std

    def forward(self, x): 
        mu = self.mu_layer(self.encoder(x))
        log_var = self.log_var_layer(self.encoder(x))
        z = self.reparameterize(mu, log_var)
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        tgv, mask = batch
        g = tgv[:,2:101]; xyz = tgv[:,101:104]
        x = torch.cat([g, xyz], dim=1)

        mu = self.mu_layer(self.encoder(x))
        log_var = self.log_var_layer(self.encoder(x))
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)

        se = (decoded - g)**2
        recon_loss = (se * mask.float()).sum() / (mask.float().sum() + 1e-8)
        kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        loss = recon_loss + self.beta * kl_div

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tgv, mask = batch
        g = tgv[:,2:101]; xyz = tgv[:,101:104]
        x = torch.cat([g, xyz], dim=1)

        mu = self.mu_layer(self.encoder(x))
        log_var = self.log_var_layer(self.encoder(x))
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)

        se = (decoded - g)**2
        recon_loss = (se * mask.float()).sum() / (mask.float().sum() + 1e-8)
        kl_div = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        val_loss = recon_loss + self.beta * kl_div
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.max_lr, foreach=False)
        cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=self.max_lr*0.05)
        return [optimizer], [cosine]

    

# ---------------- Training ----------------

maxlr_str = f"{maxlr:.0e}".replace("e-0", "en")
seed_tag = f"_seed{seed_select}" if regiontype == "all" else ""
filename_template = (
    f"VAE_{regiontype}_{subregion}{seed_tag}_ldim{ldim}_{actfcn}_lr{maxlr_str}_maxep{max_epochs}_{job_id}"
    f"_best{{epoch:04d}}-{{val_loss:.5f}}"
)


callbacks = [
    EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5000, verbose=True, mode="min"),
    ModelCheckpoint(dirpath="checkpoints", filename=filename_template, monitor="val_loss", mode="min", save_top_k=1, verbose=True)
]

vae_model = VAE()
trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision="64-true",
    log_every_n_steps=10,
    callbacks=callbacks,
    logger=False
)
trainer.fit(vae_model, train_loader, val_loader)
