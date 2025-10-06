#!/usr/bin/env python3
import os, sys, re, math, glob
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torchdiffeq import odeint, odeint_adjoint

# ─────────────────────────── CLI ───────────────────────────
if len(sys.argv) < 8:
    print(
        "Usage:\n"
        "  python Step5_NeuralODE_99g_fromVAEimputed.py "
        "<maxlr> <bsize> <max_epochs> <ep_maxlr> <Ns> <Nstep> <job_id> "
        "[type=val|valid|train|smooth] [fillmethod=auto|zero|avg|rand] "
        "[use_adjoint=1] [save_latents_csv=0]\n"
        "\nNotes:\n"
        " - VAE ckpt and imputed CSV are auto-discovered from job_id (+type/fillmethod).\n"
        " - Data is the 104-col imputed file; pairs built by (cid, time)->(cid, time+1).\n"
    )
    sys.exit(1)

max_lr      = float(sys.argv[1])
bsize       = int(sys.argv[2])
max_epochs  = int(sys.argv[3])
ep_maxlr    = int(sys.argv[4])
Ns          = int(sys.argv[5])
Nstep       = int(sys.argv[6])
job_id      = str(sys.argv[7])

ctype_raw   = sys.argv[8].lower()  if len(sys.argv) >= 9  else "valid"
fillmethod  = sys.argv[9].lower()  if len(sys.argv) >= 10 else os.getenv("FILLMETHOD_DEFAULT", "auto")
use_adjoint = int(sys.argv[10])    if len(sys.argv) >= 11 else 1
save_lat_csv= int(sys.argv[11])    if len(sys.argv) >= 12 else 0
# Optional path to test-cells list (one ID per line or a CSV column)
test_cells_csv = sys.argv[12] if len(sys.argv) >= 13 and sys.argv[12].lower() != "none" else os.getenv("TEST_CELLS_CSV", "")

# normalize common aliases
CTYPE_ALIASES = {"val": "valid", "valid": "valid", "train": "train", "smooth": "smooth"}
ctype = CTYPE_ALIASES.get(ctype_raw, ctype_raw)

# allow directory overrides via env
IMPUTED_DIR = os.getenv("IMPUTED_DIR", "ImputedOut_clean")
CKPT_DIR    = os.getenv("CKPT_DIR", "checkpoints")


# tolerances via env (or defaults)
rtol = float(os.getenv("ODE_RTOL", "1e-12"))
atol = float(os.getenv("ODE_ATOL", "1e-12"))
endpoints_only = int(os.getenv("ODE_ENDPOINTS_ONLY", "1"))
max_num_steps_env = os.getenv("ODE_MAX_STEPS", "").strip()
max_num_steps = int(max_num_steps_env) if max_num_steps_env.isdigit() else None

# ─────────────────────── Globals & layout ───────────────────────
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float64
SEED   = 42
np.random.seed(SEED); torch.manual_seed(SEED)

# 104-col layout (cid, time, 99 genes, 3 xyz)
N_GENES  = 99
COL_CID  = 0
COL_TIME = 1
COL_G0   = 2
COL_GN   = COL_G0 + N_GENES
COL_XYZ0 = COL_GN
COL_XYZN = COL_XYZ0 + 3

CLAMP_LOGVAR_MIN = -8.0
CLAMP_LOGVAR_MAX =  8.0
def load_test_cells(csv_path: str) -> set:
    """
    Reads a list of cell IDs to exclude. Accepts:
      - plain text with one ID per line
      - CSV with one column (with or without header)
    Returns a set of ints. If path empty/None, returns empty set.
    """
    if not csv_path:
        return set()
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"test_cells file not found: {csv_path}")
    try:
        df = pd.read_csv(csv_path, header=None)
        # If a header exists, try again letting pandas infer it
        if df.shape[1] > 1:
            df = pd.read_csv(csv_path)
            # pick the first column
            col0 = df.columns[0]
            ids = pd.to_numeric(df[col0], errors="coerce").dropna().astype(int).tolist()
        else:
            ids = pd.to_numeric(df.iloc[:,0], errors="coerce").dropna().astype(int).tolist()
        return set(int(x) for x in ids)
    except Exception as e:
        # last resort: line-by-line
        with open(csv_path, "r") as f:
            ids = []
            for line in f:
                line = line.strip().split(",")[0]
                if not line:
                    continue
                try:
                    ids.append(int(float(line)))
                except:
                    pass
        return set(ids)

# ─────────────────────── utils: discover files ───────────────────────
def find_imputed(job: str, ctype: str, fm: str, imputed_dir: str = None) -> Tuple[str, str]:
    """
    Returns (imputed_csv_path, fillmethod_used).

    Supports both:
      - NEW scheme (no fillmethod token):  Imputed_<job>_<type>_ldimK.csv
      - OLD scheme (with fillmethod):      Imputed_<fm>_<job>_<type>*.csv

    If fm='auto' and the filename doesn't encode the fillmethod, we default to 'rand'.
    Also accepts type aliases ('valid'/'val', 'smooth'/case variants).
    """
    imputed_dir = imputed_dir or IMPUTED_DIR
    os.makedirs(imputed_dir, exist_ok=True)

    # type alternatives commonly seen in filenames
    if ctype == "valid":
        ctype_alts = ["valid", "val"]
    elif ctype == "smooth":
        ctype_alts = ["smooth", "SMOOTH"]
    elif ctype == "train":
        ctype_alts = ["train", "TRAIN"]
    else:
        ctype_alts = [ctype]

    patterns = []

    # 1) NEW scheme (no fillmethod token in filename)
    for ct in ctype_alts:
        patterns += [
            os.path.join(imputed_dir, f"Imputed_{job}_{ct}_ldim*.csv"),
            os.path.join(imputed_dir, f"Imputed_{job}_{ct}*.csv"),
            os.path.join(imputed_dir, f"*Imputed*{job}*{ct}*ldim*.csv"),
        ]

    # 2) OLD scheme (with fillmethod token)
    fms = ["rand", "avg", "zero"] if fm == "auto" else [fm]
    for f in fms:
        for ct in ctype_alts:
            patterns += [
                os.path.join(imputed_dir, f"Imputed_{f}_{job}_{ct}*.csv"),
                os.path.join(imputed_dir, f"*Imputed*{f}*{job}*{ct}*.csv"),
            ]

    hits: List[str] = []
    for patt in patterns:
        hits.extend(glob.glob(patt))

    if not hits:
        print(f"[find_imputed] searched patterns (none hit):")
        for p in patterns:
            print("   ", p)
        try:
            print(f"[find_imputed] listing {imputed_dir}:")
            for name in sorted(os.listdir(imputed_dir))[:200]:
                print("   ", name)
        except Exception as e:
            print(f"[find_imputed] cannot list {imputed_dir}: {e}")
        raise FileNotFoundError(f"No Imputed file found for job {job} type {ctype} under {imputed_dir}")

    # newest by mtime wins
    hits.sort(key=os.path.getmtime, reverse=True)
    best = hits[0]
    base = os.path.basename(best)

    # infer fillmethod if present; otherwise honor CLI or default to 'rand'
    if "_rand_" in base:
        fm_used = "rand"
    elif "_avg_" in base:
        fm_used = "avg"
    elif "_zero_" in base:
        fm_used = "zero"
    else:
        fm_used = fm if fm != "auto" else "rand"

    return best, fm_used



def find_ckpt(job: str, ctype: str, ckpt_dir: str = None) -> str:
    ckpt_dir = ckpt_dir or CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)

    pats: List[str] = []
    if ctype == "train":
        pats += [f"*{job}*TRAIN*.ckpt", f"*{job}*train*.ckpt"]
    elif ctype == "smooth":
        pats += [f"*{job}*SMOOTH*.ckpt", f"*{job}*smooth*.ckpt"]
    else:  # valid/best
        pats += [f"*{job}*best*.ckpt", f"*{job}*valid*.ckpt", f"*{job}*val*.ckpt"]

    hits: List[str] = []
    for p in pats:
        hits.extend(glob.glob(os.path.join(ckpt_dir, p)))

    # last resort: anything with job id
    if not hits:
        hits.extend(glob.glob(os.path.join(ckpt_dir, f"*{job}*.ckpt")))

    if not hits:
        print("[find_ckpt] tried patterns:")
        for p in pats:
            print("   ", os.path.join(ckpt_dir, p))
        try:
            print(f"[find_ckpt] listing {ckpt_dir}:")
            for name in sorted(os.listdir(ckpt_dir))[:200]:
                print("   ", name)
        except Exception as e:
            print(f"[find_ckpt] cannot list {ckpt_dir}: {e}")
        raise FileNotFoundError(f"No ckpt found for job {job} type {ctype} in {ckpt_dir}")

    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]


# ─────────────────────── layers / encoder ───────────────────────
class CenteredSoftplus(nn.Module):
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.sp = nn.Softplus(beta=beta, threshold=threshold)
    def forward(self, x):
        return self.sp(x) - math.log(2.0)

def make_activation(name: str):
    name = (name or "tanh").lower()
    return {
        "relu": nn.ReLU(), "tanh": nn.Tanh(), "gelu": nn.GELU(), "silu": nn.SiLU(),
        "mish": nn.Mish(), "elu": nn.ELU(), "celu": nn.CELU(),
        "softplus": nn.Softplus(), "softplus0": CenteredSoftplus(),
        "leakyrelu": nn.LeakyReLU(negative_slope=0.1),
    }.get(name, nn.Tanh())

class VAEInfer(nn.Module):
    def __init__(self,
                 ndense1: int, ndense2: int, ndense3: int,
                 latent_dim: int,
                 input_dim: int,    # genes + xyz
                 act_name: str = "tanh",
                 enc_dropout_p: float = 0.0):
        super().__init__()
        act = make_activation(act_name)
        self.enc_drop = nn.Dropout(p=enc_dropout_p) if enc_dropout_p and enc_dropout_p > 0 else nn.Identity()
        self.enc_fc1 = nn.Linear(input_dim, ndense1, dtype=dtype)
        self.enc_fc2 = nn.Linear(ndense1, ndense1, dtype=dtype)
        self.enc_fc3 = nn.Linear(ndense1, ndense2, dtype=dtype)
        self.enc_fc4 = nn.Linear(ndense2, ndense3, dtype=dtype)
        self.act = act
        self.mu_layer      = nn.Linear(ndense3, latent_dim, dtype=dtype)
        self.log_var_layer = nn.Linear(ndense3, latent_dim, dtype=dtype)
        self.double(); self.eval()

    @torch.no_grad()
    def encode_mu_logvar(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.enc_drop(self.act(self.enc_fc1(x)))
        x = self.enc_drop(self.act(self.enc_fc2(x)))
        x = self.enc_drop(self.act(self.enc_fc3(x)))
        x = self.enc_drop(self.act(self.enc_fc4(x)))
        mu = self.mu_layer(x)
        lv = self.log_var_layer(x).clamp(min=CLAMP_LOGVAR_MIN, max=CLAMP_LOGVAR_MAX)
        return mu, lv

@torch.no_grad()
def load_encoder_from_ckpt(ckpt_path: str, n_genes_expected=99) -> Tuple[VAEInfer, int, float, int]:
    """Returns (encoder_model, latent_dim, priorstd, input_dim_for_encoder)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    h = ckpt.get("hyper_parameters", {})
    ld   = int(h.get("latent_dim", h.get("ldim", 6)))
    n1   = int(h.get("ndense1", 1024))
    n2   = int(h.get("ndense2", 512))
    n3   = int(h.get("ndense3", 256))
    act  = str(h.get("act_name", h.get("actfcn", "tanh")))
    enc_do = float(h.get("enc_dropout_p", 0.0))
    inp_dim = int(h.get("input_dim", n_genes_expected + 3))
    priorstd = float(h.get("priorstd", 0.10))
    model = VAEInfer(n1, n2, n3, ld, inp_dim, act_name=act, enc_dropout_p=enc_do).to(device).eval()
    sd = ckpt.get("state_dict", ckpt)
    enc_keys = [k for k in sd.keys() if k.startswith(("enc_fc", "mu_layer", "log_var_layer", "enc_drop"))]
    sub_sd = {k: sd[k] for k in enc_keys}
    missing, unexpected = model.load_state_dict(sub_sd, strict=False)
    if missing or unexpected:
        print(f"[load_encoder] missing={len(missing)} unexpected={len(unexpected)}")
    return model, ld, priorstd, inp_dim

# ─────────────────────── pairs from IMPUTED 104-col ───────────────────────
def build_mu_pairs_from_imputed(
    imputed_csv: str,
    vae_encoder: VAEInfer,
    latent_dim: int,
    priorstd: float,
    enc_input_dim: int,
    batch: int = 4096,
    exclude_cids: Optional[set] = None,   # <── NEW
) -> np.ndarray:
    """
    Build pairs by (cid,time) → (cid,time+1) from a 104-col imputed CSV.
    Returns ndarray [M, 3*ldim]: [mu_t | mu_{t+1} | sigma_{t+1}]
    """
    df = pd.read_csv(imputed_csv, header=None)
    assert df.shape[1] == 104, f"Expected 104 columns, got {df.shape[1]}"
    df.columns = [*range(104)]

    # ensure integer cids
    df[COL_CID] = pd.to_numeric(df[COL_CID], errors="coerce").astype("Int64")

    # EXCLUDE test cells (applies to both t and t+1 by filtering before pairing)
    if exclude_cids:
        n_before = len(df)
        df = df[~df[COL_CID].isin(list(exclude_cids))].copy()
        print(f"[filter] excluded test cells: {n_before - len(df)} rows removed (remaining {len(df)})")

    # keep times 0..5; we need t and t+1 → so t≤4
    df = df[(df[COL_TIME] >= 0) & (df[COL_TIME] <= 5)].copy()
    df_idx = df[[COL_CID, COL_TIME]].copy()
    df_idx["idx"] = np.arange(len(df), dtype=np.int64)

    # map (cid,time) pairs: take rows at t and t+1
    df_tp1 = df_idx.copy(); df_tp1[COL_TIME] = df_tp1[COL_TIME] - 1  # shift so t+1 joins t
    merged = df_tp1.merge(df_idx, on=[COL_CID, COL_TIME], suffixes=("_tp1","_t"), how="inner")
    if merged.empty:
        raise RuntimeError("No consecutive (cid,time→time+1) pairs found.")
    idx_t  = merged["idx_t"  ].to_numpy()
    idx_tp = merged["idx_tp1"].to_numpy()

    # slice features [genes|xyz] at t and t+1
    X_t  = np.hstack([df.iloc[idx_t,  COL_G0:COL_GN].values,
                      df.iloc[idx_t,  COL_XYZ0:COL_XYZN].values]).astype(np.float64, copy=False)
    X_tp = np.hstack([df.iloc[idx_tp, COL_G0:COL_GN].values,
                      df.iloc[idx_tp, COL_XYZ0:COL_XYZN].values]).astype(np.float64, copy=False)

    # if encoder expects a different input dim (safety), slice last dims
    if X_t.shape[1] != enc_input_dim:
        if X_t.shape[1] > enc_input_dim:
            X_t  = X_t[:, :enc_input_dim]
            X_tp = X_tp[:, :enc_input_dim]
            print(f"[info] Sliced features to match encoder input_dim={enc_input_dim}")
        else:
            raise RuntimeError(f"Encoder expects {enc_input_dim} dims but data has {X_t.shape[1]}")

    # encode in batches
    mu_t_list, mu_tp_list, lv_tp_list = [], [], []
    with torch.no_grad():
        for i in range(0, len(X_t), batch):
            xb  = torch.tensor(X_t [i:i+batch], dtype=dtype, device=device)
            xbp = torch.tensor(X_tp[i:i+batch], dtype=dtype, device=device)
            mu_t,  _  = vae_encoder.encode_mu_logvar(xb)
            mu_tp, lv = vae_encoder.encode_mu_logvar(xbp)
            mu_t_list.append(mu_t.cpu())
            mu_tp_list.append(mu_tp.cpu())
            lv_tp_list.append(lv.cpu())

    mu_t_all  = torch.cat(mu_t_list,  dim=0).numpy()
    mu_tp_all = torch.cat(mu_tp_list, dim=0).numpy()
    lv_tp_all = torch.cat(lv_tp_list, dim=0).numpy()
    sigma_tp_all = priorstd * np.exp(0.5 * lv_tp_all)

    pairs = np.concatenate([mu_t_all, mu_tp_all, sigma_tp_all], axis=1)
    return pairs

# ─────────────────────── data / model for ODE ───────────────────────
class PairDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=dtype)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i]

class NeuralODE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, activation):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, dtype=dtype), activation,
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype), activation,
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype), activation,
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype), activation,
            nn.Linear(hidden_dim, latent_dim, dtype=dtype),
        )
    def forward(self, t, z):
        return self.v_net(z)

class SuccessorTrainer(pl.LightningModule):
    def __init__(self, max_lr, hidden, Nstep, latent_dim, use_adjoint=True,
                 rtol: float = 1e-12, atol: float = 1e-12, endpoints_only: bool = True,
                 max_num_steps: Optional[int] = None):
        super().__init__()
        self.max_lr = max_lr
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.Nstep = Nstep
        self.use_adjoint = bool(use_adjoint)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.endpoints_only = bool(endpoints_only)
        self.max_num_steps = max_num_steps
        self.successor = NeuralODE(latent_dim, hidden, nn.Tanh())
        self.tspan = torch.tensor([0.0, 1.0], device=device, dtype=dtype) if self.endpoints_only \
                     else torch.linspace(0., 1., Nstep, device=device, dtype=dtype)
        self.double()
        self.save_hyperparameters()

    def _odeint(self, func, z0, tspan):
        solver = odeint_adjoint if self.use_adjoint else odeint
        options = {}
        if self.max_num_steps is not None:
            options["max_num_steps"] = int(self.max_num_steps)
        return solver(
            func, z0, tspan, method="dopri5",
            rtol=self.rtol, atol=self.atol,
            options=options if options else None
        )

    def training_step(self, batch, batch_idx):
        ld = self.latent_dim
        mu_now    = batch[:, :ld]
        mu_target = batch[:, ld:2*ld]
        std_target= batch[:, 2*ld:3*ld]
        traj     = self._odeint(self.successor, mu_now, self.tspan)
        z_future = traj[-1]
        rel_err  = (z_future - mu_target)**2 / (std_target**2 + 1e-8)
        loss = rel_err.mean()
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        ld = self.latent_dim
        mu_now    = batch[:, :ld]
        mu_target = batch[:, ld:2*ld]
        std_target= batch[:, 2*ld:3*ld]
        traj     = self._odeint(self.successor, mu_now, self.tspan)
        z_future = traj[-1]
        rel_err  = (z_future - mu_target)**2 / (std_target**2 + 1e-8)
        val_loss = rel_err.mean()
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        opt = optim.Adam(self.successor.parameters(), lr=self.max_lr, foreach=False)
        def lr_lambda(epoch):
            steep1 = 10 / ep_maxlr
            steep2 = 10 / max_epochs
            if epoch <= ep_maxlr:
                c1 = 1 + np.exp(-steep1 * (ep_maxlr / 2))
                return c1 / (1 + np.exp(-steep1 * (epoch - ep_maxlr / 2)))
            else:
                c2 = 1 + np.exp(-steep2 * (max_epochs * 3 / 4 - ep_maxlr))
                return c2 / (1 + np.exp(-steep2 * (max_epochs * 3 / 4 - epoch)))
        sch = lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return [opt], [sch]

# ─────────────────────────── main ───────────────────────────
def main():
    print(f"DEVICE={device} | RTOL={rtol} ATOL={atol} | endpoints_only={bool(endpoints_only)} | max_steps={max_num_steps}")
    print(f"[info] job_id={job_id}  type={ctype}  fillmethod={fillmethod}")

    # discover files
    imputed_csv, fm_used = find_imputed(job_id, ctype, fillmethod, imputed_dir=IMPUTED_DIR)
    ckpt_path = find_ckpt(job_id, ctype, ckpt_dir=CKPT_DIR)

    print(f"[info] imputed={os.path.basename(imputed_csv)}  (fm={fm_used})")
    print(f"[info] ckpt   ={os.path.basename(ckpt_path)}")

    # encoder
    # encoder (unchanged)
    encoder, ldim, priorstd, enc_in = load_encoder_from_ckpt(ckpt_path, n_genes_expected=99)

    # load test cell IDs (optional)
    exclude_cids = load_test_cells(test_cells_csv)
    if exclude_cids:
        print(f"[info] loaded {len(exclude_cids)} test cell IDs to exclude")

    # build pairs from imputed (cid,time)->(cid,time+1)
    pairs = build_mu_pairs_from_imputed(
        imputed_csv=imputed_csv,
        vae_encoder=encoder,
        latent_dim=ldim,
        priorstd=priorstd,
        enc_input_dim=enc_in,
        batch=4096,
        exclude_cids=exclude_cids,   # <── NEW
    )
    print(f"[pairs] built {pairs.shape[0]} pairs (μ_t→μ_{'{'}t+1{'}'}) with ldim={ldim}")

    # shuffle
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(pairs.shape[0])
    pairs = pairs[idx]

    # split 90/10
    N = pairs.shape[0]
    split_at = int(0.9 * N)
    train_np = pairs[:split_at]; val_np = pairs[split_at:]
    print(f"[split] train={len(train_np)}  val={len(val_np)}")

    train_ds = PairDataset(train_np)
    val_ds   = PairDataset(val_np)
    num_workers = 0 if device.type == "cpu" else min(16, os.cpu_count() or 4)

    train_loader = DataLoader(train_ds, batch_size=bsize, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=bsize, shuffle=False, num_workers=num_workers)

    # model
    model = SuccessorTrainer(
        max_lr, Ns, Nstep, latent_dim=ldim, use_adjoint=use_adjoint,
        rtol=rtol, atol=atol, endpoints_only=bool(endpoints_only),
        max_num_steps=max_num_steps
    ).to(device)

    # checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    tag = f"{job_id}_{ctype}_{fm_used}"
    ck_name = f"neuODE99_fromVAE_{tag}_ld{ldim}_lr{max_lr:.0e}_maxep{max_epochs}_best" + "{epoch:04d}-{val_loss:.5f}"

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=5000, verbose=True, mode="min"),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename=ck_name,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=False,
    )
    trainer.fit(model, train_loader, val_loader)

    # optional: save pairs for reuse
    if save_lat_csv:
        os.makedirs("encoded_latents", exist_ok=True)
        out_lat = f"encoded_latents/EncodedMuStd_{tag}_ld{ldim}.csv"
        pd.DataFrame(pairs).to_csv(out_lat, index=False)
        print(f"[info] saved latent pairs CSV: {out_lat}")

if __name__ == "__main__":
    main()
