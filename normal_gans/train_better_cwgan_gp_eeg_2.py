#!/usr/bin/env python3
"""
train_better_cwgan_gp_eeg.py

Improved conditional WGAN-GP for BCICIV-2b EEG (T.gdf), with:
- Residual + multi-scale Conv1D generator
- Stronger conditioning via FiLM (scale+shift)
- Projection critic (label usage enforced)
- PSD auxiliary loss (G only) + tiny smoothness regularizer
- TTUR (lr_G > lr_D)
- WGAN-GP + small drift penalty for critic stability
- Constant n_critic (NOT scheduled)

Outputs (in --outdir):
- checkpoints/*.pt (periodic + last)
- loss_curves.png
- samples_left.png, samples_right.png
- metrics.json

Example:
  python train_better_cwgan_gp_eeg.py --data-dir BCICIV_2b_gdf --epochs 500 --outdir runs/vanilla_improved_2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader

import mne
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -------------------------
# Data preprocessing
# -------------------------
def preprocess_one_file(
    gdf_path: Path,
    event_keys: Dict[str, str],
    tmin: float,
    tmax: float,
    l_freq: float,
    h_freq: float,
    notch_hz: float,
    eog_reject_uv: float,
    resample_hz: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, C, T) EEG only
      y: (N,) 0=left, 1=right
    """
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR" if not verbose else None)

    for ch in raw.ch_names:
        if "EOG" in ch.upper():
            raw.set_channel_types({ch: "eog"})

    raw.filter(l_freq, h_freq, verbose="ERROR" if not verbose else None)
    raw.notch_filter(notch_hz, verbose="ERROR" if not verbose else None)

    events, event_id_all = mne.events_from_annotations(raw, verbose="ERROR" if not verbose else None)

    event_id = {
        "left": event_id_all[event_keys["left"]],
        "right": event_id_all[event_keys["right"]],
    }

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        picks=mne.pick_types(raw.info, eeg=True, eog=True),
        baseline=None,
        preload=True,
        verbose="ERROR" if not verbose else None,
    )

    # EOG rejection (peak-to-peak) in µV
    eog_picks = mne.pick_types(epochs.info, eog=True)
    X = epochs.get_data()  # volts
    X_eog = X[:, eog_picks, :]
    ptp_uv = (X_eog.max(-1) - X_eog.min(-1)) * 1e6
    bad_idx = np.where(ptp_uv.max(axis=1) > eog_reject_uv)[0]
    if len(bad_idx) > 0:
        epochs.drop(bad_idx)

    epochs = epochs.pick_types(eeg=True)
    epochs.resample(resample_hz, verbose="ERROR" if not verbose else None)

    y_ev = epochs.events[:, 2]
    y = np.array([0 if v == event_id["left"] else 1 for v in y_ev], dtype=np.int64)

    return epochs.get_data().astype(np.float32), y


def normalize_to_unit(X_volts: np.ndarray, clip_uv: float) -> np.ndarray:
    """
    volts -> µV, clip to [-clip_uv, clip_uv], scale to [-1,1]
    """
    X_uv = X_volts * 1e6
    X_uv = np.clip(X_uv, -clip_uv, clip_uv)
    return (X_uv / clip_uv).astype(np.float32)


# -------------------------
# Dataset
# -------------------------
class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def stratified_split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
):
    """
    Stratified split into train/val/test.
    Fractions are based on full dataset.
    """
    assert 0 < val_frac < 1
    assert 0 < test_frac < 1
    assert (val_frac + test_frac) < 1

    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0 = len(idx0)
    n1 = len(idx1)

    n0_val = int(round(n0 * val_frac))
    n1_val = int(round(n1 * val_frac))

    n0_test = int(round(n0 * test_frac))
    n1_test = int(round(n1 * test_frac))

    val0 = idx0[:n0_val]
    val1 = idx1[:n1_val]

    test0 = idx0[n0_val:n0_val + n0_test]
    test1 = idx1[n1_val:n1_val + n1_test]

    train0 = idx0[n0_val + n0_test:]
    train1 = idx1[n1_val + n1_test:]

    tr_idx = np.concatenate([train0, train1])
    val_idx = np.concatenate([val0, val1])
    te_idx = np.concatenate([test0, test1])

    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)
    rng.shuffle(te_idx)

    return (
        X[tr_idx], y[tr_idx],
        X[val_idx], y[val_idx],
        X[te_idx], y[te_idx],
    )

# -------------------------
# Generator blocks (FiLM conditioning)
# -------------------------
class MultiScaleConv1d(nn.Module):
    def __init__(self, channels: int, kernels=(3, 7, 15, 31)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(channels, channels, k, padding=k // 2) for k in kernels])

    def forward(self, x):
        out = 0.0
        for conv in self.convs:
            out = out + conv(x)
        return out / len(self.convs)


class ResBlock1d(nn.Module):
    """
    FiLM: apply per-channel (scale, shift) from condition:
      h = (1 + gamma) * h + beta
    Stronger than bias-only, helps LEFT vs RIGHT separation.
    """
    def __init__(self, channels: int, cond_dim: int, use_multiscale: bool = True):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.ms = MultiScaleConv1d(channels) if use_multiscale else nn.Identity()
        self.cond_to_affine = nn.Linear(cond_dim, 2 * channels)  # gamma,beta
        self.act = nn.LeakyReLU(0.2)

    def film(self, h, cond):
        gb = self.cond_to_affine(cond)              # (B, 2C)
        gamma, beta = gb.chunk(2, dim=1)            # (B,C), (B,C)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return (1.0 + gamma) * h + beta

    def forward(self, x, cond):
        h = self.conv1(x)
        h = self.film(h, cond)
        h = self.act(h)

        h = self.ms(h)
        h = self.act(h)

        h = self.conv2(h)
        h = self.film(h, cond)

        return self.act(x + 0.5 * h)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2)
        self.res = ResBlock1d(out_ch, cond_dim, use_multiscale=True)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, cond):
        x = self.up(x)
        x = self.act(self.conv(x))
        x = self.res(x, cond)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim: int, n_classes: int, C: int, T: int,
                 base_ch: int = 256, cond_dim: int = 96):
        super().__init__()
        self.C, self.T = C, T
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        self.y_embed = nn.Embedding(n_classes, cond_dim)

        self.L0 = int(math.ceil(T / 16))
        self.fc = nn.Sequential(
            nn.Linear(z_dim + cond_dim, base_ch * self.L0),
            nn.LeakyReLU(0.2),
        )

        self.up1 = UpBlock(base_ch, base_ch // 2, cond_dim)
        self.up2 = UpBlock(base_ch // 2, base_ch // 4, cond_dim)
        self.up3 = UpBlock(base_ch // 4, base_ch // 8, cond_dim)
        self.up4 = UpBlock(base_ch // 8, base_ch // 16, cond_dim)

        self.to_out = nn.Sequential(
            nn.Conv1d(base_ch // 16, C, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = self.y_embed(y)
        x = torch.cat([z, cond], dim=1)
        x = self.fc(x).view(z.size(0), -1, self.L0)

        x = self.up1(x, cond)
        x = self.up2(x, cond)
        x = self.up3(x, cond)
        x = self.up4(x, cond)

        x = self.to_out(x)

        if x.size(-1) > self.T:
            x = x[..., :self.T]
        elif x.size(-1) < self.T:
            x = F.pad(x, (0, self.T - x.size(-1)))
        return x


# -------------------------
# Critic (Projection Discriminator)
# -------------------------
class Critic(nn.Module):
    def __init__(self, n_classes: int, C: int, T: int,
                 base_ch: int = 64, feat_dim: int = 256,
                 use_spectral_norm: bool = True):
        super().__init__()

        def SN(layer):
            return spectral_norm(layer) if use_spectral_norm else layer

        self.conv = nn.Sequential(
            SN(nn.Conv1d(C, base_ch, 7, padding=3)),
            nn.LeakyReLU(0.2),

            SN(nn.Conv1d(base_ch, base_ch * 2, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),

            SN(nn.Conv1d(base_ch * 2, base_ch * 4, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),

            SN(nn.Conv1d(base_ch * 4, base_ch * 4, 5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_feat = nn.Sequential(
            SN(nn.Linear(base_ch * 4, feat_dim)),
            nn.LeakyReLU(0.2),
        )
        self.fc_out = SN(nn.Linear(feat_dim, 1))
        self.y_embed = nn.Embedding(n_classes, feat_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.pool(h).squeeze(-1)
        h = self.fc_feat(h)
        out = self.fc_out(h).squeeze(-1)
        ey = self.y_embed(y)
        proj = (h * ey).sum(dim=1)
        return out + proj

# -------------------------
# WGAN-GP bits
# -------------------------
def gradient_penalty(D: nn.Module, real: torch.Tensor, fake: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=real.device)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)

    d_interp = D(interp, y)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(B, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def batch_log_psd(x: torch.Tensor) -> torch.Tensor:
    Xf = torch.fft.rfft(x, dim=-1)
    psd = (Xf.real ** 2 + Xf.imag ** 2)
    return torch.log(psd + 1e-8).mean(dim=0)


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def log_psd_mse_torch(X_real: np.ndarray, X_fake: np.ndarray, device: str, max_samples: int = 256) -> float:
    def subsample(X):
        if len(X) <= max_samples:
            return X
        idx = np.random.choice(len(X), max_samples, replace=False)
        return X[idx]

    Xr = subsample(X_real)
    Xf = subsample(X_fake)

    xr = torch.tensor(Xr, dtype=torch.float32, device=device)
    xf = torch.tensor(Xf, dtype=torch.float32, device=device)

    pr = batch_log_psd(xr)
    pf = batch_log_psd(xf)
    return float(((pf - pr) ** 2).mean().item())


def global_percentile_mae(X_real: np.ndarray, X_fake: np.ndarray, percentiles=(5, 50, 95), max_points=2_000_000) -> float:
    def flatten_cap(X):
        C = X.shape[1]
        flat = X.transpose(1, 0, 2).reshape(C, -1)
        if flat.shape[1] > max_points:
            idx = np.random.choice(flat.shape[1], max_points, replace=False)
            flat = flat[:, idx]
        return flat

    r = flatten_cap(X_real)
    f = flatten_cap(X_fake)

    errs = []
    for p in percentiles:
        pr = np.percentile(r, p, axis=1)
        pf = np.percentile(f, p, axis=1)
        errs.append(np.mean(np.abs(pf - pr)))
    return float(np.mean(errs))


# -------------------------
# EMA helper
# -------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(d).add_(p.data, alpha=(1.0 - d))

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, sd):
        self.ema_model.load_state_dict(sd)


# -------------------------
# Plot helpers
# -------------------------
def plot_loss_curves(D_hist, G_hist, out_png: Path):
    epochs = np.arange(1, len(G_hist) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, G_hist, label="Generator Loss")
    plt.plot(epochs, D_hist, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_stacked_pair(real_uv: np.ndarray, fake_uv: np.ndarray, tmin: float, tmax: float,
                      title: str, out_png: Path, ch_names: Optional[list] = None, sep_uv: Optional[float] = None):
    C, Tn = real_uv.shape
    t = np.linspace(tmin, tmax, Tn)

    if ch_names is None:
        ch_names = [f"ch{i}" for i in range(C)]

    def _stack(ax, epoch_uv, subtitle):
        nonlocal sep_uv
        if sep_uv is None:
            robust_amp = np.percentile(np.abs(epoch_uv), 95)
            sep = max(robust_amp * 2.5, 5.0)
        else:
            sep = sep_uv
        offsets = np.arange(C)[::-1] * sep
        for i in range(C):
            ax.plot(t, epoch_uv[i] + offsets[i], linewidth=1.0)
            ax.text(t[0] - 0.02 * (tmax - tmin), offsets[i], ch_names[i],
                    va="center", ha="right", fontsize=8)
        ax.set_title(subtitle)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("µV (stacked)")
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)
        ax.set_xlim(tmin, tmax)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    _stack(axes[0], real_uv, "REAL (one epoch)")
    _stack(axes[1], fake_uv, "FAKE (one epoch)")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# -------------------------
# Training
# -------------------------
@dataclass
class TrainConfig:
    data_dir: Path
    outdir: Path

    event_left: str = "769"
    event_right: str = "770"
    tmin: float = 0.0
    tmax: float = 4.0
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_hz: float = 50.0
    eog_reject_uv: float = 150.0
    resample_hz: int = 250

    # Changed: clip_uv a bit higher to reduce clipping artifacts
    clip_uv: float = 150.0

    batch_size: int = 32
    epochs: int = 500
    z_dim: int = 128

    # IMPORTANT: stays constant (your requirement)
    n_critic: int = 5

    lr_d: float = 1e-4
    lr_g: float = 2e-4

    lambda_gp: float = 5.0
    lambda_psd: float = 0.10

    lambda_amp: float = 0.15   # try 0.05–0.20

    # Changed: smaller smoothness to avoid over-smoothing EEG texture
    lambda_smooth: float = 1e-5

    # Added: drift penalty for critic stability
    eps_drift: float = 1e-3

    # Added: EMA for better samples/eval
    ema_decay: float = 0.999

    grad_clip_g: Optional[float] = 10.0

    seed: int = 42
    num_workers: int = 0
    save_every: int = 25
    eval_num_per_class: int = 256
    verbose_mne: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_spectral_norm: bool = True


def save_checkpoint(path: Path, epoch: int, G: nn.Module, D: nn.Module,
                    opt_G: torch.optim.Optimizer, opt_D: torch.optim.Optimizer,
                    ema: EMA,
                    cfg: TrainConfig):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_state": G.state_dict(),
            "critic_state": D.state_dict(),
            "opt_G_state": opt_G.state_dict(),
            "opt_D_state": opt_D.state_dict(),
            "ema_G_state": ema.state_dict(),
            "config": cfg.__dict__,
        },
        path,
    )


@torch.no_grad()
def generate_epochs(G: nn.Module, n: int, label01: int, z_dim: int, device: str) -> np.ndarray:
    y = torch.full((n,), label01, dtype=torch.long, device=device)
    z = torch.randn(n, z_dim, device=device)
    x = G(z, y)
    return x.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="BCICIV_2b_gdf", help="Folder containing *T.gdf files")

    # Changed default outdir HERE:
    ap.add_argument("--outdir", type=str, default="runs/vanilla_improved_2", help="Output directory")

    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--z-dim", type=int, default=128)

    # stays constant
    ap.add_argument("--n-critic", type=int, default=5, help="Keep constant (no scheduling).")

    ap.add_argument("--lr-d", type=float, default=1e-4)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lambda-gp", type=float, default=5.0)
    ap.add_argument("--lambda-psd", type=float, default=0.10)
    ap.add_argument("--lambda-smooth", type=float, default=1e-5)
    ap.add_argument("--eps-drift", type=float, default=1e-3)
    ap.add_argument("--ema-decay", type=float, default=0.999)

    ap.add_argument("--clip-uv", type=float, default=150.0)
    ap.add_argument("--resample-hz", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="", help="Optional checkpoint to resume from")
    ap.add_argument("--eval-num-per-class", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--use-spectral-norm", action="store_true", help="Enable spectral norm in critic (baseline).")
    ap.add_argument("--no-spectral-norm", action="store_true", help="Disable spectral norm in critic (recommended exp).")

    ap.add_argument("--lambda-amp", type=float, default=0.10)

    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=Path(args.data_dir),
        outdir=Path(args.outdir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        n_critic=args.n_critic,  # constant
        lr_d=args.lr_d,
        lr_g=args.lr_g,
        lambda_gp=args.lambda_gp,
        lambda_psd=args.lambda_psd,
        lambda_smooth=args.lambda_smooth,
        eps_drift=args.eps_drift,
        ema_decay=args.ema_decay,
        clip_uv=args.clip_uv,
        resample_hz=args.resample_hz,
        save_every=args.save_every,
        seed=args.seed,
        eval_num_per_class=args.eval_num_per_class,
        num_workers=args.num_workers,
        device=args.device,
        use_spectral_norm = (not args.no_spectral_norm),
        lambda_amp=args.lambda_amp,
    )

    seed_everything(cfg.seed)

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    (cfg.outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load + preprocess dataset
    # -------------------------
    t_files = sorted(cfg.data_dir.glob("*T.gdf"))
    if len(t_files) == 0:
        raise FileNotFoundError(f"No *T.gdf found in: {cfg.data_dir.resolve()}")

    event_keys = {"left": cfg.event_left, "right": cfg.event_right}

    X_list, y_list = [], []
    print(f"[DATA] Found {len(t_files)} T files in {cfg.data_dir}")
    for f in t_files:
        print(f"[DATA] Processing: {f.name}")
        Xi, yi = preprocess_one_file(
            f, event_keys,
            tmin=cfg.tmin, tmax=cfg.tmax,
            l_freq=cfg.l_freq, h_freq=cfg.h_freq,
            notch_hz=cfg.notch_hz,
            eog_reject_uv=cfg.eog_reject_uv,
            resample_hz=cfg.resample_hz,
            verbose=cfg.verbose_mne,
        )
        print(f"       epochs: {Xi.shape[0]}  shape: {Xi.shape}")
        X_list.append(Xi)
        y_list.append(yi)

    X_volts = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"[DATA] Combined: X={X_volts.shape}, y={y.shape}, counts={{0:{int((y==0).sum())}, 1:{int((y==1).sum())}}}")

    X = normalize_to_unit(X_volts, clip_uv=cfg.clip_uv)
    print(f"[DATA] Normalized: min={float(X.min()):.4f}, max={float(X.max()):.4f}, dtype={X.dtype}")

    X_tr, y_tr, X_val, y_val, X_te, y_te = stratified_split_train_val_test(
        X, y, val_frac=0.1, test_frac=0.1, seed=cfg.seed
    )
    print(f"[DATA] Split: train={X_tr.shape[0]}, val={X_val.shape[0]}, test={X_te.shape[0]}")

    train_loader = DataLoader(
        EEGDataset(X_tr, y_tr),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
    )

    # -------------------------
    # Build models
    # -------------------------
    C, T = X.shape[1], X.shape[2]
    G = Generator(cfg.z_dim, 2, C, T).to(cfg.device)
    D = Critic(2, C, T, use_spectral_norm=cfg.use_spectral_norm).to(cfg.device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))

    ema = EMA(G, decay=cfg.ema_decay)

    start_epoch = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        print(f"[RESUME] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        G.load_state_dict(ckpt["generator_state"])
        D.load_state_dict(ckpt["critic_state"])
        opt_G.load_state_dict(ckpt["opt_G_state"])
        opt_D.load_state_dict(ckpt["opt_D_state"])
        if "ema_G_state" in ckpt:
            ema.load_state_dict(ckpt["ema_G_state"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[RESUME] Resumed from epoch {start_epoch}")

    # -------------------------
    # Train
    # -------------------------
    D_hist, G_hist = [], []
    print(f"[TRAIN] device={cfg.device} epochs={cfg.epochs} batch_size={cfg.batch_size} n_critic={cfg.n_critic} (constant)")
    print(f"[TRAIN] lrD={cfg.lr_d} lrG={cfg.lr_g} lambda_gp={cfg.lambda_gp} eps_drift={cfg.eps_drift} lambda_psd={cfg.lambda_psd} lambda_smooth={cfg.lambda_smooth} ema_decay={cfg.ema_decay}")

    epoch_pbar = tqdm(range(start_epoch, cfg.epochs), desc="Epochs", dynamic_ncols=True)

    for epoch in epoch_pbar:
        G.train()
        D.train()
        g_losses = []
        wdist_vals = []

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False, dynamic_ncols=True)

        for x_real, y_real in batch_pbar:
            x_real = x_real.to(cfg.device, non_blocking=True)
            y_real = y_real.to(cfg.device, non_blocking=True)

            # ---- Critic steps (constant n_critic) ----
            x_fake_last = None
            for _ in range(cfg.n_critic):
                z = torch.randn(x_real.size(0), cfg.z_dim, device=cfg.device)
                x_fake = G(z, y_real).detach()
                x_fake_last = x_fake

                d_real = D(x_real, y_real).mean()
                d_fake = D(x_fake, y_real).mean()
                gp = gradient_penalty(D, x_real, x_fake, y_real)

                # WGAN-GP critic loss + drift penalty
                loss_D = (d_fake - d_real) + cfg.lambda_gp * gp + cfg.eps_drift * (d_real ** 2)

                opt_D.zero_grad(set_to_none=True)
                loss_D.backward()
                opt_D.step()

            # Wasserstein estimate (no GP)
            with torch.no_grad():
                d_real_eval = D(x_real, y_real).mean().item()
                d_fake_eval = D(x_fake_last, y_real).mean().item()
                wdist_vals.append(d_real_eval - d_fake_eval)

            # ---- Generator step ----
            z = torch.randn(x_real.size(0), cfg.z_dim, device=cfg.device)
            x_fake = G(z, y_real)

            loss_G_wgan = -D(x_fake, y_real).mean()

            # PSD auxiliary
            psd_real = batch_log_psd(x_real)
            psd_fake = batch_log_psd(x_fake)
            loss_psd = (psd_fake - psd_real).pow(2).mean()

            # Smoothness (optional) - we'll set lambda_smooth=0 for this experiment
            loss_smooth = (x_fake[:, :, 1:] - x_fake[:, :, :-1]).pow(2).mean()

            # Amplitude / energy matching (per-channel std)
            # Helps when fake looks slightly "too smooth" or has weaker variability than real
            std_real = x_real.std(dim=-1, unbiased=False).mean(dim=0)  # (C,)
            std_fake = x_fake.std(dim=-1, unbiased=False).mean(dim=0)  # (C,)
            loss_amp = (std_fake - std_real).pow(2).mean()

            loss_G_total = (
                loss_G_wgan
                + (cfg.lambda_psd * loss_psd)
                + (cfg.lambda_smooth * loss_smooth)
                + (cfg.lambda_amp * loss_amp)
            )

            opt_G.zero_grad(set_to_none=True)
            loss_G_total.backward()
            if cfg.grad_clip_g is not None:
                torch.nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip_g)
            opt_G.step()

            # EMA update AFTER generator step
            ema.update(G)

            g_losses.append(float(loss_G_wgan.item()))

            batch_pbar.set_postfix(
                G=float(np.mean(g_losses)),
                Dcurve=float(-np.mean(wdist_vals)) if wdist_vals else float("nan"),
                psd=float(loss_psd.item()),
                smooth=float(loss_smooth.item()),
                amp=float(loss_amp.item()),
            )

        mean_g = float(np.mean(g_losses)) if g_losses else float("nan")
        mean_wdist = float(np.mean(wdist_vals)) if wdist_vals else float("nan")

        # Classic-looking curves
        G_hist.append(mean_g)
        D_hist.append(-mean_wdist)  # negative curve

        epoch_pbar.set_postfix(D=-mean_wdist, G=mean_g, wdist=mean_wdist)

        if (epoch + 1) % cfg.save_every == 0 or (epoch + 1) == cfg.epochs:
            ckpt_name = f"better_wgan_gp_epoch_{epoch+1:04d}.pt"
            save_checkpoint(cfg.outdir / "checkpoints" / ckpt_name, epoch + 1, G, D, opt_G, opt_D, ema, cfg)
            save_checkpoint(cfg.outdir / "checkpoints" / "last.pt", epoch + 1, G, D, opt_G, opt_D, ema, cfg)
            plot_loss_curves(D_hist, G_hist, cfg.outdir / "loss_curves.png")

    # -------------------------
    # Final eval (val split) using EMA generator
    # -------------------------
    G_eval = ema.ema_model.to(cfg.device).eval()

    rng = np.random.default_rng(cfg.seed)
    te0 = np.where(y_te == 0)[0]
    te1 = np.where(y_te == 1)[0]
    n0 = min(len(te0), cfg.eval_num_per_class)
    n1 = min(len(te1), cfg.eval_num_per_class)
    idx0 = rng.choice(te0, n0, replace=False) if n0 > 0 else np.array([], dtype=int)
    idx1 = rng.choice(te1, n1, replace=False) if n1 > 0 else np.array([], dtype=int)

    Xt0 = X_te[idx0] if n0 > 0 else None
    Xt1 = X_te[idx1] if n1 > 0 else None

    Xf0 = generate_epochs(G_eval, n0 if n0 > 0 else 0, 0, cfg.z_dim, cfg.device) if n0 > 0 else None
    Xf1 = generate_epochs(G_eval, n1 if n1 > 0 else 0, 1, cfg.z_dim, cfg.device) if n1 > 0 else None

    metrics = {}
    if n0 > 0:
        metrics["left"] = {
            "log_psd_mse": log_psd_mse_torch(Xt0, Xf0, device=cfg.device),
            "global_percentile_mae": global_percentile_mae(Xt0, Xf0),
            "n_real": int(n0),
            "n_fake": int(n0),
        }
    if n1 > 0:
        metrics["right"] = {
            "log_psd_mse": log_psd_mse_torch(Xt1, Xf1, device=cfg.device),
            "global_percentile_mae": global_percentile_mae(Xt1, Xf1),
            "n_real": int(n1),
            "n_fake": int(n1),
        }

    with open(cfg.outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # sample plots (µV)
    if n0 > 0:
        real_uv = Xt0[0] * cfg.clip_uv
        fake_uv = Xf0[0] * cfg.clip_uv
        plot_stacked_pair(real_uv, fake_uv, cfg.tmin, cfg.tmax,
                          "LEFT: Real vs Fake (stacked channels) [EMA G]",
                          cfg.outdir / "samples_left.png")

    if n1 > 0:
        real_uv = Xt1[0] * cfg.clip_uv
        fake_uv = Xf1[0] * cfg.clip_uv
        plot_stacked_pair(real_uv, fake_uv, cfg.tmin, cfg.tmax,
                          "RIGHT: Real vs Fake (stacked channels) [EMA G]",
                          cfg.outdir / "samples_right.png")

    plot_loss_curves(D_hist, G_hist, cfg.outdir / "loss_curves.png")

    print("[DONE] Training complete.")
    print(f"[DONE] Outputs in: {cfg.outdir.resolve()}")


if __name__ == "__main__":
    main()
