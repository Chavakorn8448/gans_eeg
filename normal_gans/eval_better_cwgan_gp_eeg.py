#!/usr/bin/env python3
"""
eval_better_cwgan_gp_eeg.py

Standalone EVALUATION + PLOTTING script for your improved conditional WGAN-GP EEG model.
- Does NOT train.
- Loads a saved checkpoint (.pt) from your training run.
- Rebuilds the dataset the same way (preprocess + normalize + stratified split).
- Evaluates on the *TEST split (unseen)* by default.
- Generates:
  - metrics_test.json
  - samples_left_test.png
  - samples_right_test.png

Example:
  python eval_better_cwgan_gp_eeg.py \
    --data-dir BCICIV_2b_gdf \
    --ckpt runs/vanilla_improved_2/checkpoints/last.pt \
    --outdir runs/vanilla_improved_2/eval_test \
    --eval-num-per-class 256

Notes:
- This script mirrors the key parts of your fixed training file, but only runs the final evaluation portion.
- It will use EMA generator if present in the checkpoint (recommended). You can disable via --no-ema.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import pathlib
import torch.serialization

import mne

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Reproducibility
# =========================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# Data preprocessing (same as training)
# =========================
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
      X: (N, C, T) EEG only (in volts, float32)
      y: (N,) 0=left, 1=right
    """
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR" if not verbose else None)

    for ch in raw.ch_names:
        if "EOG" in ch.upper():
            raw.set_channel_types({ch: "eog"})

    raw.filter(l_freq, h_freq, verbose="ERROR" if not verbose else None)
    raw.notch_filter(notch_hz, verbose="ERROR" if not verbose else None)

    events, event_id_all = mne.events_from_annotations(raw, verbose="ERROR" if not verbose else None)

    # map event ids from annotation strings
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


# =========================
# Model (same as training)
# =========================
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


# =========================
# Metrics + plots (same as training)
# =========================
def batch_log_psd(x: torch.Tensor) -> torch.Tensor:
    Xf = torch.fft.rfft(x, dim=-1)
    psd = (Xf.real ** 2 + Xf.imag ** 2)
    return torch.log(psd + 1e-8).mean(dim=0)


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


def plot_stacked_pair(
    real_uv: np.ndarray,
    fake_uv: np.ndarray,
    tmin: float,
    tmax: float,
    title: str,
    out_png: Path,
    ch_names: Optional[list] = None,
    sep_uv: Optional[float] = None,
):
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
            ax.text(
                t[0] - 0.02 * (tmax - tmin),
                offsets[i],
                ch_names[i],
                va="center",
                ha="right",
                fontsize=8,
            )
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


@torch.no_grad()
def generate_epochs(G: nn.Module, n: int, label01: int, z_dim: int, device: str) -> np.ndarray:
    y = torch.full((n,), label01, dtype=torch.long, device=device)
    z = torch.randn(n, z_dim, device=device)
    x = G(z, y)
    return x.detach().cpu().numpy()


# =========================
# Config (read from checkpoint if present)
# =========================
@dataclass
class EvalConfig:
    data_dir: Path
    ckpt: Path
    outdir: Path

    # dataset pipeline (defaults match your training file)
    event_left: str = "769"
    event_right: str = "770"
    tmin: float = 0.0
    tmax: float = 4.0
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch_hz: float = 50.0
    eog_reject_uv: float = 150.0
    resample_hz: int = 250
    clip_uv: float = 150.0

    # split
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42

    # eval
    z_dim: int = 128
    eval_num_per_class: int = 256
    use_ema: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose_mne: bool = False


def _merge_from_ckpt_config(cfg: EvalConfig, ckpt_dict: dict) -> EvalConfig:
    """
    If checkpoint contains a saved training config, reuse its preprocessing/split params
    so eval matches training exactly.
    """
    c = ckpt_dict.get("config", None)
    if not isinstance(c, dict):
        return cfg

    # Only overwrite keys that exist on EvalConfig
    for k in list(c.keys()):
        if hasattr(cfg, k):
            setattr(cfg, k, c[k])

    # z_dim is critical (for Generator shape)
    if "z_dim" in c:
        cfg.z_dim = int(c["z_dim"])

    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="Folder containing *T.gdf files")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint .pt (e.g., .../checkpoints/last.pt)")
    ap.add_argument("--outdir", type=str, default="eval_out", help="Output directory for eval results")
    ap.add_argument("--eval-num-per-class", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--no-ema", action="store_true", help="Use raw generator_state (not EMA) even if EMA exists")
    ap.add_argument("--verbose-mne", action="store_true", help="Verbose MNE logs")
    args = ap.parse_args()

    cfg = EvalConfig(
        data_dir=Path(args.data_dir),
        ckpt=Path(args.ckpt),
        outdir=Path(args.outdir),
        eval_num_per_class=int(args.eval_num_per_class),
        seed=int(args.seed),
        device=str(args.device),
        use_ema=(not args.no_ema),
        verbose_mne=bool(args.verbose_mne),
    )

    seed_everything(cfg.seed)
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    if not cfg.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt.resolve()}")

    # -------------------------
    # Load checkpoint
    # -------------------------
    # Allowlist PosixPath for safe weights-only loading (PyTorch 2.6+)
    torch.serialization.add_safe_globals([pathlib.PosixPath])

    ckpt = torch.load(cfg.ckpt, map_location=cfg.device, weights_only=True)
    cfg = _merge_from_ckpt_config(cfg, ckpt)

    # -------------------------
    # Rebuild dataset exactly like training, then take TEST split
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

    X = normalize_to_unit(X_volts, clip_uv=float(cfg.clip_uv))
    print(f"[DATA] Normalized: min={float(X.min()):.4f}, max={float(X.max()):.4f}, clip_uv={cfg.clip_uv}")

    X_tr, y_tr, X_val, y_val, X_te, y_te = stratified_split_train_val_test(
        X, y, val_frac=float(cfg.val_frac), test_frac=float(cfg.test_frac), seed=int(cfg.seed)
    )
    print(f"[DATA] Split: train={X_tr.shape[0]}, val={X_val.shape[0]}, test={X_te.shape[0]} (EVAL ON TEST)")

    # -------------------------
    # Build generator and load weights
    # -------------------------
    C, T = X.shape[1], X.shape[2]
    G = Generator(cfg.z_dim, 2, C, T).to(cfg.device).eval()

    use_ema = cfg.use_ema and ("ema_G_state" in ckpt)
    if use_ema:
        print("[MODEL] Using EMA generator weights from checkpoint (ema_G_state).")
        G.load_state_dict(ckpt["ema_G_state"])
    else:
        print("[MODEL] Using raw generator weights from checkpoint (generator_state).")
        G.load_state_dict(ckpt["generator_state"])

    # -------------------------
    # Sample real test per class + generate fake per class
    # -------------------------
    rng = np.random.default_rng(cfg.seed)
    te0 = np.where(y_te == 0)[0]
    te1 = np.where(y_te == 1)[0]
    n0 = min(len(te0), int(cfg.eval_num_per_class))
    n1 = min(len(te1), int(cfg.eval_num_per_class))

    idx0 = rng.choice(te0, n0, replace=False) if n0 > 0 else np.array([], dtype=int)
    idx1 = rng.choice(te1, n1, replace=False) if n1 > 0 else np.array([], dtype=int)

    Xt0 = X_te[idx0] if n0 > 0 else None
    Xt1 = X_te[idx1] if n1 > 0 else None

    Xf0 = generate_epochs(G, n0 if n0 > 0 else 0, 0, cfg.z_dim, cfg.device) if n0 > 0 else None
    Xf1 = generate_epochs(G, n1 if n1 > 0 else 0, 1, cfg.z_dim, cfg.device) if n1 > 0 else None

    # -------------------------
    # Metrics (TEST)
    # -------------------------
    metrics = {"split": "test", "use_ema": bool(use_ema), "eval_num_per_class": int(cfg.eval_num_per_class)}
    if n0 > 0:
        metrics["left"] = {
            "log_psd_mse": log_psd_mse_torch(Xt0, Xf0, device=cfg.device),
            "global_percentile_mae": global_percentile_mae(Xt0, Xf0),
            "n_real": int(n0),
            "n_fake": int(n0),
            "real_minmax": [float(Xt0.min()), float(Xt0.max())],
            "fake_minmax": [float(Xf0.min()), float(Xf0.max())],
        }
    if n1 > 0:
        metrics["right"] = {
            "log_psd_mse": log_psd_mse_torch(Xt1, Xf1, device=cfg.device),
            "global_percentile_mae": global_percentile_mae(Xt1, Xf1),
            "n_real": int(n1),
            "n_fake": int(n1),
            "real_minmax": [float(Xt1.min()), float(Xt1.max())],
            "fake_minmax": [float(Xf1.min()), float(Xf1.max())],
        }

    out_metrics = cfg.outdir / "metrics_test.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OUT] Wrote metrics: {out_metrics.resolve()}")

    # -------------------------
    # Plot: one real vs one fake (TEST), in µV
    # -------------------------
    if n0 > 0:
        real_uv = Xt0[0] * float(cfg.clip_uv)
        fake_uv = Xf0[0] * float(cfg.clip_uv)
        out_png = cfg.outdir / "samples_left_test.png"
        plot_stacked_pair(
            real_uv, fake_uv, cfg.tmin, cfg.tmax,
            f"LEFT (TEST): Real vs Fake (stacked channels) [{'EMA' if use_ema else 'RAW'} G]",
            out_png,
        )
        print(f"[OUT] Wrote plot: {out_png.resolve()}")

    if n1 > 0:
        real_uv = Xt1[0] * float(cfg.clip_uv)
        fake_uv = Xf1[0] * float(cfg.clip_uv)
        out_png = cfg.outdir / "samples_right_test.png"
        plot_stacked_pair(
            real_uv, fake_uv, cfg.tmin, cfg.tmax,
            f"RIGHT (TEST): Real vs Fake (stacked channels) [{'EMA' if use_ema else 'RAW'} G]",
            out_png,
        )
        print(f"[OUT] Wrote plot: {out_png.resolve()}")

    print("[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
