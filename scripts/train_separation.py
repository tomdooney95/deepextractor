"""Training script for the multi-detector time-domain signal/glitch separator.

Trains a plain UNET1D (no LSTM/attention bottleneck) on the sharded HDF5
output of generate_separation_data.py, with the presence-flag + masked-loss
mechanism from HDF5SeparationDataset/train_fn_separation: detectors not
listed in --active-detectors are simulated as permanently offline (input
zeroed post-scaling, a presence-flag channel appended, output channels
excluded from the loss so those output-head weights get exactly zero
gradient and stay at initialization -- ready for a later warm start once
real data for that detector exists).

All configuration is via command-line arguments so this can be submitted
directly to a SLURM scheduler without editing.

Example (V1 permanently off, matching real O3/O4 data availability):
    python scripts/train_separation.py \\
        --shard-dir ~/data_separation \\
        --detectors H1 L1 V1 --active-detectors H1 L1 \\
        --features 64 128 256 512 1024 2048 --dropout-p 0.1 --norm gn \\
        --out checkpoints/separation_v1_off \\
        --epochs 300 --batch-size 32 --workers 16

Example (Snellius SLURM):
    python scripts/train_separation.py \\
        --shard-dir $TMPDIR/data_separation \\
        --detectors H1 L1 V1 --active-detectors H1 L1 \\
        --features 64 128 256 512 1024 2048 --dropout-p 0.1 --norm gn \\
        --out $HOME/checkpoints/separation_v1_off \\
        --epochs 300 --batch-size 32 --workers 16 --amp
"""

import argparse
import logging
import os
import pickle
import subprocess
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deepextractor.data import ChannelStandardScaler, HDF5SeparationDataset
from deepextractor.models import UNET1D
from deepextractor.training import eval_fn_separation, train_fn_separation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_gpu() -> int:
    """Return the GPU index with the most free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, check=True,
        )
        free = [int(x) for x in result.stdout.strip().split("\n")]
        return free.index(max(free))
    except Exception:
        return 0


def _load_or_fit_scaler(path: str, shard_dir: str, detectors: list[str]) -> ChannelStandardScaler:
    path = Path(path)
    if path.is_file():
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Loaded scaler from %s (mean_=%s, scale_=%s)", path, scaler.mean_, scaler.scale_)
        return scaler

    logger.info("No scaler found at %s — fitting on train split (%s)...", path, shard_dir)
    scaler = ChannelStandardScaler().fit_from_separation_shards(
        shard_dir, split="train", detectors=detectors,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Fitted and saved scaler to %s (mean_=%s, scale_=%s)", path, scaler.mean_, scaler.scale_)
    return scaler


def _save_losses(out_dir: Path, prefix: str, history: dict):
    import numpy as np
    for name, arr in history.items():
        np.save(out_dir / f"{name}_{prefix}.npy", np.array(arr))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train the multi-detector signal/glitch separation UNET1D")

    # --- Data ---
    p.add_argument("--shard-dir", required=True,
                   help="Directory of separation_shard_*.h5 files from generate_separation_data.py")
    p.add_argument("--detectors", nargs="+", default=["H1", "L1", "V1"],
                   help="All detectors present in the generated data.")
    p.add_argument("--active-detectors", nargs="+", required=True,
                   help="Subset of --detectors to actually train with active this run "
                        "(the rest are simulated as permanently offline via the presence-flag "
                        "+ masked-loss mechanism). Pass all of --detectors to keep every "
                        "detector active while still using the flag-channel input shape.")
    p.add_argument("--scaler", required=True,
                   help="Path to a pickled ChannelStandardScaler. Fit on --shard-dir's train "
                        "split and saved here if it doesn't already exist.")
    p.add_argument("--target-signal-only", action="store_true", default=False)

    # --- Model ---
    p.add_argument("--features", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--dropout-p", type=float, default=0.1,
                   help="Dropout in decoder+bottleneck only (never encoder). "
                        "Trained-in from the start, not a separate MC-Dropout run -- "
                        "toggle it back on at inference time via enable_mc_dropout().")
    p.add_argument("--norm", default="gn", choices=["bn", "gn"])
    p.add_argument("--num-groups", type=int, default=8)

    # --- Training ---
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="Batches pre-loaded per DataLoader worker. Ignored when --workers is 0.")
    p.add_argument("--amp", action="store_true", default=False,
                   help="Enable mixed-precision (disabled by default -- prior TD training on "
                        "Snellius found AMP unstable with whitened targets)")
    p.add_argument("--patience", type=int, default=9, help="Early stopping patience (epochs)")
    p.add_argument("--lr-patience", type=int, default=4, help="ReduceLROnPlateau patience")
    p.add_argument("--lr-factor", type=float, default=0.1)

    # --- Checkpointing / output ---
    p.add_argument("--out", required=True, help="Output directory for checkpoints and loss arrays")
    p.add_argument("--resume", default=None, help="Path to a .pth.tar checkpoint to resume from")
    p.add_argument("--save-every", type=int, default=10, help="Save loss arrays every N epochs (0 = only at end)")

    # --- Device ---
    p.add_argument("--device", default=None, help="Torch device string. Auto-selects best GPU if None.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    unknown = set(args.active_detectors) - set(args.detectors)
    if unknown:
        raise ValueError(f"--active-detectors {sorted(unknown)} not in --detectors {args.detectors}")

    # --- Device ---
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{_best_gpu()}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)
    logger.info("Detectors: %s | Active: %s | Offline: %s",
                args.detectors, args.active_detectors,
                sorted(set(args.detectors) - set(args.active_detectors)))

    # --- Output dirs ---
    out_dir = Path(args.out)
    loss_dir = out_dir / "losses"
    out_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    # --- Scaler ---
    scaler = _load_or_fit_scaler(args.scaler, args.shard_dir, args.detectors)

    # --- Datasets & loaders ---
    train_ds = HDF5SeparationDataset(
        args.shard_dir, split="train", detectors=args.detectors,
        input_scaler=scaler, target_signal_only=args.target_signal_only,
        active_detectors=args.active_detectors,
    )
    val_ds = HDF5SeparationDataset(
        args.shard_dir, split="val", detectors=args.detectors,
        input_scaler=scaler, target_signal_only=args.target_signal_only,
        active_detectors=args.active_detectors,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    logger.info("Train samples: %d  Val samples: %d", len(train_ds), len(val_ds))

    # --- Model ---
    n_det = len(args.detectors)
    in_channels = 2 * n_det       # strain + presence-flag channels
    out_channels = n_det if args.target_signal_only else 2 * n_det
    model = UNET1D(
        in_channels=in_channels, out_channels=out_channels, features=args.features,
        dropout_p=args.dropout_p, norm=args.norm, num_groups=args.num_groups,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: UNET1D  in=%d out=%d features=%s norm=%s dropout_p=%.2f  (%.1fM params)",
        in_channels, out_channels, args.features, args.norm, args.dropout_p, n_params / 1e6,
    )

    # --- Optimiser / scheduler ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    # Config recorded in every checkpoint -- so "did this run use dropout / which
    # detectors were active" is never something we have to reconstruct from
    # directory names or memory later.
    run_config = dict(
        detectors=args.detectors, active_detectors=args.active_detectors,
        features=args.features, dropout_p=args.dropout_p, norm=args.norm,
        num_groups=args.num_groups, in_channels=in_channels, out_channels=out_channels,
        target_signal_only=args.target_signal_only, batch_size=args.batch_size, lr=args.lr,
    )

    # --- Resume ---
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt["scheduler"]["best"]
        logger.info("Resumed from %s (epoch %d, best_val=%.4e)", args.resume, start_epoch, best_val_loss)

    # --- Training loop ---
    history = {
        "train_total": [], "train_bg": [], "train_sig": [],
        "val_total": [], "val_bg": [], "val_sig": [],
    }
    es_counter = 0
    epoch = start_epoch

    for epoch in range(start_epoch, start_epoch + args.epochs):
        logger.info("Epoch %d/%d", epoch + 1, start_epoch + args.epochs)

        train_total, train_bg, train_sig = train_fn_separation(
            train_loader, model, optimizer, grad_scaler, device, use_amp=args.amp,
        )
        val_total, val_bg, val_sig = eval_fn_separation(val_loader, model, device)
        val_loss = val_total  # what scheduler/early-stopping/checkpointing act on

        history["train_total"].append(train_total)
        history["train_bg"].append(train_bg)
        history["train_sig"].append(train_sig)
        history["val_total"].append(val_total)
        history["val_bg"].append(val_bg)
        history["val_sig"].append(val_sig)
        logger.info(
            "LR=%.3e | train total=%.4e bg=%.4e sig=%.4e | val total=%.4e bg=%.4e sig=%.4e",
            optimizer.param_groups[0]["lr"], train_total, train_bg, train_sig, val_total, val_bg, val_sig,
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            es_counter = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "mode": "separation",
                "config": run_config,
            }
            ckpt_path = out_dir / "checkpoint_best.pth.tar"
            torch.save(ckpt, ckpt_path)
            logger.info("Val improved to %.4e — saved %s", best_val_loss, ckpt_path)
        else:
            es_counter += 1
            logger.info("No val improvement (%d/%d)", es_counter, args.patience)

        if es_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_losses(loss_dir, "periodic", history)

    _save_losses(loss_dir, "final", history)
    logger.info("Training complete. Losses saved to %s", loss_dir)


if __name__ == "__main__":
    main()
