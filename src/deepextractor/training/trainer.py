"""
CLI entry point for training DeepExtractor models.

Usage::

    deepextractor-train --model DeepExtractor_257 --data-dir data/pycbc_noise/spectrogram_domain_clean_glitch_129/

"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from deepextractor.models.architectures import (
    Autoencoder1D,
    Autoencoder2D,
    DnCNN1D,
    ModifiedAutoencoder2D,
    UNET1D,
    UNET2D,
)
from torch.utils.data import DataLoader

from deepextractor.data.datasets import HDF5ReconstructionDataset
from deepextractor.training.train_fn import train_fn
from deepextractor.utils.checkpoints import load_checkpoint, load_optimizer, save_checkpoint
from deepextractor.utils.io import check_accuracy, get_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Registry of available model architectures
MODEL_REGISTRY = {
    "UNET1D": lambda: UNET1D(in_channels=1, out_channels=1),
    "DnCNN1D": DnCNN1D,
    "Autoencoder1D": lambda: Autoencoder1D(in_channels=1, out_channels=1),
    "UNET2D": lambda: UNET2D(in_channels=2, out_channels=2),
    "UNET2D_glitch_target": lambda: UNET2D(in_channels=2, out_channels=2),
    "Autoencoder2D": lambda: Autoencoder2D(in_channels=2, out_channels=2),
    "ModifiedAutoencoder2D": lambda: ModifiedAutoencoder2D(in_channels=2, out_channels=2),
    "DeepExtractor_65": lambda: UNET2D(in_channels=2, out_channels=2),
    "DeepExtractor_129": lambda: UNET2D(in_channels=2, out_channels=2),
    "DeepExtractor_257": lambda: UNET2D(in_channels=2, out_channels=2),
}


def main():
    parser = argparse.ArgumentParser(
        description="Train a DeepExtractor model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        default="DeepExtractor_257",
        help="Model architecture to train.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument(
        "--dropout-p", type=float, default=0.0,
        help="Dropout probability for MC Dropout uncertainty estimation. "
             "Applied to the bottleneck and decoder DoubleConv2D blocks. "
             "0.0 (default) = no dropout, matches existing behaviour. "
             "Typical values: 0.1–0.2. Requires retraining from scratch.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--time-domain",
        action="store_true",
        help="Train on time-domain data instead of spectrograms.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the training .npy arrays (--format npy).",
    )
    parser.add_argument(
        "--hdf5",
        action="store_true",
        help="Load data from per-key HDF5 files produced by deepextractor-specgen --format hdf5.",
    )
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=None,
        help="Directory containing per-key spectrogram HDF5 files (--hdf5 mode). "
             "Expects noisy_glitch_train.h5, background_train.h5, etc.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches pre-loaded per DataLoader worker. "
             "Increase to hide NFS I/O latency (e.g. 4 or 8). "
             "Ignored when --num-workers is 0.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--loss-dir",
        type=Path,
        default=Path("losses"),
        help="Directory to save loss arrays.",
    )
    parser.add_argument(
        "--transfer-learn",
        action="store_true",
        help="Resume from an existing checkpoint (transfer learning).",
    )
    parser.add_argument(
        "--bilby-noise",
        action="store_true",
        help="Use bilby noise suffix in checkpoint filenames.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use, e.g. 'cuda:0' or 'cpu'. Auto-detected if not set.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=9,
        help="Number of epochs without improvement before stopping.",
    )
    args = parser.parse_args()

    # --- Device ---
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{torch.cuda.device_count() - 1}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # --- Model ---
    model_name = args.model
    _unet2d_names = {
        "UNET2D", "UNET2D_glitch_target", "DeepExtractor_65",
        "DeepExtractor_129", "DeepExtractor_257",
    }
    if model_name in _unet2d_names:
        model = UNET2D(in_channels=2, out_channels=2, dropout_p=args.dropout_p).to(device)
    else:
        model = MODEL_REGISTRY[model_name]().to(device)
    if args.dropout_p > 0.0:
        logger.info(f"Training {model_name} on {device} with MC Dropout p={args.dropout_p}")
    else:
        logger.info(f"Training {model_name} on {device}")

    # --- Directories ---
    noise_ext = "bilby_noise" if args.bilby_noise else "pycbc_noise"
    tl_ext = "transfer_learn" if args.transfer_learn else "base"
    model_checkpoint_dir = args.checkpoint_dir / f"{model_name}_checkpoints"
    model_loss_dir = args.loss_dir / f"{model_name}_{noise_ext}_{tl_ext}_losses"

    for d in [args.checkpoint_dir, model_checkpoint_dir, args.loss_dir, model_loss_dir]:
        os.makedirs(d, exist_ok=True)

    # --- Data loaders ---
    if args.hdf5:
        if args.spec_dir is None:
            parser.error("--hdf5 requires --spec-dir")
        spec_dir = args.spec_dir
        train_ds = HDF5ReconstructionDataset(
            input_h5=str(spec_dir / "noisy_glitch_train.h5"),
            target_h5=str(spec_dir / "background_train.h5"),
        )
        val_ds = HDF5ReconstructionDataset(
            input_h5=str(spec_dir / "noisy_glitch_val.h5"),
            target_h5=str(spec_dir / "background_val.h5"),
        )
        loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=False,           # data pre-shuffled at generation time
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        train_loader = DataLoader(train_ds, **loader_kwargs)
        val_loader   = DataLoader(val_ds,   **loader_kwargs)
    elif args.time_domain:
        if args.data_dir is None:
            parser.error("--time-domain requires --data-dir")
        data_dir = args.data_dir
        train_loader, val_loader = get_loaders(
            str(data_dir / "glitch_train_scaled.npy"),
            str(data_dir / "background_train_scaled.npy"),
            str(data_dir / "glitch_val_scaled.npy"),
            str(data_dir / "background_val_scaled.npy"),
            args.batch_size, None, None, args.num_workers, True, time_domain=True,
        )
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required unless --hdf5 is set")
        data_dir = args.data_dir
        train_loader, val_loader = get_loaders(
            str(data_dir / "glitch_train_scaled_mag_phase.npy"),
            str(data_dir / "background_train_scaled_mag_phase.npy"),
            str(data_dir / "glitch_val_scaled_mag_phase.npy"),
            str(data_dir / "background_val_scaled_mag_phase.npy"),
            batch_size=args.batch_size, train_transform=None, val_transform=None,
            num_workers=args.num_workers, pin_memory=True, time_domain=False,
        )

    # --- Optimizer and scheduler ---
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)

    start_epoch = 0
    if args.transfer_learn:
        checkpoint_path = model_checkpoint_dir / "checkpoint_best.pth.tar"
        try:
            logger.info("Loading model checkpoint...")
            checkpoint = torch.load(str(checkpoint_path))
            load_checkpoint(checkpoint, model)
            load_optimizer(checkpoint, optimizer)
            start_epoch = checkpoint.get("epoch", start_epoch)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return

    # --- Training loop ---
    train_losses, train_noise_losses, train_constraint_losses = [], [], []
    val_losses, val_noise_losses, val_constraint_losses = [], [], []
    scaler = torch.amp.GradScaler("cuda") if device.startswith("cuda") else torch.amp.GradScaler("cpu")
    early_stopping_counter = 0
    best_val_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_noise_loss, train_constraint_loss = train_fn(
            train_loader, model, model_name, optimizer, loss_fn, scaler, device
        )
        train_losses.append(train_loss)
        train_noise_losses.append(train_noise_loss)
        train_constraint_losses.append(train_constraint_loss)

        val_loss, val_noise_loss, val_constraint_loss = check_accuracy(
            val_loader, model, model_name, device=device
        )
        val_losses.append(val_loss)
        val_noise_losses.append(val_noise_loss)
        val_constraint_losses.append(val_constraint_loss)

        scheduler.step(val_loss)
        current_lr = scheduler.optimizer.param_groups[0]["lr"]
        logger.info(f"Current learning rate: {current_lr}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            ckpt_filename = (
                model_checkpoint_dir / f"checkpoint_best_{noise_ext}_{tl_ext}.pth.tar"
            )
            save_checkpoint(checkpoint, str(ckpt_filename))
            logger.info(f"Validation loss improved to {best_val_loss:.6}. Checkpoint saved.")
        else:
            early_stopping_counter += 1
            logger.info(
                f"No improvement. Early stopping counter: "
                f"{early_stopping_counter}/{args.early_stopping_patience}"
            )

        if early_stopping_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping after {epoch + 1} epochs.")
            break

        if epoch % 10 == 0:
            _save_losses(
                model_loss_dir, start_epoch, epoch,
                train_losses, train_noise_losses, train_constraint_losses,
                val_losses, val_noise_losses, val_constraint_losses,
            )

    _save_losses(
        model_loss_dir, start_epoch, epoch,
        train_losses, train_noise_losses, train_constraint_losses,
        val_losses, val_noise_losses, val_constraint_losses,
    )
    logger.info("Training complete. Losses saved.")


def _save_losses(
    loss_dir, start_epoch, epoch,
    train_losses, train_noise_losses, train_constraint_losses,
    val_losses, val_noise_losses, val_constraint_losses,
):
    np.save(str(loss_dir / f"train_losses_epoch_{start_epoch}_to_{epoch}.npy"), np.array(train_losses))
    np.save(str(loss_dir / f"train_noise_losses_{start_epoch}_to_{epoch}.npy"), np.array(train_noise_losses))
    np.save(str(loss_dir / f"train_constraint_losses_{start_epoch}_to_{epoch}.npy"), np.array(train_constraint_losses))
    np.save(str(loss_dir / f"val_losses_{start_epoch}_to_{epoch}.npy"), np.array(val_losses))
    np.save(str(loss_dir / f"val_noise_losses_{start_epoch}_to_{epoch}.npy"), np.array(val_noise_losses))
    np.save(str(loss_dir / f"val_constraint_losses_{start_epoch}_to_{epoch}.npy"), np.array(val_constraint_losses))


if __name__ == "__main__":
    main()
