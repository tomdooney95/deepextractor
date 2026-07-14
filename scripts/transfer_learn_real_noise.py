"""
Transfer-learn a 4s DeepExtractor checkpoint on real detector backgrounds.

Loads background pickle files produced by get_clean_backgrounds.py, splits 8s
samples into 4s halves, applies a chronological 90/10 train/val split within
each detector/run source, combines across sources, injects synthetic glitches,
fits a new StandardScaler on the real-noise train split, writes time-domain HDF5
files, and fine-tunes the model at a reduced learning rate.

Pickle format expected:  {run: {ifo: {'samples': ndarray(N, 32768), 'gps_starts': ndarray(N,)}}}

Usage — O3 TL from bilby-pretrained checkpoint::

    python scripts/transfer_learn_real_noise.py \\
        --backgrounds ~/deepextractor/backgrounds_O3a.pkl \\
                      ~/deepextractor/backgrounds_O3b.pkl \\
        --checkpoint checkpoints_4s/DeepExtractor_257_checkpoints/checkpoint_best_bilby_noise_base.pth.tar \\
        --label o3 \\
        --checkpoint-dir checkpoints_4s_o3 \\
        --loss-dir losses_4s_o3 \\
        --hdf5-dir data_4s_real/o3_hdf5 \\
        --epochs 50 --lr 1e-5

Usage — O4 TL from O3 checkpoint::

    python scripts/transfer_learn_real_noise.py \\
        --backgrounds ~/deepextractor/backgrounds_O4a.pkl \\
                      ~/deepextractor/backgrounds_O4b.pkl \\
        --checkpoint checkpoints_4s_o3/DeepExtractor_257_checkpoints/checkpoint_best_o3_tl.pth.tar \\
        --label o4 \\
        --checkpoint-dir checkpoints_4s_o4 \\
        --loss-dir losses_4s_o4 \\
        --hdf5-dir data_4s_real/o4_hdf5 \\
        --epochs 50 --lr 1e-5
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from deepextractor.generation.generate_timeseries import generate_synthetic_data
from deepextractor.models.architectures import UNET2D
from deepextractor.training.train_fn import train_fn
from deepextractor.utils.checkpoints import load_checkpoint, load_optimizer, save_checkpoint
from deepextractor.utils.io import check_accuracy
from deepextractor.utils.stft import apply_stft

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 4096
LEN_8S = 8 * SAMPLE_RATE   # 32768 — raw background sample length
LEN_4S = 4 * SAMPLE_RATE   # 16384 — model input length
N_FFT, HOP_LENGTH, WIN_LENGTH = 512, 32, 64

IFOS = ["H1", "L1"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_backgrounds(pkl_paths: list[str], max_per_source: int | None = None) -> dict[str, np.ndarray]:
    """
    Load background pickle files and return per-IFO 8s arrays.

    Returns
    -------
    dict mapping ifo ('H1', 'L1') → ndarray(N, 32768), chronologically ordered.
    The arrays from multiple pkl files are concatenated in the order provided.
    """
    ifo_arrays: dict[str, list[np.ndarray]] = {ifo: [] for ifo in IFOS}
    total_sources = 0

    for pkl_path in pkl_paths:
        logger.info(f"Loading {pkl_path} ...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        for run, run_data in data.items():
            for ifo in IFOS:
                if ifo not in run_data:
                    logger.warning(f"  {run} has no {ifo} data — skipping")
                    continue
                samples = run_data[ifo]["samples"]  # (N, 32768), float32
                if max_per_source is not None:
                    samples = samples[:max_per_source]
                ifo_arrays[ifo].append(samples)
                logger.info(f"  {run} {ifo}: {len(samples)} × 8s samples  "
                            f"[{samples.nbytes / 1e9:.2f} GB]")
                total_sources += 1

    if total_sources == 0:
        raise RuntimeError(f"No samples found in {pkl_paths}")

    result = {}
    for ifo in IFOS:
        if ifo_arrays[ifo]:
            result[ifo] = np.concatenate(ifo_arrays[ifo], axis=0)
            logger.info(f"{ifo} total: {len(result[ifo])} × 8s samples")
        else:
            logger.warning(f"No {ifo} samples across all provided pkl files")
    return result


def split_to_4s(arr_8s: np.ndarray) -> np.ndarray:
    """Split (N, 32768) → (2N, 16384) by taking first and second 4s halves."""
    n = arr_8s.shape[0]
    return arr_8s.reshape(n, 2, LEN_4S).reshape(n * 2, LEN_4S)


def chronological_split(
    arr: np.ndarray, val_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split array chronologically: first (1-val_frac) → train, last val_frac → val.

    The split is done on the original 8s segments BEFORE splitting to 4s,
    so both halves of each 8s segment always end up in the same partition.
    """
    n = len(arr)
    n_val = max(1, int(np.round(n * val_frac)))
    n_train = n - n_val
    return arr[:n_train], arr[n_train:]


# ── In-memory injection ───────────────────────────────────────────────────────

def inject_in_memory(
    train_4s: np.ndarray,
    val_4s: np.ndarray,
    batch_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Inject glitches in-memory (batched to avoid peak RAM) and fit a scaler.

    Returns (train_noisy, train_bg, val_noisy, val_bg, scaler).
    """
    def _inject(arr, phase):
        noisy_parts, bg_parts = [], []
        for start in tqdm(range(0, len(arr), batch_size), desc=f"Injecting ({phase})"):
            batch = arr[start : start + batch_size].copy().astype(np.float32)
            n, b = generate_synthetic_data(batch, bilby_noise=False, phase=phase,
                                           show_progress=False)
            noisy_parts.append(n.astype(np.float32))
            bg_parts.append(b.astype(np.float32))
        return np.concatenate(noisy_parts), np.concatenate(bg_parts)

    train_noisy, train_bg = _inject(train_4s, "train")
    val_noisy,   val_bg   = _inject(val_4s,   "val")

    logger.info("Fitting StandardScaler on noisy train ...")
    scaler = StandardScaler()
    scaler.fit(train_noisy.reshape(-1, 1))

    return train_noisy, train_bg, val_noisy, val_bg, scaler


class RealNoiseInMemoryDataset(Dataset):
    """
    Holds (noisy, background) time-domain arrays in RAM and computes STFT on-the-fly.

    On Linux/macOS the numpy arrays are shared copy-on-write across DataLoader
    worker processes (forked), so memory overhead with num_workers > 0 is small.
    """

    def __init__(
        self,
        noisy: np.ndarray,
        background: np.ndarray,
        scaler_mean: float,
        scaler_scale: float,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ):
        self.noisy = noisy
        self.background = background
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def __len__(self) -> int:
        return len(self.noisy)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy = (self.noisy[idx] - self.scaler_mean) / self.scaler_scale
        bg    = (self.background[idx] - self.scaler_mean) / self.scaler_scale
        noisy_stft = apply_stft(
            noisy[np.newaxis], self.n_fft, self.hop_length, self.win_length, self.window
        ).squeeze(0)
        bg_stft = apply_stft(
            bg[np.newaxis], self.n_fft, self.hop_length, self.win_length, self.window
        ).squeeze(0)
        return noisy_stft, bg_stft


# ── HDF5 generation ───────────────────────────────────────────────────────────

def inject_and_write_hdf5(
    train_4s: np.ndarray,
    val_4s: np.ndarray,
    hdf5_dir: Path,
    batch_size: int = 1000,
) -> StandardScaler:
    """
    Inject glitches into real-noise segments and write time-domain HDF5 files.

    Processes in batches to avoid peak memory proportional to the full dataset.
    Fits a StandardScaler on the noisy train data via partial_fit.

    Returns
    -------
    Fitted StandardScaler (also saved to hdf5_dir/scaler_real.pkl).
    """
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    train_h5 = hdf5_dir / "train.h5"
    val_h5 = hdf5_dir / "val.h5"
    scaler_path = hdf5_dir / "scaler_real.pkl"

    for split_name, noise_4s, h5_path in [
        ("train", train_4s, train_h5),
        ("val",   val_4s,   val_h5),
    ]:
        n_total = len(noise_4s)
        logger.info(f"Generating {split_name}: {n_total} samples → {h5_path}")
        with h5py.File(h5_path, "w") as f:
            noisy_ds = f.create_dataset(
                "noisy", shape=(0, LEN_4S), maxshape=(None, LEN_4S),
                dtype=np.float32, chunks=(min(batch_size, n_total), LEN_4S),
            )
            bg_ds = f.create_dataset(
                "background", shape=(0, LEN_4S), maxshape=(None, LEN_4S),
                dtype=np.float32, chunks=(min(batch_size, n_total), LEN_4S),
            )
            wptr = 0
            for start in tqdm(range(0, n_total, batch_size),
                              desc=f"Injecting ({split_name})"):
                end = min(start + batch_size, n_total)
                batch = noise_4s[start:end].copy().astype(np.float32)
                noisy_batch, bg_batch = generate_synthetic_data(
                    batch, bilby_noise=False, phase=split_name,
                    show_progress=False,
                )
                n_written = noisy_batch.shape[0]
                noisy_ds.resize(wptr + n_written, axis=0)
                bg_ds.resize(wptr + n_written, axis=0)
                noisy_ds[wptr : wptr + n_written] = noisy_batch.astype(np.float32)
                bg_ds[wptr : wptr + n_written] = bg_batch.astype(np.float32)
                wptr += n_written

    # Fit scaler on noisy train via partial_fit (stream from HDF5)
    logger.info("Fitting StandardScaler on noisy train ...")
    scaler = StandardScaler()
    with h5py.File(train_h5, "r") as f:
        ds = f["noisy"]
        for start in tqdm(range(0, ds.shape[0], batch_size), desc="Fitting scaler"):
            end = min(start + batch_size, ds.shape[0])
            scaler.partial_fit(ds[start:end].reshape(-1, 1))

    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    logger.info(f"Scaler saved to {scaler_path}")

    return scaler


# ── Dataset ───────────────────────────────────────────────────────────────────

class RealNoiseHDF5Dataset(Dataset):
    """
    Reads time-domain (noisy, background) pairs from HDF5 and computes STFT on-the-fly.

    The HDF5 file is opened lazily per worker process (h5py + fork safety).
    """

    def __init__(
        self,
        h5_path: str,
        scaler_mean: float,
        scaler_scale: float,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ):
        self.h5_path = h5_path
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self._file = None  # opened lazily per-worker

        with h5py.File(h5_path, "r") as f:
            self._len = f["noisy"].shape[0]

    def __len__(self) -> int:
        return self._len

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open()
        noisy = self._file["noisy"][idx].astype(np.float32)
        bg    = self._file["background"][idx].astype(np.float32)

        # StandardScaler: (x - mean) / scale
        noisy = (noisy - self.scaler_mean) / self.scaler_scale
        bg    = (bg    - self.scaler_mean) / self.scaler_scale

        noisy_stft = apply_stft(
            noisy[np.newaxis], self.n_fft, self.hop_length, self.win_length, self.window
        ).squeeze(0)  # (2, F, T)
        bg_stft = apply_stft(
            bg[np.newaxis], self.n_fft, self.hop_length, self.win_length, self.window
        ).squeeze(0)  # (2, F, T)

        return noisy_stft, bg_stft

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


# ── Training loop ─────────────────────────────────────────────────────────────

def _save_losses(
    loss_dir: Path,
    start_epoch: int,
    end_epoch: int,
    train_losses,
    train_noise_losses,
    train_constraint_losses,
    val_losses,
    val_noise_losses,
    val_constraint_losses,
):
    np.save(str(loss_dir / f"train_losses_{start_epoch}_to_{end_epoch}.npy"),
            np.array(train_losses))
    np.save(str(loss_dir / f"train_noise_losses_{start_epoch}_to_{end_epoch}.npy"),
            np.array(train_noise_losses))
    np.save(str(loss_dir / f"train_constraint_losses_{start_epoch}_to_{end_epoch}.npy"),
            np.array(train_constraint_losses))
    np.save(str(loss_dir / f"val_losses_{start_epoch}_to_{end_epoch}.npy"),
            np.array(val_losses))
    np.save(str(loss_dir / f"val_noise_losses_{start_epoch}_to_{end_epoch}.npy"),
            np.array(val_noise_losses))
    np.save(str(loss_dir / f"val_constraint_losses_{start_epoch}_to_{end_epoch}.npy"),
            np.array(val_constraint_losses))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--backgrounds", nargs="+", required=True,
        metavar="PKL",
        help="Background pickle file(s) from get_clean_backgrounds.py.",
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Pretrained model checkpoint (.pth.tar) to fine-tune from.",
    )
    p.add_argument(
        "--label", default="real",
        help="Short label used in checkpoint/loss filenames (e.g. 'o3', 'o4').",
    )
    p.add_argument(
        "--hdf5-dir", type=Path, default=Path("data_4s_real/hdf5"),
        help="Directory to write generated time-domain HDF5 files.",
    )
    p.add_argument(
        "--in-memory", action="store_true",
        help="Skip HDF5 entirely: inject glitches into RAM, keep all data in memory. "
             "Simpler and faster to start, but requires enough RAM to hold the full "
             "dataset (time-domain only — STFT is always computed on-the-fly). "
             "Good default for O3. Use HDF5 mode for O4b-scale datasets (>50k samples).",
    )
    p.add_argument(
        "--no-regen", action="store_true",
        help="(HDF5 mode only) Skip HDF5 generation if train.h5 and val.h5 already "
             "exist in --hdf5-dir.",
    )
    p.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints_tl"),
        help="Root directory for saving checkpoints.",
    )
    p.add_argument(
        "--loss-dir", type=Path, default=Path("losses_tl"),
        help="Root directory for saving loss arrays.",
    )
    p.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of each source to use as validation (chronological tail).",
    )
    p.add_argument(
        "--max-per-source", type=int, default=None,
        help="Cap on 8s samples taken from each IFO/run in each pkl file. "
             "Useful for quick tests.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--early-stopping-patience", type=int, default=9)
    p.add_argument(
        "--dropout-p", type=float, default=0.0,
        help="Must match the dropout-p the pretrained checkpoint was trained with.",
    )
    p.add_argument(
        "--injection-batch-size", type=int, default=1000,
        help="Batch size for glitch injection + HDF5 writing.",
    )
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{torch.cuda.device_count() - 1}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    # ── Directories ───────────────────────────────────────────────────────────
    model_name = "DeepExtractor_257"
    ckpt_dir = args.checkpoint_dir / f"{model_name}_checkpoints"
    loss_dir = args.loss_dir / f"{model_name}_{args.label}_tl_losses"
    for d in [ckpt_dir, loss_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load and split backgrounds ─────────────────────────────────────────────
    ifo_data = load_backgrounds(args.backgrounds, max_per_source=args.max_per_source)

    train_parts = []
    val_parts = []
    for ifo, samples_8s in ifo_data.items():
        train_8s, val_8s = chronological_split(samples_8s, val_frac=args.val_frac)
        train_4s = split_to_4s(train_8s)
        val_4s   = split_to_4s(val_8s)
        logger.info(f"{ifo}  train: {len(train_4s)}  val: {len(val_4s)}  (4s samples)")
        train_parts.append(train_4s)
        val_parts.append(val_4s)

    # Combine across detectors then shuffle train
    train_combined = np.concatenate(train_parts, axis=0)
    val_combined   = np.concatenate(val_parts,   axis=0)
    rng = np.random.default_rng(42)
    train_combined = train_combined[rng.permutation(len(train_combined))]
    logger.info(f"Combined train: {len(train_combined)}  val: {len(val_combined)}  (4s)")

    # ── Injection + dataset construction ──────────────────────────────────────
    dataset_kwargs = dict(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
    )

    if args.in_memory:
        n_samples = len(train_combined) + len(val_combined)
        est_gb = n_samples * LEN_4S * 4 * 2 / 1e9
        logger.info(f"In-memory mode: estimated peak RAM for time-domain arrays ≈ {est_gb:.1f} GB")

        train_noisy, train_bg, val_noisy, val_bg, scaler = inject_in_memory(
            train_combined, val_combined, args.injection_batch_size,
        )
        scaler_mean  = float(scaler.mean_[0])
        scaler_scale = float(scaler.scale_[0])
        logger.info(f"Scaler: mean={scaler_mean:.6f}  scale={scaler_scale:.6f}")

        train_ds = RealNoiseInMemoryDataset(
            train_noisy, train_bg, scaler_mean, scaler_scale, **dataset_kwargs
        )
        val_ds = RealNoiseInMemoryDataset(
            val_noisy, val_bg, scaler_mean, scaler_scale, **dataset_kwargs
        )
    else:
        train_h5    = args.hdf5_dir / "train.h5"
        val_h5      = args.hdf5_dir / "val.h5"
        scaler_path = args.hdf5_dir / "scaler_real.pkl"

        if args.no_regen and train_h5.exists() and val_h5.exists() and scaler_path.exists():
            logger.info("--no-regen: loading existing HDF5 and scaler")
            with open(scaler_path, "rb") as fh:
                scaler = pickle.load(fh)
        else:
            scaler = inject_and_write_hdf5(
                train_combined, val_combined, args.hdf5_dir, args.injection_batch_size,
            )

        scaler_mean  = float(scaler.mean_[0])
        scaler_scale = float(scaler.scale_[0])
        logger.info(f"Scaler: mean={scaler_mean:.6f}  scale={scaler_scale:.6f}")

        train_ds = RealNoiseHDF5Dataset(
            str(train_h5), scaler_mean, scaler_scale, **dataset_kwargs
        )
        val_ds = RealNoiseHDF5Dataset(
            str(val_h5), scaler_mean, scaler_scale, **dataset_kwargs
        )
    logger.info(f"Dataset sizes — train: {len(train_ds)}  val: {len(val_ds)}")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNET2D(in_channels=2, out_channels=2, dropout_p=args.dropout_p).to(device)
    logger.info(f"Loading pretrained checkpoint: {args.checkpoint}")
    ckpt = torch.load(str(args.checkpoint), map_location=device)
    load_checkpoint(ckpt, model)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    loss_fn   = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)
    amp_scaler = (
        torch.amp.GradScaler("cuda") if device.startswith("cuda")
        else torch.amp.GradScaler("cpu")
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    train_losses, train_noise_losses, train_constraint_losses = [], [], []
    val_losses,   val_noise_losses,   val_constraint_losses   = [], [], []
    best_val_loss = float("inf")
    early_counter = 0
    start_epoch   = 0
    ckpt_filename = ckpt_dir / f"checkpoint_best_{args.label}_tl.pth.tar"

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        tr_loss, tr_noise, tr_constr = train_fn(
            train_loader, model, model_name, optimizer, loss_fn, amp_scaler, device
        )
        train_losses.append(tr_loss)
        train_noise_losses.append(tr_noise)
        train_constraint_losses.append(tr_constr)

        v_loss, v_noise, v_constr = check_accuracy(
            val_loader, model, model_name, device=device
        )
        val_losses.append(v_loss)
        val_noise_losses.append(v_noise)
        val_constraint_losses.append(v_constr)

        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  train={tr_loss:.6f}  val={v_loss:.6f}  lr={current_lr:.2e}"
        )

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            early_counter = 0
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict(),
                    "epoch":      start_epoch + epoch,
                },
                str(ckpt_filename),
            )
            logger.info(f"  Val loss improved → checkpoint saved")
        else:
            early_counter += 1
            logger.info(
                f"  No improvement ({early_counter}/{args.early_stopping_patience})"
            )

        if early_counter >= args.early_stopping_patience:
            logger.info(f"Early stopping after epoch {epoch + 1}.")
            break

        if (epoch + 1) % 10 == 0:
            _save_losses(
                loss_dir, start_epoch, start_epoch + epoch,
                train_losses, train_noise_losses, train_constraint_losses,
                val_losses, val_noise_losses, val_constraint_losses,
            )

    _save_losses(
        loss_dir, start_epoch, start_epoch + epoch,
        train_losses, train_noise_losses, train_constraint_losses,
        val_losses, val_noise_losses, val_constraint_losses,
    )
    logger.info(f"Done. Best val loss: {best_val_loss:.6f}")
    logger.info(f"Checkpoint: {ckpt_filename}")


if __name__ == "__main__":
    main()
