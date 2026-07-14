"""
Transfer-learn a 4s DeepExtractor checkpoint on real detector backgrounds.

Mirrors the three-phase simulated training pipeline:

  Phase 1 — Prepare  : pkl → inject glitches → fit scaler → scaled time-domain HDF5
  Phase 2 — Specgen  : scaled HDF5 → STFT HDF5 (one file per key, same format as
                        deepextractor-specgen --format hdf5)
  Phase 3 — Train    : fine-tune from pretrained checkpoint using HDF5ReconstructionDataset
                        (fast sequential reads; no STFT computed during training)

Already-completed phases are detected automatically and skipped (idempotent).
Override with --redo-prepare or --redo-specgen to force regeneration.

Pkl format expected:  {run: {ifo: {'samples': ndarray(N, 32768), 'gps_starts': ndarray(N,)}}}

Example — O3 TL from bilby-pretrained checkpoint::

    python scripts/transfer_learn_real_noise.py \\
        --backgrounds backgrounds_O3a.pkl backgrounds_O3b.pkl \\
        --checkpoint checkpoints_4s/DeepExtractor_257_checkpoints/checkpoint_best_bilby_noise_base.pth.tar \\
        --label o3 \\
        --td-dir  data_4s_real/o3_td \\
        --spec-dir data_4s_real/o3_stft \\
        --checkpoint-dir checkpoints_4s_o3 \\
        --loss-dir losses_4s_o3 \\
        --epochs 50 --lr 1e-5

Then O4 TL from the O3 checkpoint::

    python scripts/transfer_learn_real_noise.py \\
        --backgrounds backgrounds_O4a.pkl backgrounds_O4b.pkl \\
        --checkpoint checkpoints_4s_o3/DeepExtractor_257_checkpoints/checkpoint_best_o3_tl.pth.tar \\
        --label o4 \\
        --td-dir  data_4s_real/o4_td \\
        --spec-dir data_4s_real/o4_stft \\
        --checkpoint-dir checkpoints_4s_o4 \\
        --loss-dir losses_4s_o4 \\
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepextractor.data.datasets import HDF5ReconstructionDataset
from torch.utils.data import TensorDataset
from deepextractor.generation.generate_timeseries import generate_synthetic_data
from deepextractor.models.architectures import UNET2D
from deepextractor.training.train_fn import train_fn
from deepextractor.utils.checkpoints import load_checkpoint, save_checkpoint
from deepextractor.utils.io import check_accuracy
from deepextractor.utils.stft import apply_stft

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 4096
LEN_8S = 8 * SAMPLE_RATE   # 32768 — raw pkl sample length
LEN_4S = 4 * SAMPLE_RATE   # 16384 — model input length
N_FFT, HOP_LENGTH, WIN_LENGTH = 512, 32, 64

IFOS = ["H1", "L1"]

# Keys written to the scaled time-domain HDF5 — must match what specgen expects.
KEYS = ["noisy_glitch_train", "background_train", "noisy_glitch_val", "background_val"]


# ── Phase 1 helpers ───────────────────────────────────────────────────────────

def load_backgrounds(
    pkl_paths: list[str], max_per_source: int | None = None
) -> dict[str, np.ndarray]:
    """Load pkl files and return per-IFO chronologically ordered 8s arrays."""
    ifo_arrays: dict[str, list[np.ndarray]] = {ifo: [] for ifo in IFOS}
    n_sources = 0
    for pkl_path in pkl_paths:
        logger.info(f"Loading {pkl_path} ...")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        for run, run_data in data.items():
            for ifo in IFOS:
                if ifo not in run_data:
                    logger.warning(f"  {run} has no {ifo} — skipping")
                    continue
                samples = run_data[ifo]["samples"]
                if max_per_source is not None:
                    samples = samples[:max_per_source]
                ifo_arrays[ifo].append(samples)
                logger.info(
                    f"  {run} {ifo}: {len(samples)} × 8s  [{samples.nbytes / 1e9:.2f} GB]"
                )
                n_sources += 1
    if n_sources == 0:
        raise RuntimeError(f"No samples found in {pkl_paths}")
    result = {}
    for ifo in IFOS:
        if ifo_arrays[ifo]:
            result[ifo] = np.concatenate(ifo_arrays[ifo], axis=0)
            logger.info(f"{ifo} total: {len(result[ifo])} × 8s samples")
    return result


def chronological_split(arr: np.ndarray, val_frac: float = 0.1):
    """First 90% → train, last 10% → val, split on 8s segments before halving."""
    n_val = max(1, int(np.round(len(arr) * val_frac)))
    return arr[: len(arr) - n_val], arr[len(arr) - n_val :]


def split_to_4s(arr_8s: np.ndarray) -> np.ndarray:
    """(N, 32768) → (2N, 16384) by splitting each 8s segment into two 4s halves."""
    n = arr_8s.shape[0]
    return arr_8s.reshape(n, 2, LEN_4S).reshape(n * 2, LEN_4S)


def phase1_prepare(
    pkl_paths: list[str],
    td_dir: Path,
    val_frac: float = 0.1,
    max_per_source: int | None = None,
    injection_batch: int = 1000,
) -> None:
    """
    Inject glitches into real backgrounds, fit a StandardScaler, and write
    a scaled time-domain HDF5 with keys matching the simulated pipeline format.

    Outputs
    -------
    td_dir/strain_data_scaled.h5  — keys: noisy_glitch_train, background_train,
                                          noisy_glitch_val, background_val
    td_dir/scaler_real.pkl        — fitted StandardScaler
    """
    td_dir.mkdir(parents=True, exist_ok=True)
    raw_h5_path    = td_dir / "strain_data_raw.h5"
    scaled_h5_path = td_dir / "strain_data_scaled.h5"
    scaler_path    = td_dir / "scaler_real.pkl"

    # ── 1a. Load, split, combine ──────────────────────────────────────────────
    ifo_data = load_backgrounds(pkl_paths, max_per_source=max_per_source)

    train_parts, val_parts = [], []
    for ifo, samples_8s in ifo_data.items():
        tr_8s, va_8s = chronological_split(samples_8s, val_frac)
        train_parts.append(split_to_4s(tr_8s))
        val_parts.append(split_to_4s(va_8s))
        logger.info(f"{ifo}  train 4s: {len(train_parts[-1])}  val 4s: {len(val_parts[-1])}")

    rng = np.random.default_rng(42)
    train_4s = np.concatenate(train_parts, axis=0)
    val_4s   = np.concatenate(val_parts,   axis=0)
    train_4s = train_4s[rng.permutation(len(train_4s))]
    logger.info(f"Combined — train: {len(train_4s)}  val: {len(val_4s)}  (4s samples)")

    # ── 1b. Inject glitches → raw HDF5 ───────────────────────────────────────
    logger.info(f"Phase 1a: injecting glitches → {raw_h5_path}")
    splits = [("train", train_4s), ("val", val_4s)]
    with h5py.File(raw_h5_path, "w") as f:
        for split_name, noise_4s in splits:
            n_total = len(noise_4s)
            ng_ds = f.create_dataset(
                f"noisy_glitch_{split_name}", shape=(0, LEN_4S),
                maxshape=(None, LEN_4S), dtype=np.float32,
                chunks=(min(injection_batch, n_total), LEN_4S),
            )
            bg_ds = f.create_dataset(
                f"background_{split_name}", shape=(0, LEN_4S),
                maxshape=(None, LEN_4S), dtype=np.float32,
                chunks=(min(injection_batch, n_total), LEN_4S),
            )
            wptr = 0
            for start in tqdm(range(0, n_total, injection_batch),
                              desc=f"Injecting ({split_name})"):
                end = min(start + injection_batch, n_total)
                batch = noise_4s[start:end].copy().astype(np.float32)
                noisy, bg = generate_synthetic_data(
                    batch, bilby_noise=False, phase=split_name, show_progress=False,
                )
                n_written = noisy.shape[0]
                ng_ds.resize(wptr + n_written, axis=0)
                bg_ds.resize(wptr + n_written, axis=0)
                ng_ds[wptr : wptr + n_written] = noisy.astype(np.float32)
                bg_ds[wptr : wptr + n_written] = bg.astype(np.float32)
                wptr += n_written

    # ── 1c. Fit scaler on noisy_glitch_train ─────────────────────────────────
    logger.info("Phase 1b: fitting StandardScaler ...")
    scaler = StandardScaler()
    with h5py.File(raw_h5_path, "r") as f:
        ds = f["noisy_glitch_train"]
        for start in tqdm(range(0, ds.shape[0], injection_batch), desc="Fitting scaler"):
            end = min(start + injection_batch, ds.shape[0])
            scaler.partial_fit(ds[start:end].reshape(-1, 1))
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    logger.info(
        f"Scaler: mean={scaler.mean_[0]:.6f}  scale={scaler.scale_[0]:.6f}  → {scaler_path}"
    )

    # ── 1d. Scale → scaled HDF5 ───────────────────────────────────────────────
    logger.info(f"Phase 1c: scaling → {scaled_h5_path}")
    with h5py.File(raw_h5_path, "r") as fin, h5py.File(scaled_h5_path, "w") as fout:
        for key in tqdm(fin.keys(), desc="Scaling keys"):
            shape = fin[key].shape
            chunk_n = min(injection_batch, shape[0])
            out_ds = fout.create_dataset(
                key, shape=shape, dtype=np.float32, chunks=(chunk_n, LEN_4S)
            )
            for start in range(0, shape[0], injection_batch):
                end = min(start + injection_batch, shape[0])
                chunk = fin[key][start:end]
                out_ds[start:end] = scaler.transform(
                    chunk.reshape(-1, 1)
                ).reshape(chunk.shape).astype(np.float32)

    logger.info(f"Phase 1 complete. Raw HDF5 can be removed: {raw_h5_path}")


# ── Phase 2 helpers ───────────────────────────────────────────────────────────

def phase2_specgen(
    td_dir: Path,
    spec_dir: Path,
    chunk_size: int = 2000,
) -> None:
    """
    Compute STFTs from the scaled time-domain HDF5 and write one file per key.

    Mirrors deepextractor-specgen --format hdf5:
      spec_dir/noisy_glitch_train.h5  (dataset key "data", shape (N, 2, F, T))
      spec_dir/background_train.h5
      spec_dir/noisy_glitch_val.h5
      spec_dir/background_val.h5
    """
    scaled_h5 = td_dir / "strain_data_scaled.h5"
    spec_dir.mkdir(parents=True, exist_ok=True)
    window = torch.hann_window(WIN_LENGTH)

    with h5py.File(scaled_h5, "r") as fin:
        for key in KEYS:
            out_path = spec_dir / f"{key}.h5"
            in_ds = fin[key]
            n = in_ds.shape[0]

            if out_path.exists():
                try:
                    with h5py.File(out_path, "r") as fcheck:
                        if "data" in fcheck and fcheck["data"].shape[0] == n:
                            logger.info(f"Skipping {key} — already complete ({n} samples)")
                            continue
                except Exception:
                    pass
                logger.info(f"Re-generating {key} — incomplete or corrupt")

            # Probe shape from a single dummy sample
            probe = apply_stft(
                np.zeros((1, LEN_4S), dtype=np.float32),
                N_FFT, HOP_LENGTH, WIN_LENGTH, window,
            )
            out_shape = (n, *probe.shape[1:])
            per_sample_bytes = int(np.prod(probe.shape[1:])) * 4
            max_chunk = max(1, (2 * 1024 ** 3) // per_sample_bytes)
            chunk_n = min(chunk_size, n, max_chunk)

            logger.info(f"STFT {key} → {out_path}  shape={out_shape}")
            with h5py.File(out_path, "w") as fout:
                out_ds = fout.create_dataset(
                    "data", shape=out_shape, dtype=np.float32,
                    chunks=(chunk_n, *probe.shape[1:]),
                )
                for start in tqdm(range(0, n, chunk_size), desc=f"STFT {key}"):
                    end = min(start + chunk_size, n)
                    stft = apply_stft(
                        in_ds[start:end], N_FFT, HOP_LENGTH, WIN_LENGTH, window,
                    )
                    out_ds[start:end] = stft.numpy().astype(np.float32)


# ── Phase 3 helpers ───────────────────────────────────────────────────────────

def _save_losses(loss_dir, start_epoch, end_epoch,
                 train_losses, train_noise_losses, train_constraint_losses,
                 val_losses, val_noise_losses, val_constraint_losses):
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


def _load_stft_to_tensor(h5_path: Path) -> torch.Tensor:
    """Read the full HDF5 'data' dataset into a float32 CPU tensor."""
    with h5py.File(h5_path, "r") as f:
        arr = f["data"][:]
    gb = arr.nbytes / 1e9
    logger.info(f"  Loaded {h5_path.name}: {arr.shape}  [{gb:.1f} GB]")
    return torch.tensor(arr, dtype=torch.float32)


def phase3_train(
    spec_dir: Path,
    pretrained_checkpoint: Path,
    ckpt_dir: Path,
    loss_dir: Path,
    label: str,
    device: str,
    epochs: int = 50,
    lr: float = 1e-5,
    batch_size: int = 32,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    early_stopping_patience: int = 9,
    dropout_p: float = 0.0,
    in_memory: bool = False,
) -> None:
    model_name = "DeepExtractor_257"

    if in_memory:
        logger.info("Loading all STFT data into memory ...")
        train_ds = TensorDataset(
            _load_stft_to_tensor(spec_dir / "noisy_glitch_train.h5"),
            _load_stft_to_tensor(spec_dir / "background_train.h5"),
        )
        val_ds = TensorDataset(
            _load_stft_to_tensor(spec_dir / "noisy_glitch_val.h5"),
            _load_stft_to_tensor(spec_dir / "background_val.h5"),
        )
        # Data is already in RAM — workers only add IPC overhead
        num_workers = 0
    else:
        train_ds = HDF5ReconstructionDataset(
            input_h5=str(spec_dir / "noisy_glitch_train.h5"),
            target_h5=str(spec_dir / "background_train.h5"),
        )
        val_ds = HDF5ReconstructionDataset(
            input_h5=str(spec_dir / "noisy_glitch_val.h5"),
            target_h5=str(spec_dir / "background_val.h5"),
        )
    logger.info(f"Dataset — train: {len(train_ds)}  val: {len(val_ds)}")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    model = UNET2D(in_channels=2, out_channels=2, dropout_p=dropout_p).to(device)
    logger.info(f"Loading pretrained checkpoint: {pretrained_checkpoint}")
    ckpt = torch.load(str(pretrained_checkpoint), map_location=device)
    load_checkpoint(ckpt, model)

    loss_fn   = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=4)
    amp_scaler = (
        torch.amp.GradScaler("cuda") if device.startswith("cuda")
        else torch.amp.GradScaler("cpu")
    )

    ckpt_path = ckpt_dir / f"checkpoint_best_{label}_tl.pth.tar"
    train_losses, train_noise_losses, train_constraint_losses = [], [], []
    val_losses,   val_noise_losses,   val_constraint_losses   = [], [], []
    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
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
        logger.info(
            f"  train={tr_loss:.6f}  val={v_loss:.6f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if v_loss < best_val:
            best_val = v_loss
            no_improve = 0
            save_checkpoint(
                {"state_dict": model.state_dict(),
                 "optimizer":  optimizer.state_dict(),
                 "scheduler":  scheduler.state_dict(),
                 "epoch":      epoch},
                str(ckpt_path),
            )
            logger.info(f"  Val improved → {ckpt_path}")
        else:
            no_improve += 1
            logger.info(f"  No improvement ({no_improve}/{early_stopping_patience})")
            if no_improve >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

        if (epoch + 1) % 10 == 0:
            _save_losses(loss_dir, 0, epoch,
                         train_losses, train_noise_losses, train_constraint_losses,
                         val_losses, val_noise_losses, val_constraint_losses)

    _save_losses(loss_dir, 0, epoch,
                 train_losses, train_noise_losses, train_constraint_losses,
                 val_losses, val_noise_losses, val_constraint_losses)
    logger.info(f"Done. Best val loss: {best_val:.6f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data sources
    p.add_argument("--backgrounds", nargs="+", required=True, metavar="PKL",
                   help="Background pkl file(s) from get_clean_backgrounds.py.")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Pretrained checkpoint to fine-tune from.")
    p.add_argument("--label", default="real",
                   help="Short label for checkpoint/loss filenames (e.g. 'o3', 'o4').")
    # Directories
    p.add_argument("--td-dir",         type=Path, default=Path("data_4s_real/td"),
                   help="Directory for time-domain HDF5 files (phase 1 output).")
    p.add_argument("--spec-dir",       type=Path, default=Path("data_4s_real/stft"),
                   help="Directory for STFT HDF5 files (phase 2 output).")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_tl"))
    p.add_argument("--loss-dir",       type=Path, default=Path("losses_tl"))
    # Phase control
    p.add_argument("--redo-prepare", action="store_true",
                   help="Force re-run of phase 1 even if outputs exist.")
    p.add_argument("--redo-specgen", action="store_true",
                   help="Force re-run of phase 2 even if outputs exist.")
    # Data options
    p.add_argument("--val-frac",          type=float, default=0.1)
    p.add_argument("--max-per-source",    type=int,   default=None)
    p.add_argument("--injection-batch",   type=int,   default=1000)
    p.add_argument("--specgen-chunk",     type=int,   default=2000)
    # Training options
    p.add_argument("--epochs",                    type=int,   default=50)
    p.add_argument("--batch-size",                type=int,   default=32)
    p.add_argument("--lr",                        type=float, default=1e-5)
    p.add_argument("--num-workers",               type=int,   default=4)
    p.add_argument("--prefetch-factor",           type=int,   default=2)
    p.add_argument("--early-stopping-patience",   type=int,   default=9)
    p.add_argument("--dropout-p",  type=float, default=0.0)
    p.add_argument("--in-memory", action="store_true",
                   help="Load all STFT data into RAM before training. Fastest option "
                        "when the node has enough memory (~1 MB per sample). "
                        "num-workers is forced to 0 (no benefit with in-memory data).")
    p.add_argument("--device",    default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = f"cuda:{torch.cuda.device_count() - 1}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    model_name = "DeepExtractor_257"
    ckpt_dir = args.checkpoint_dir / f"{model_name}_checkpoints"
    loss_dir = args.loss_dir / f"{model_name}_{args.label}_tl_losses"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: inject → time-domain HDF5 ───────────────────────────────────
    scaled_h5 = args.td_dir / "strain_data_scaled.h5"
    if not args.redo_prepare and scaled_h5.exists():
        logger.info(f"Phase 1: skipping — {scaled_h5} exists (use --redo-prepare to force)")
    else:
        logger.info("Phase 1: preparing time-domain HDF5 ...")
        phase1_prepare(
            pkl_paths=args.backgrounds,
            td_dir=args.td_dir,
            val_frac=args.val_frac,
            max_per_source=args.max_per_source,
            injection_batch=args.injection_batch,
        )

    # ── Phase 2: STFT → spectrogram HDF5 ─────────────────────────────────────
    all_spec_done = all((args.spec_dir / f"{k}.h5").exists() for k in KEYS)
    if not args.redo_specgen and all_spec_done:
        logger.info(f"Phase 2: skipping — all STFT files exist in {args.spec_dir} "
                    f"(use --redo-specgen to force)")
    else:
        logger.info("Phase 2: computing STFTs ...")
        phase2_specgen(
            td_dir=args.td_dir,
            spec_dir=args.spec_dir,
            chunk_size=args.specgen_chunk,
        )

    # ── Phase 3: train ────────────────────────────────────────────────────────
    logger.info("Phase 3: training ...")
    phase3_train(
        spec_dir=args.spec_dir,
        pretrained_checkpoint=args.checkpoint,
        ckpt_dir=ckpt_dir,
        loss_dir=loss_dir,
        label=args.label,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        early_stopping_patience=args.early_stopping_patience,
        dropout_p=args.dropout_p,
        in_memory=args.in_memory,
    )


if __name__ == "__main__":
    main()
