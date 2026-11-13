"""Simulated evaluation script for the two-detector time-domain separator.

Generates synthetic two-detector (H1 + L1) data by injecting glitches into
independent Gaussian noise realisations, runs the trained UNET1D separator,
and reports match-filter metrics per glitch class.

Designed to produce paper-quality results from the checkpoint trained on
Snellius (UNET1D in_channels=2, out_channels=4, 6 encoder layers).

Example:
    python scripts/evaluate_td.py \\
        --checkpoint checkpoints/UNET1D_4_channel_6_layers/checkpoint_best_bilby_noise_hdf5_transfer_learn.pth.tar \\
        --scaler data/standard_scaler.pkl \\
        --out evaluation/td_results

    # Without scaler (fits on test data — qualitative only):
    python scripts/evaluate_td.py \\
        --checkpoint checkpoints/UNET1D_4_channel_6_layers/checkpoint_best_bilby_noise_hdf5_transfer_learn.pth.tar \\
        --out evaluation/td_results
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pycbc.filter.matchedfilter import match
from pycbc.types import TimeSeries as PyCBCTimeSeries

from deepextractor.generation.glitch_functions import (
    generate_chirp,
    generate_gaussian_pulse,
    generate_sine,
    generate_sine_gaussian,
)
from deepextractor.models import UNET1D
from deepextractor.utils.signal import generate_gaussian_noise, whitened_snr_scaling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (match training configuration)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 4096
T = 2.0
T_INJ = T / 2
LENGTH = int(T * SAMPLE_RATE)
T_MIN, T_MAX = 0.125, 2.0
SNR_MIN, SNR_MAX = 7.5, 100.0
NOISE_STD = 50.0
BATCH_SIZE = 32

GLITCH_TYPES = ["chirp", "sine", "sine_gaussian", "gaussian_pulse"]


# ---------------------------------------------------------------------------
# Glitch injection
# ---------------------------------------------------------------------------

def _generate_signal(glitch_type: str) -> np.ndarray:
    if glitch_type == "chirp":
        _, sig = generate_chirp(np.random.uniform(T_MIN, T_MAX))
    elif glitch_type == "sine":
        _, sig = generate_sine(np.random.uniform(T_MIN, T_MAX))
    elif glitch_type == "sine_gaussian":
        _, sig = generate_sine_gaussian(np.random.uniform(T_MIN, T_MAX))
    elif glitch_type == "gaussian_pulse":
        _, sig = generate_gaussian_pulse(np.random.uniform(T_MIN, T_MAX))
    else:
        raise ValueError(f"Unknown glitch type: {glitch_type}")
    return sig.squeeze()


def generate_two_detector_data(
    glitch_type_h1: str,
    glitch_type_l1: str,
    noise_samples_h1: np.ndarray,
    noise_samples_l1: np.ndarray,
) -> dict:
    """Inject independent glitches into H1 and L1 noise.

    Returns a dict with arrays of shape (N, LENGTH) for each component.
    """
    noisy_h1, noisy_l1 = [], []
    bg_h1,    bg_l1    = [], []
    sig_h1,   sig_l1   = [], []
    snrs = []

    for h1_noise, l1_noise in tqdm(
        zip(noise_samples_h1, noise_samples_l1),
        total=len(noise_samples_h1),
        desc=f"Generating {glitch_type_h1}/{glitch_type_l1}",
        leave=False,
    ):
        snr = np.random.uniform(SNR_MIN, SNR_MAX)

        # H1
        h1_glitch = _generate_signal(glitch_type_h1)
        if np.isnan(h1_glitch).any():
            continue
        h1_glitch = h1_glitch - h1_glitch.mean()
        h1_glitch = whitened_snr_scaling(h1_glitch, snr=snr)
        len_g = len(h1_glitch)
        id_start = int((T_INJ * SAMPLE_RATE / LENGTH) * LENGTH) - len_g // 2
        id_start = max(0, min(id_start, LENGTH - len_g))
        h1_injected = h1_noise.copy()
        h1_injected[id_start:id_start + len_g] += h1_glitch
        h1_signal = h1_injected - h1_noise

        # L1 — independent glitch, same SNR
        l1_glitch = _generate_signal(glitch_type_l1)
        if np.isnan(l1_glitch).any():
            continue
        l1_glitch = l1_glitch - l1_glitch.mean()
        l1_glitch = whitened_snr_scaling(l1_glitch, snr=snr)
        len_g = len(l1_glitch)
        id_start = int((T_INJ * SAMPLE_RATE / LENGTH) * LENGTH) - len_g // 2
        id_start = max(0, min(id_start, LENGTH - len_g))
        l1_injected = l1_noise.copy()
        l1_injected[id_start:id_start + len_g] += l1_glitch
        l1_signal = l1_injected - l1_noise

        noisy_h1.append(h1_injected)
        noisy_l1.append(l1_injected)
        bg_h1.append(h1_noise)
        bg_l1.append(l1_noise)
        sig_h1.append(h1_signal)
        sig_l1.append(l1_signal)
        snrs.append(snr)

    return {
        "noisy_h1": np.array(noisy_h1, dtype=np.float32),
        "noisy_l1": np.array(noisy_l1, dtype=np.float32),
        "bg_h1":    np.array(bg_h1,    dtype=np.float32),
        "bg_l1":    np.array(bg_l1,    dtype=np.float32),
        "sig_h1":   np.array(sig_h1,   dtype=np.float32),
        "sig_l1":   np.array(sig_l1,   dtype=np.float32),
        "snr":      np.array(snrs,     dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def build_scaler(scaler_path: str | None, data: dict) -> object:
    """Load or fit a StandardScaler.

    If no path is given, fits a global scaler on the test inputs themselves.
    This gives qualitatively meaningful results but is not the training scaler.
    """
    if scaler_path is not None:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Loaded scaler from %s", scaler_path)
        return scaler

    warnings.warn(
        "No scaler provided — fitting StandardScaler on the test inputs. "
        "Results are qualitatively meaningful but NOT directly comparable to "
        "training-scaler results. Retrieve standard_scaler.pkl from Snellius "
        "for quantitative evaluation.",
        UserWarning,
        stacklevel=2,
    )
    inputs = np.concatenate([data["noisy_h1"].ravel(), data["noisy_l1"].ravel()])
    scaler = StandardScaler().fit(inputs.reshape(-1, 1))
    return scaler


def scale_inputs(h1: np.ndarray, l1: np.ndarray, scaler) -> np.ndarray:
    """Scale H1 and L1 arrays and stack to (N, 2, T)."""
    h1_s = scaler.transform(h1.reshape(-1, 1)).reshape(h1.shape).astype(np.float32)
    l1_s = scaler.transform(l1.reshape(-1, 1)).reshape(l1.shape).astype(np.float32)
    return np.stack([h1_s, l1_s], axis=1)  # (N, 2, T)


# ---------------------------------------------------------------------------
# Match filter
# ---------------------------------------------------------------------------

def overlap_match(true: np.ndarray, pred: np.ndarray) -> float:
    """Return the match (overlap) between two 1-D time-domain arrays."""
    dt = 1.0 / SAMPLE_RATE
    try:
        t = PyCBCTimeSeries(true.astype(np.float64), delta_t=dt)
        p = PyCBCTimeSeries(pred.astype(np.float64), delta_t=dt)
        return float(match(t, p)[0])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Run model on (N, 2, T) array, return (N, 4, T) numpy array."""
    all_out = []
    for start in range(0, len(x), BATCH_SIZE):
        batch = torch.tensor(x[start:start + BATCH_SIZE]).to(device)
        with torch.no_grad():
            out = model(batch)
        all_out.append(out.cpu().numpy())
    return np.concatenate(all_out, axis=0)


# ---------------------------------------------------------------------------
# Per-class evaluation
# ---------------------------------------------------------------------------

def evaluate_class(
    data: dict,
    model: nn.Module,
    scaler,
    device: torch.device,
    label: str,
) -> dict:
    """Run inference and compute match metrics for one glitch-class dataset."""
    x = scale_inputs(data["noisy_h1"], data["noisy_l1"], scaler)
    preds = run_inference(model, x, device)  # (N, 4, T)

    # Output layout: [h1_bg, l1_bg, h1_sig, l1_sig]
    pred_h1_bg  = preds[:, 0, :]
    pred_l1_bg  = preds[:, 1, :]
    pred_h1_sig = preds[:, 2, :]
    pred_l1_sig = preds[:, 3, :]

    match_h1_sig, match_l1_sig = [], []
    match_h1_bg,  match_l1_bg  = [], []
    mse_h1_sig,   mse_l1_sig   = [], []

    for i in tqdm(range(len(preds)), desc=f"Metrics {label}", leave=False):
        match_h1_sig.append(overlap_match(data["sig_h1"][i], pred_h1_sig[i]))
        match_l1_sig.append(overlap_match(data["sig_l1"][i], pred_l1_sig[i]))
        match_h1_bg.append( overlap_match(data["bg_h1"][i],  pred_h1_bg[i]))
        match_l1_bg.append( overlap_match(data["bg_l1"][i],  pred_l1_bg[i]))
        mse_h1_sig.append(float(np.mean((data["sig_h1"][i] - pred_h1_sig[i])**2)))
        mse_l1_sig.append(float(np.mean((data["sig_l1"][i] - pred_l1_sig[i])**2)))

    return {
        "label":        label,
        "n":            len(preds),
        "snr":          data["snr"].tolist(),
        "match_h1_sig": match_h1_sig,
        "match_l1_sig": match_l1_sig,
        "match_h1_bg":  match_h1_bg,
        "match_l1_bg":  match_l1_bg,
        "mse_h1_sig":   mse_h1_sig,
        "mse_l1_sig":   mse_l1_sig,
        "pred_h1_sig":  pred_h1_sig.tolist(),
        "pred_l1_sig":  pred_l1_sig.tolist(),
        "pred_h1_bg":   pred_h1_bg.tolist(),
        "pred_l1_bg":   pred_l1_bg.tolist(),
    }


def print_summary(results: dict):
    header = f"{'Class':<25} {'N':>5}  {'Match H1 sig':>13}  {'Match L1 sig':>13}  {'Match H1 bg':>12}  {'Match L1 bg':>12}"
    logger.info("\n" + header)
    logger.info("-" * len(header))
    for label, r in results.items():
        def _fmt(vals):
            arr = np.array([v for v in vals if not np.isnan(v)])
            return f"{arr.mean():.3f} ± {arr.std():.3f}" if len(arr) else "  N/A"
        logger.info(
            f"{label:<25} {r['n']:>5}  {_fmt(r['match_h1_sig']):>13}  "
            f"{_fmt(r['match_l1_sig']):>13}  {_fmt(r['match_h1_bg']):>12}  "
            f"{_fmt(r['match_l1_bg']):>12}"
        )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(results: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- 1. Mismatch distributions per class (H1 and L1 side-by-side) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, ifo, key in zip(axes, ["H1", "L1"], ["match_h1_sig", "match_l1_sig"]):
        for i, (label, r) in enumerate(results.items()):
            mismatch = (1 - np.array([v for v in r[key] if not np.isnan(v)])) * 100
            ax.hist(mismatch, bins=30, alpha=0.6, label=label, color=colors[i % len(colors)])
        ax.set_xlabel("Mismatch (%)", fontsize=12)
        ax.set_ylabel("Count" if ifo == "H1" else "", fontsize=12)
        ax.set_title(f"{ifo} signal mismatch", fontsize=12)
        ax.legend(fontsize=9)
    fig.suptitle("Signal mismatch distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "mismatch_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mismatch_distributions.png")

    # --- 2. Match vs SNR scatter (H1 signal, one panel per class) ---
    ncols = min(len(labels), 4)
    nrows = (len(labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    for idx, (label, r) in enumerate(results.items()):
        ax = axes[idx // ncols][idx % ncols]
        snr = np.array(r["snr"])
        m_h1 = np.array(r["match_h1_sig"])
        m_l1 = np.array(r["match_l1_sig"])
        valid = ~np.isnan(m_h1) & ~np.isnan(m_l1)
        ax.scatter(snr[valid], m_h1[valid], s=8, alpha=0.5, label="H1", color=colors[0])
        ax.scatter(snr[valid], m_l1[valid], s=8, alpha=0.5, label="L1", color=colors[1])
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Injected SNR", fontsize=9)
        ax.set_ylabel("Match", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    # Hide unused axes
    for idx in range(len(labels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    fig.suptitle("Match vs injected SNR (signal reconstruction)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "match_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved match_vs_snr.png")

    # --- 3. Mean match bar chart (signal + background, H1 and L1) ---
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 4))
    for offset, key, ifo, component in zip(
        [-1.5, -0.5, 0.5, 1.5],
        ["match_h1_sig", "match_l1_sig", "match_h1_bg", "match_l1_bg"],
        ["H1", "L1", "H1", "L1"],
        ["signal", "signal", "background", "background"],
    ):
        means = []
        errs  = []
        for r in results.values():
            arr = np.array([v for v in r[key] if not np.isnan(v)])
            means.append(arr.mean())
            errs.append(arr.std() / np.sqrt(len(arr)))
        ax.bar(x + offset * width, means, width, yerr=errs, capsize=3,
               label=f"{ifo} {component}", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean match")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="k", lw=0.5, ls="--")
    ax.legend(fontsize=9, ncol=2)
    ax.set_title("Mean match per class — signal and background", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_match_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mean_match_bar.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate TD two-detector separator")
    p.add_argument("--checkpoint", required=True, help="Path to .pth.tar checkpoint")
    p.add_argument("--scaler",     default=None,  help="Path to pickled StandardScaler (optional)")
    p.add_argument("--out",        required=True, help="Output directory for results")
    p.add_argument("--n-samples",  type=int, default=256,
                   help="Number of test samples per glitch class (default 256)")
    p.add_argument("--features",   nargs="+", type=int,
                   default=[64, 128, 256, 512, 1024, 2048],
                   help="UNET1D feature sizes (must match checkpoint)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Device: %s", device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    model = UNET1D(in_channels=2, out_channels=4, features=args.features).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    logger.info("Loaded checkpoint: epoch %d, best val loss %.4e",
                ckpt.get("epoch", -1), ckpt.get("best_val", ckpt.get("scheduler", {}).get("best", float("nan"))))

    # --- Generate independent noise pools ---
    logger.info("Generating noise samples (N=%d per class × %d classes)...",
                args.n_samples, len(GLITCH_TYPES))
    noise_pool_h1 = generate_gaussian_noise(0, NOISE_STD, args.n_samples * len(GLITCH_TYPES), (LENGTH,))
    noise_pool_l1 = generate_gaussian_noise(0, NOISE_STD, args.n_samples * len(GLITCH_TYPES), (LENGTH,))

    # --- Generate per-class test sets ---
    all_data = {}
    for i, glitch_type in enumerate(GLITCH_TYPES):
        h1_noise = noise_pool_h1[i * args.n_samples:(i + 1) * args.n_samples]
        l1_noise = noise_pool_l1[i * args.n_samples:(i + 1) * args.n_samples]
        all_data[glitch_type] = generate_two_detector_data(
            glitch_type, glitch_type, h1_noise, l1_noise
        )

    # Build scaler (fit on all test inputs combined if not provided)
    combined_h1 = np.concatenate([d["noisy_h1"] for d in all_data.values()])
    combined_l1 = np.concatenate([d["noisy_l1"] for d in all_data.values()])
    scaler = build_scaler(args.scaler, {"noisy_h1": combined_h1, "noisy_l1": combined_l1})

    # --- Evaluate ---
    results = {}
    for glitch_type, data in all_data.items():
        logger.info("Evaluating: %s", glitch_type)
        results[glitch_type] = evaluate_class(data, model, scaler, device, glitch_type)

    print_summary(results)

    # --- Save ---
    out_pkl = out_dir / "td_evaluation_results.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({"args": vars(args), "results": results}, f)
    logger.info("Results saved to %s", out_pkl)

    # Also save a compact CSV summary
    import csv
    csv_path = out_dir / "td_evaluation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "n",
                         "mean_match_h1_sig", "std_match_h1_sig",
                         "mean_match_l1_sig", "std_match_l1_sig",
                         "mean_match_h1_bg",  "std_match_h1_bg",
                         "mean_match_l1_bg",  "std_match_l1_bg",
                         "mean_mse_h1_sig",   "mean_mse_l1_sig"])
        for label, r in results.items():
            def _ms(vals):
                arr = np.array([v for v in vals if not np.isnan(v)])
                return arr.mean(), arr.std()
            row = [label, r["n"]]
            for key in ["match_h1_sig", "match_l1_sig", "match_h1_bg", "match_l1_bg"]:
                m, s = _ms(r[key])
                row += [f"{m:.4f}", f"{s:.4f}"]
            row += [f"{np.mean(r['mse_h1_sig']):.4e}", f"{np.mean(r['mse_l1_sig']):.4e}"]
            writer.writerow(row)
    logger.info("Summary CSV saved to %s", csv_path)

    # --- Plots ---
    save_plots(results, out_dir)


if __name__ == "__main__":
    main()
