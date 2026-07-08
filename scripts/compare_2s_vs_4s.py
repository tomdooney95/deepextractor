"""
Compare 2-second and 4-second DeepExtractor models over the middle-2s window.

Both models reconstruct the background noise from a 4s window that contains
a glitch centered in the middle 2s.  The comparison is fair because:
  - the 2s model sees only the middle 2s of the noisy input.
  - the 4s model sees the full 4s window; its output is trimmed to the same
    middle 2s ([4096:12288] samples).

Glitch injection uses shift_int=0 (no jitter) so the glitch is always fully
inside the middle 2s window, ensuring neither model is penalised for missing
glitch energy outside its receptive window.

Usage (after copying checkpoint + scaler from CIT)::

    python scripts/compare_2s_vs_4s.py \\
        --checkpoint-2s pretrained/DeepExtractor_257/checkpoint_best_bilby_noise_base.pth.tar \\
        --checkpoint-4s /path/to/checkpoints_4s/DeepExtractor_257_checkpoints/checkpoint_best_bilby_noise_base.pth.tar \\
        --scaler-2s src/deepextractor/assets/scaler_bilby.pkl \\
        --scaler-4s /path/to/data_4s/bilby_noise_hdf5/scaler.pkl \\
        --num-samples 500 \\
        --output-dir evaluation/compare_2s_vs_4s/
"""

import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from deepextractor.generation.glitch_functions import (
    generate_chirp,
    generate_gaussian_pulse,
    generate_gengli_glitch,
    generate_sine,
    generate_sine_gaussian,
    ringdown,
)
from deepextractor.models.architectures import UNET2D
from deepextractor.utils.checkpoints import load_checkpoint
from deepextractor.utils.signal import whitened_snr_scaling
from deepextractor.utils.stft import apply_istft, apply_stft

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 4096
LEN_4S = 4 * SAMPLE_RATE   # 16384 samples
LEN_2S = 2 * SAMPLE_RATE   # 8192 samples
MID_START = LEN_4S // 4    # 4096  — start of middle 2s in 4s signal
MID_END = 3 * LEN_4S // 4  # 12288 — end of middle 2s

N_FFT, WIN_LENGTH, HOP_LENGTH = 512, 64, 32

SIGNAL_FN = {
    "chirp": generate_chirp,
    "sine": generate_sine,
    "sine_gaussian": generate_sine_gaussian,
    "gaussian_pulse": generate_gaussian_pulse,
    "ringdown": ringdown,
}
# GenGLI glitches don't take a duration argument
SIGNAL_FN_NODUR = {
    "gengli_H1": lambda: generate_gengli_glitch(ifo="H1"),
    "gengli_L1": lambda: generate_gengli_glitch(ifo="L1"),
}
SNR_MIN, SNR_MAX = 7.5, 100.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_model(checkpoint_path, device):
    model = UNET2D(in_channels=2, out_channels=2).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    load_checkpoint(ckpt, model)
    model.eval()
    return model


def _generate_test_sample():
    """Unit-variance noise + single centered glitch (no time jitter).

    Returns
    -------
    noisy_4s, noise_4s : np.ndarray, shape (LEN_4S,)
    true_gl_4s         : np.ndarray, shape (LEN_4S,)  — injected glitch, zero outside support
    s_type             : str
    """
    noise = np.random.randn(LEN_4S).astype(np.float32)

    all_types = list(SIGNAL_FN) + list(SIGNAL_FN_NODUR)
    s_type = random.choice(all_types)
    if s_type in SIGNAL_FN_NODUR:
        _, signal = SIGNAL_FN_NODUR[s_type]()
    else:
        duration = np.random.uniform(0.125, 2.0)
        _, signal = SIGNAL_FN[s_type](duration)
    signal = signal.squeeze()

    if np.isnan(signal).any() or len(signal) == 0:
        return None, None, None, None

    snr = np.random.uniform(SNR_MIN, SNR_MAX)
    glitch = signal - np.mean(signal)
    # bilby unit-variance convention: divide snr by sqrt(SAMPLE_RATE/2)
    glitch = whitened_snr_scaling(glitch, snr=snr / np.sqrt(SAMPLE_RATE / 2))

    id_start = LEN_4S // 2 - len(glitch) // 2  # exact centre, no jitter
    id_end = id_start + len(glitch)
    if id_end > LEN_4S:
        return None, None, None, None

    noisy = noise.copy()
    noisy[id_start:id_end] += glitch

    true_gl = np.zeros(LEN_4S, dtype=np.float32)
    true_gl[id_start:id_end] = glitch

    return noisy, noise, true_gl, s_type


def _run_model(model, scaler, noisy_ts, window, device):
    """Scale → STFT → model → iSTFT → inverse-scale.

    Parameters
    ----------
    noisy_ts : np.ndarray, shape (N,)  (1D, any length)

    Returns
    -------
    np.ndarray, shape (N,)  — reconstructed background in original units
    """
    scaled = scaler.transform(noisy_ts.reshape(-1, 1)).reshape(1, -1).astype(np.float32)
    stft_in = apply_stft(scaled, N_FFT, HOP_LENGTH, WIN_LENGTH, window).to(device)
    with torch.no_grad():
        pred = model(stft_in)
    td = apply_istft(pred, N_FFT, HOP_LENGTH, WIN_LENGTH, window)
    td_np = td.cpu().numpy().squeeze()   # shape (N,)
    return scaler.inverse_transform(td_np.reshape(-1, 1)).reshape(-1)


def _overlap(a, b):
    """Normalised inner product in [−1, 1] as a match proxy."""
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 1e-30 else 0.0


def _add_specgram(ax, signal, title, sr=SAMPLE_RATE, fmax=512):
    """Plot a spectrogram on *ax* using matplotlib's built-in specgram."""
    ax.specgram(signal, Fs=sr, NFFT=256, noverlap=224, cmap="viridis", scale="dB")
    ax.set_ylim(0, fmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=9)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare 2s vs 4s DeepExtractor over the middle-2s window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-2s", required=True,
                        help="2s model checkpoint (.pth.tar).")
    parser.add_argument("--checkpoint-4s", required=True,
                        help="4s model checkpoint (.pth.tar).")
    parser.add_argument("--scaler-2s", required=True,
                        help="2s StandardScaler (.pkl).")
    parser.add_argument("--scaler-4s", required=True,
                        help="4s StandardScaler (.pkl).")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--output-dir", default="evaluation/compare_2s_vs_4s/")
    parser.add_argument("--plot-n", type=int, default=6,
                        help="Number of example time-domain plots to save.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    window = torch.hann_window(WIN_LENGTH)

    with open(args.scaler_2s, "rb") as f:
        scaler_2s = pickle.load(f)
    with open(args.scaler_4s, "rb") as f:
        scaler_4s = pickle.load(f)

    model_2s = _load_model(args.checkpoint_2s, device)
    model_4s = _load_model(args.checkpoint_4s, device)

    results = {
        "match_bg_2s": [], "match_bg_4s": [],
        "match_gl_2s": [], "match_gl_4s": [],
        "signal_type": [],
    }
    examples = []
    n_tried = 0

    with tqdm(total=args.num_samples, desc="Evaluating") as pbar:
        while len(results["match_bg_2s"]) < args.num_samples:
            n_tried += 1
            if n_tried > args.num_samples * 3:
                print("WARNING: too many failed sample generations — stopping early.")
                break

            noisy_4s, noise_4s, true_gl_4s, s_type = _generate_test_sample()
            if noisy_4s is None:
                continue

            true_bg_mid = noise_4s[MID_START:MID_END]
            true_gl_mid = true_gl_4s[MID_START:MID_END]

            # 2s model — middle 2s input only
            noisy_mid = noisy_4s[MID_START:MID_END]
            recon_bg_2s = _run_model(model_2s, scaler_2s, noisy_mid, window, device)
            recon_gl_2s = noisy_mid - recon_bg_2s

            # 4s model — full 4s input, trim output to middle 2s
            recon_bg_4s_full = _run_model(model_4s, scaler_4s, noisy_4s, window, device)
            recon_bg_4s = recon_bg_4s_full[MID_START:MID_END]
            recon_gl_4s = noisy_4s[MID_START:MID_END] - recon_bg_4s

            results["match_bg_2s"].append(_overlap(true_bg_mid, recon_bg_2s))
            results["match_bg_4s"].append(_overlap(true_bg_mid, recon_bg_4s))
            results["match_gl_2s"].append(_overlap(true_gl_mid, recon_gl_2s))
            results["match_gl_4s"].append(_overlap(true_gl_mid, recon_gl_4s))
            results["signal_type"].append(s_type)

            if len(examples) < args.plot_n:
                examples.append({
                    "s_type": s_type,
                    "noisy_mid": noisy_mid,
                    "true_bg_mid": true_bg_mid,
                    "true_gl_mid": true_gl_mid,
                    "recon_bg_2s": recon_bg_2s,
                    "recon_bg_4s": recon_bg_4s,
                    "recon_gl_2s": recon_gl_2s,
                    "recon_gl_4s": recon_gl_4s,
                })

            pbar.update(1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Results (middle-2s window) ──────────────────────────────")
    for key in ("match_bg_2s", "match_bg_4s", "match_gl_2s", "match_gl_4s"):
        arr = np.array(results[key])
        print(f"  {key:20s}  mean={arr.mean():.4f}  median={np.median(arr):.4f}  std={arr.std():.4f}")

    # Per signal-type breakdown
    print("\n── Background match by signal type ─────────────────────────")
    signal_types = sorted(set(results["signal_type"]))
    for stype in signal_types:
        idx = [i for i, s in enumerate(results["signal_type"]) if s == stype]
        m2 = np.mean([results["match_bg_2s"][i] for i in idx])
        m4 = np.mean([results["match_bg_4s"][i] for i in idx])
        print(f"  {stype:20s}  2s={m2:.4f}  4s={m4:.4f}  Δ={m4-m2:+.4f}  (n={len(idx)})")

    # ── Save results ──────────────────────────────────────────────────────────
    out_pkl = os.path.join(args.output_dir, "comparison_results.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {out_pkl}")

    # ── Summary plot (match distributions) ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key_2s, key_4s, title in zip(
        axes,
        ("match_bg_2s", "match_gl_2s"),
        ("match_bg_4s", "match_gl_4s"),
        ("Background match", "Glitch residual match"),
    ):
        ax.hist(results[key_2s], bins=40, alpha=0.6, label="2s model", color="steelblue")
        ax.hist(results[key_4s], bins=40, alpha=0.6, label="4s model", color="tomato")
        ax.set_xlabel("Overlap")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
    plt.suptitle("2s vs 4s model — middle-2s window comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "match_distributions.png"), dpi=150)
    plt.close()

    # ── Per-sample example plots ──────────────────────────────────────────────
    t2 = np.linspace(0, 2, LEN_2S)
    for idx, ex in enumerate(examples):
        m_bg_2s = _overlap(ex["true_bg_mid"], ex["recon_bg_2s"])
        m_bg_4s = _overlap(ex["true_bg_mid"], ex["recon_bg_4s"])
        m_gl_2s = _overlap(ex["true_gl_mid"], ex["recon_gl_2s"])
        m_gl_4s = _overlap(ex["true_gl_mid"], ex["recon_gl_4s"])

        # Time-domain: 2×2 (background top row, glitch residual bottom row)
        fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex=True)
        fig.suptitle(f"Example {idx} — {ex['s_type']} | time domain (middle 2s)")

        axes[0, 0].plot(t2, ex["recon_bg_2s"], lw=0.9, alpha=0.85, label="2s model")
        axes[0, 0].plot(t2, ex["true_bg_mid"], lw=1.0, ls="--", color="k", label="True bg")
        axes[0, 0].set_title(f"Background — 2s model  (M={m_bg_2s:.3f})")
        axes[0, 0].legend(fontsize=8)

        axes[0, 1].plot(t2, ex["recon_bg_4s"], lw=0.9, alpha=0.85, color="tomato", label="4s model")
        axes[0, 1].plot(t2, ex["true_bg_mid"], lw=1.0, ls="--", color="k", label="True bg")
        axes[0, 1].set_title(f"Background — 4s model  (M={m_bg_4s:.3f})")
        axes[0, 1].legend(fontsize=8)

        axes[1, 0].plot(t2, ex["recon_gl_2s"], lw=0.9, alpha=0.85, label="2s residual")
        axes[1, 0].plot(t2, ex["true_gl_mid"], lw=1.0, ls="--", color="k", label="True glitch")
        axes[1, 0].set_title(f"Glitch residual — 2s model  (M={m_gl_2s:.3f})")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].plot(t2, ex["recon_gl_4s"], lw=0.9, alpha=0.85, color="tomato", label="4s residual")
        axes[1, 1].plot(t2, ex["true_gl_mid"], lw=1.0, ls="--", color="k", label="True glitch")
        axes[1, 1].set_title(f"Glitch residual — 4s model  (M={m_gl_4s:.3f})")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"example_{idx}_{ex['s_type']}_td.png"), dpi=150)
        plt.close()

        # Spectrogram: 3×2 grid
        #   row 0: noisy input  |  true background
        #   row 1: 2s recon bg  |  4s recon bg
        #   row 2: 2s residual  |  4s residual
        fig, axes = plt.subplots(3, 2, figsize=(12, 9))
        fig.suptitle(f"Example {idx} — {ex['s_type']} | spectrograms (middle 2s)")

        _add_specgram(axes[0, 0], ex["noisy_mid"],    "Noisy input")
        _add_specgram(axes[0, 1], ex["true_bg_mid"],  "True background")
        _add_specgram(axes[1, 0], ex["recon_bg_2s"],  f"2s model recon  (M={m_bg_2s:.3f})")
        _add_specgram(axes[1, 1], ex["recon_bg_4s"],  f"4s model recon  (M={m_bg_4s:.3f})")
        _add_specgram(axes[2, 0], ex["recon_gl_2s"],  f"2s glitch residual  (M={m_gl_2s:.3f})")
        _add_specgram(axes[2, 1], ex["recon_gl_4s"],  f"4s glitch residual  (M={m_gl_4s:.3f})")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"example_{idx}_{ex['s_type']}_spec.png"), dpi=150)
        plt.close()

    # ── Single-panel showcase plots (first 2 examples only) ──────────────────
    for idx, ex in enumerate(examples[:2]):
        m_bg_2s = _overlap(ex["true_bg_mid"], ex["recon_bg_2s"])
        m_bg_4s = _overlap(ex["true_bg_mid"], ex["recon_bg_4s"])

        fig, (ax_bg, ax_gl) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        fig.suptitle(f"Reconstruction showcase — {ex['s_type']} (middle 2s)", fontsize=11)

        # Background reconstruction
        ax_bg.plot(t2, ex["noisy_mid"], lw=0.6, color="grey", alpha=0.5, label="Noisy input")
        ax_bg.plot(t2, ex["recon_bg_2s"], lw=1.1, color="steelblue",
                   label=f"2s model  (M={m_bg_2s:.3f})")
        ax_bg.plot(t2, ex["recon_bg_4s"], lw=1.1, color="tomato",
                   label=f"4s model  (M={m_bg_4s:.3f})")
        ax_bg.plot(t2, ex["true_bg_mid"], lw=1.2, ls="--", color="k", label="True background")
        ax_bg.set_ylabel("Strain (scaled)")
        ax_bg.set_title("Background reconstruction")
        ax_bg.legend(fontsize=8, loc="upper right")

        # Glitch residual
        m_gl_2s = _overlap(ex["true_gl_mid"], ex["recon_gl_2s"])
        m_gl_4s = _overlap(ex["true_gl_mid"], ex["recon_gl_4s"])
        ax_gl.plot(t2, ex["recon_gl_2s"], lw=1.1, color="steelblue",
                   label=f"2s residual  (M={m_gl_2s:.3f})")
        ax_gl.plot(t2, ex["recon_gl_4s"], lw=1.1, color="tomato",
                   label=f"4s residual  (M={m_gl_4s:.3f})")
        ax_gl.plot(t2, ex["true_gl_mid"], lw=1.2, ls="--", color="k", label="True glitch")
        ax_gl.set_ylabel("Strain (scaled)")
        ax_gl.set_xlabel("Time (s)")
        ax_gl.set_title("Glitch residual")
        ax_gl.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"showcase_{idx}_{ex['s_type']}.png"), dpi=150)
        plt.close()

    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
