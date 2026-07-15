"""
Plot real detector background samples from the scaled time-domain HDF5,
optionally alongside bilby noise for comparison.

Usage::

    python scripts/plot_real_backgrounds.py \\
        --td-dir data_4s_real/o3_td \\
        --n-samples 6 \\
        --output-dir evaluation/real_backgrounds/

    # With bilby comparison
    python scripts/plot_real_backgrounds.py \\
        --td-dir data_4s_real/o3_td \\
        --bilby-h5 data_4s/bilby_noise_hdf5/time_domain/strain_data_scaled.h5 \\
        --n-samples 6 \\
        --output-dir evaluation/real_backgrounds/
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from deepextractor.utils.stft import apply_stft

SAMPLE_RATE = 4096
N_FFT, HOP_LENGTH, WIN_LENGTH = 512, 32, 64


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--td-dir", required=True,
                        help="Directory containing strain_data_scaled.h5 (phase 1 output).")
    parser.add_argument("--bilby-h5", default=None,
                        help="Path to bilby strain_data_scaled.h5 for comparison.")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--output-dir", default="evaluation/real_backgrounds/")
    parser.add_argument("--split", default="train",
                        choices=["train", "val"],
                        help="Which split to sample from.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    window = torch.hann_window(WIN_LENGTH)
    t = np.linspace(0, 4, 4 * SAMPLE_RATE)

    scaled_h5 = os.path.join(args.td_dir, "strain_data_scaled.h5")
    bg_key    = f"background_{args.split}"
    ng_key    = f"noisy_glitch_{args.split}"

    with h5py.File(scaled_h5, "r") as f:
        n_total = f[bg_key].shape[0]
        idx = np.random.default_rng(0).choice(n_total, size=args.n_samples, replace=False)
        bg_real = f[bg_key][sorted(idx)]        # (N, 16384) — real noise, no glitch
        ng_real = f[ng_key][sorted(idx)]        # (N, 16384) — real noise + injected glitch

    bilby_bg = None
    if args.bilby_h5 is not None:
        with h5py.File(args.bilby_h5, "r") as f:
            n_bilby = f[bg_key].shape[0]
            idx_b = np.random.default_rng(1).choice(n_bilby, size=args.n_samples, replace=False)
            bilby_bg = f[bg_key][sorted(idx_b)]

    # ── 1. Time-domain panel ─────────────────────────────────────────────────
    ncols = 2 if bilby_bg is not None else 1
    fig, axes = plt.subplots(
        args.n_samples, ncols, figsize=(7 * ncols, 2.5 * args.n_samples), squeeze=False
    )
    for i in range(args.n_samples):
        ax = axes[i, 0]
        ax.plot(t, bg_real[i], lw=0.5, color="steelblue", label="background")
        ax.plot(t, ng_real[i], lw=0.5, color="crimson", alpha=0.7, label="noisy+glitch")
        ax.set_ylabel("Scaled strain")
        ax.set_title(f"Real O3 sample {i}")
        if i == 0:
            ax.legend(fontsize=8)
        if bilby_bg is not None:
            ax2 = axes[i, 1]
            ax2.plot(t, bilby_bg[i], lw=0.5, color="steelblue")
            ax2.set_title(f"Bilby sample {i}")
            ax2.set_ylabel("Scaled strain")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"Scaled time-domain: real O3 vs bilby ({args.split})", y=1.01)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "timeseries.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ── 2. Spectrogram panel ─────────────────────────────────────────────────
    ncols = 2 if bilby_bg is not None else 1
    fig, axes = plt.subplots(
        args.n_samples, ncols, figsize=(7 * ncols, 3 * args.n_samples), squeeze=False
    )
    for i in range(args.n_samples):
        stft = apply_stft(bg_real[i:i+1], N_FFT, HOP_LENGTH, WIN_LENGTH, window)
        mag = stft[0, 0].numpy()  # (F, T)
        ax = axes[i, 0]
        ax.imshow(np.log1p(mag), aspect="auto", origin="lower",
                  extent=[0, 4, 0, SAMPLE_RATE // 2], cmap="viridis")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f"Real O3 sample {i} — log(1+|STFT|)")
        if bilby_bg is not None:
            stft_b = apply_stft(bilby_bg[i:i+1], N_FFT, HOP_LENGTH, WIN_LENGTH, window)
            mag_b = stft_b[0, 0].numpy()
            ax2 = axes[i, 1]
            ax2.imshow(np.log1p(mag_b), aspect="auto", origin="lower",
                       extent=[0, 4, 0, SAMPLE_RATE // 2], cmap="viridis")
            ax2.set_title(f"Bilby sample {i} — log(1+|STFT|)")
            ax2.set_ylabel("Frequency (Hz)")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"Spectrograms: real O3 vs bilby ({args.split})", y=1.01)
    plt.tight_layout()
    out = os.path.join(args.output_dir, "spectrograms.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ── 3. Distribution comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(bg_real.ravel(), bins=200, density=True, color="steelblue",
                 alpha=0.7, label="Real O3 background")
    if bilby_bg is not None:
        axes[0].hist(bilby_bg.ravel(), bins=200, density=True, color="crimson",
                     alpha=0.7, label="Bilby background")
    axes[0].set_xlabel("Scaled strain")
    axes[0].set_title("Time-domain value distribution")
    axes[0].legend()

    stft_real = apply_stft(bg_real, N_FFT, HOP_LENGTH, WIN_LENGTH, window)
    mag_real = stft_real[:, 0].numpy().ravel()
    axes[1].hist(np.log1p(mag_real), bins=200, density=True, color="steelblue",
                 alpha=0.7, label="Real O3")
    if bilby_bg is not None:
        stft_bilby = apply_stft(bilby_bg, N_FFT, HOP_LENGTH, WIN_LENGTH, window)
        mag_bilby = stft_bilby[:, 0].numpy().ravel()
        axes[1].hist(np.log1p(mag_bilby), bins=200, density=True, color="crimson",
                     alpha=0.7, label="Bilby")
    axes[1].set_xlabel("log(1 + |STFT magnitude|)")
    axes[1].set_title("STFT magnitude distribution")
    axes[1].legend()

    plt.tight_layout()
    out = os.path.join(args.output_dir, "distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
