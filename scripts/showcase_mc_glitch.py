"""
MC Dropout glitch extraction showcase.

For each test sample, runs N stochastic forward passes and shows the
extracted glitch residual as a median + 1-sigma band, with the true
injected glitch overlaid at low alpha.

Usage::

    python scripts/showcase_mc_glitch.py \\
        --checkpoint pretrained/DeepExtractor_257_4s_mcd/checkpoint_best_bilby_noise_base.pth.tar \\
        --scaler     src/deepextractor/assets/scaler_bilby_4s.pkl \\
        --signal-type gengli_H1 \\
        --n-passes   50 \\
        --num-samples 10 \\
        --output-dir evaluation/showcase_mc/
"""

import argparse
import functools
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
from deepextractor.utils.mc_dropout import enable_mc_dropout, mc_predict
from deepextractor.utils.signal import whitened_snr_scaling
from deepextractor.utils.stft import apply_istft, apply_stft

SAMPLE_RATE = 4096
LEN_4S    = 4 * SAMPLE_RATE
LEN_2S    = 2 * SAMPLE_RATE
MID_START  = LEN_4S // 4           # 4096 — start of middle 2s
MID_END    = 3 * LEN_4S // 4       # 12288 — end of middle 2s
MID1S_START = LEN_4S // 2 - SAMPLE_RATE // 2   # 6144 — start of middle 1s
MID1S_END   = LEN_4S // 2 + SAMPLE_RATE // 2   # 10240 — end of middle 1s

N_FFT, WIN_LENGTH, HOP_LENGTH = 512, 64, 32

SIGNAL_FN = {
    "chirp": generate_chirp,
    "sine": generate_sine,
    "sine_gaussian": generate_sine_gaussian,
    "gaussian_pulse": generate_gaussian_pulse,
    "ringdown": ringdown,
}
SIGNAL_FN_NODUR = {
    "gengli_H1": lambda: generate_gengli_glitch(ifo="H1"),
    "gengli_L1": lambda: generate_gengli_glitch(ifo="L1"),
}
SNR_MIN, SNR_MAX = 7.5, 100.0


def _load_model(checkpoint_path, dropout_p, device):
    model = UNET2D(in_channels=2, out_channels=2, dropout_p=dropout_p).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    load_checkpoint(ckpt, model)
    model.eval()
    return model


def _generate_sample(force_type=None):
    noise = np.random.randn(LEN_4S).astype(np.float32)
    all_types = list(SIGNAL_FN) + list(SIGNAL_FN_NODUR)
    s_type = force_type if force_type is not None else random.choice(all_types)
    if s_type in SIGNAL_FN_NODUR:
        _, signal = SIGNAL_FN_NODUR[s_type]()
    else:
        _, signal = SIGNAL_FN[s_type](np.random.uniform(0.125, 2.0))
    signal = signal.squeeze()
    if np.isnan(signal).any() or len(signal) == 0:
        return None, None, None, None, None
    snr = np.random.uniform(SNR_MIN, SNR_MAX)
    glitch = signal - np.mean(signal)
    glitch = whitened_snr_scaling(glitch, snr=snr / np.sqrt(SAMPLE_RATE / 2))
    id_start = LEN_4S // 2 - len(glitch) // 2
    id_end = id_start + len(glitch)
    if id_end > LEN_4S:
        return None, None, None, None, None
    noisy = noise.copy()
    noisy[id_start:id_end] += glitch
    true_gl = np.zeros(LEN_4S, dtype=np.float32)
    true_gl[id_start:id_end] = glitch
    return noisy, noise, true_gl, s_type, snr


def _inv_scale(td_np, scaler):
    P, L = td_np.shape
    return scaler.inverse_transform(td_np.reshape(-1, 1)).reshape(P, L)


def main():
    parser = argparse.ArgumentParser(
        description="MC Dropout glitch extraction showcase.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="MC Dropout model checkpoint (.pth.tar).")
    parser.add_argument("--scaler", required=True,
                        help="StandardScaler (.pkl).")
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--n-passes", type=int, default=50,
                        help="Number of MC forward passes.")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument(
        "--signal-type", default=None,
        choices=list(SIGNAL_FN) + list(SIGNAL_FN_NODUR),
        help="Fix all samples to this signal type. Random if not set.",
    )
    parser.add_argument("--true-glitch-alpha", type=float, default=0.35,
                        help="Alpha for the true glitch overlay.")
    parser.add_argument("--output-dir", default="evaluation/showcase_mc/")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    window = torch.hann_window(WIN_LENGTH).to(device)

    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)

    model = _load_model(args.checkpoint, args.dropout_p, device)
    enable_mc_dropout(model)

    istft_fn = functools.partial(
        apply_istft, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window,
    )

    collected = 0
    n_tried = 0

    with tqdm(total=args.num_samples, desc="Generating") as pbar:
        while collected < args.num_samples:
            n_tried += 1
            if n_tried > args.num_samples * 5:
                print("WARNING: too many failed generations — stopping early.")
                break

            noisy_4s, noise_4s, true_gl_4s, s_type, snr = _generate_sample(args.signal_type)
            if noisy_4s is None:
                continue


            scaled = scaler.transform(noisy_4s.reshape(-1, 1)).reshape(1, -1).astype(np.float32)
            x_stft = apply_stft(scaled, N_FFT, HOP_LENGTH, WIN_LENGTH, window).to(device)

            samples_td = mc_predict(
                model, x_stft, n_passes=args.n_passes, postprocess_fn=istft_fn,
            )  # (P, 1, L)
            samples_np = _inv_scale(samples_td.cpu().numpy().squeeze(1), scaler)  # (P, L)

            # Glitch residuals: noisy - reconstructed background
            gl_samples = noisy_4s[np.newaxis, :] - samples_np       # (P, L)
            gl_1s      = gl_samples[:, MID1S_START:MID1S_END]       # (P, 1s)
            noisy_1s   = noisy_4s[MID1S_START:MID1S_END]
            true_gl_1s = true_gl_4s[MID1S_START:MID1S_END]

            gl_median = np.median(gl_1s, axis=0)
            gl_p10    = np.percentile(gl_1s, 10, axis=0)
            gl_p90    = np.percentile(gl_1s, 90, axis=0)

            # Mismatch of MC median vs true glitch over the middle 1s
            denom = np.sqrt(np.dot(gl_median, gl_median) * np.dot(true_gl_1s, true_gl_1s))
            overlap = float(np.dot(gl_median, true_gl_1s) / denom) if denom > 1e-30 else 0.0
            mismatch = (1 - overlap) * 100

            t = np.linspace(0, 1, SAMPLE_RATE)

            fig, ax = plt.subplots(figsize=(11, 4))
            # Noisy input behind everything
            ax.plot(t, noisy_1s, color="grey", lw=0.7, alpha=0.25,
                    label="Noisy input", zorder=1)
            # MC posterior
            ax.fill_between(t, gl_p10, gl_p90, color="crimson", alpha=0.45,
                            label="MC 10–90% band", zorder=2)
            ax.plot(t, gl_median, color="steelblue", lw=1.2,
                    label="MC median (extracted glitch)", zorder=3)
            # True glitch overlaid at low alpha on top
            ax.plot(t, true_gl_1s, color="k", lw=1.0, ls="--",
                    alpha=args.true_glitch_alpha, label="True glitch", zorder=4)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Strain (scaled)")
            ax.set_title(
                f"{s_type}  |  SNR={snr:.1f}  |  mismatch={mismatch:.2f}%  "
                f"|  {args.n_passes} MC passes  |  10–90% band  (middle 1s)"
            )
            ax.legend(fontsize=9, loc="upper right")
            plt.tight_layout()
            out_path = os.path.join(args.output_dir, f"mc_glitch_{collected:03d}_{s_type}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

            collected += 1
            pbar.update(1)

    print(f"\n{collected} plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
