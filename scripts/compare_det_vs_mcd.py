"""
Compare deterministic 4s model vs MC Dropout 4s model.

For each test sample the MC Dropout model runs N stochastic forward passes
to produce a posterior over reconstructions.  The plot shows:
  - Deterministic model: single reconstruction line
  - MC Dropout model: median line + shaded 1-sigma band (16th–84th percentile)
  - Ground truth: black dashed line

Mismatch is reported for both the deterministic reconstruction and the MC median.

Usage::

    python scripts/compare_det_vs_mcd.py \\
        --checkpoint-det  pretrained/DeepExtractor_257_4s/checkpoint_best_bilby_noise_base.pth.tar \\
        --checkpoint-mcd  pretrained/DeepExtractor_257_4s_mcd/checkpoint_best_bilby_noise_base.pth.tar \\
        --scaler          src/deepextractor/assets/scaler_bilby_4s.pkl \\
        --dropout-p       0.1 \\
        --n-passes        100 \\
        --num-samples     300 \\
        --output-dir      evaluation/compare_det_vs_mcd/
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

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 4096
LEN_4S = 4 * SAMPLE_RATE    # 16384
LEN_2S = 2 * SAMPLE_RATE    # 8192
MID_START = LEN_4S // 4     # 4096
MID_END = 3 * LEN_4S // 4   # 12288

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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_model(checkpoint_path, dropout_p, device):
    model = UNET2D(in_channels=2, out_channels=2, dropout_p=dropout_p).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    load_checkpoint(ckpt, model)
    model.eval()
    return model


def _generate_test_sample(force_type=None):
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


def _stft_input(noisy_ts, scaler, window, device):
    scaled = scaler.transform(noisy_ts.reshape(-1, 1)).reshape(1, -1).astype(np.float32)
    return apply_stft(scaled, N_FFT, HOP_LENGTH, WIN_LENGTH, window).to(device)


def _inv_scale(td_np, scaler):
    """Inverse-scale a (P, L) numpy array in one vectorised call."""
    P, L = td_np.shape
    return scaler.inverse_transform(td_np.reshape(-1, 1)).reshape(P, L)


def _overlap(a, b):
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 1e-30 else 0.0


def _mismatch_pct(a, b):
    return (1 - _overlap(a, b)) * 100


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare deterministic vs MC Dropout 4s DeepExtractor models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-det", required=True,
                        help="Deterministic 4s model checkpoint.")
    parser.add_argument("--checkpoint-mcd", required=True,
                        help="MC Dropout 4s model checkpoint.")
    parser.add_argument("--scaler", required=True,
                        help="4s StandardScaler (.pkl) — same for both models.")
    parser.add_argument("--dropout-p", type=float, default=0.1,
                        help="Dropout probability used when the MCD model was trained.")
    parser.add_argument("--n-passes", type=int, default=100,
                        help="Number of MC forward passes for the dropout model.")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--plot-n", type=int, default=6,
                        help="Number of example plots to save.")
    parser.add_argument("--output-dir", default="evaluation/compare_det_vs_mcd/")
    parser.add_argument(
        "--signal-type", default=None,
        choices=list(SIGNAL_FN) + list(SIGNAL_FN_NODUR),
        help="Fix all test samples to this signal type. Random if not set.",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    window = torch.hann_window(WIN_LENGTH).to(device)

    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)

    model_det = _load_model(args.checkpoint_det, dropout_p=0.0, device=device)
    model_mcd = _load_model(args.checkpoint_mcd, dropout_p=args.dropout_p, device=device)
    enable_mc_dropout(model_mcd)

    istft_fn = functools.partial(
        apply_istft, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window,
    )

    results = {
        "mismatch_det": [], "mismatch_mcd_median": [],
        "mismatch_det_gl": [], "mismatch_mcd_gl_median": [],
        "signal_type": [],
    }
    examples = []
    n_tried = 0

    with tqdm(total=args.num_samples, desc="Evaluating") as pbar:
        while len(results["mismatch_det"]) < args.num_samples:
            n_tried += 1
            if n_tried > args.num_samples * 3:
                print("WARNING: too many failed sample generations — stopping early.")
                break

            noisy_4s, noise_4s, true_gl_4s, s_type, snr = _generate_test_sample(args.signal_type)
            if noisy_4s is None:
                continue

            true_bg_mid = noise_4s[MID_START:MID_END]
            true_gl_mid = true_gl_4s[MID_START:MID_END]
            x_stft = _stft_input(noisy_4s, scaler, window, device)

            # Deterministic model — single forward pass
            with torch.no_grad():
                pred_det = model_det(x_stft)
            td_det = apply_istft(pred_det, N_FFT, HOP_LENGTH, WIN_LENGTH, window)
            recon_bg_det = scaler.inverse_transform(
                td_det.cpu().numpy().squeeze().reshape(-1, 1)
            ).reshape(-1)
            recon_bg_det_mid = recon_bg_det[MID_START:MID_END]
            recon_gl_det_mid = noisy_4s[MID_START:MID_END] - recon_bg_det_mid

            # MC Dropout model — N passes, batched iSTFT
            samples_td = mc_predict(
                model_mcd, x_stft, n_passes=args.n_passes, postprocess_fn=istft_fn,
            )  # (P, 1, L)
            samples_np = _inv_scale(
                samples_td.cpu().numpy().squeeze(1), scaler
            )  # (P, L)

            mc_median    = np.median(samples_np, axis=0)
            mc_p16       = np.percentile(samples_np, 16, axis=0)
            mc_p84       = np.percentile(samples_np, 84, axis=0)
            mc_median_mid = mc_median[MID_START:MID_END]
            mc_p16_mid    = mc_p16[MID_START:MID_END]
            mc_p84_mid    = mc_p84[MID_START:MID_END]
            mc_gl_mid     = noisy_4s[MID_START:MID_END] - mc_median_mid

            results["mismatch_det"].append(_mismatch_pct(true_bg_mid, recon_bg_det_mid))
            results["mismatch_mcd_median"].append(_mismatch_pct(true_bg_mid, mc_median_mid))
            results["mismatch_det_gl"].append(_mismatch_pct(true_gl_mid, recon_gl_det_mid))
            results["mismatch_mcd_gl_median"].append(_mismatch_pct(true_gl_mid, mc_gl_mid))
            results["signal_type"].append(s_type)

            if len(examples) < args.plot_n:
                examples.append({
                    "s_type": s_type,
                    "snr": snr,
                    "t_mid": np.linspace(0, 2, LEN_2S),
                    "noisy_mid": noisy_4s[MID_START:MID_END],
                    "true_bg_mid": true_bg_mid,
                    "true_gl_mid": true_gl_mid,
                    "recon_bg_det_mid": recon_bg_det_mid,
                    "recon_gl_det_mid": recon_gl_det_mid,
                    "mc_median_mid": mc_median_mid,
                    "mc_p16_mid": mc_p16_mid,
                    "mc_p84_mid": mc_p84_mid,
                    "mc_gl_mid": mc_gl_mid,
                    "mc_gl_p16": np.percentile(
                        samples_np[:, MID_START:MID_END] - noisy_4s[MID_START:MID_END], 16, axis=0,
                    ) * -1,
                    "mc_gl_p84": np.percentile(
                        samples_np[:, MID_START:MID_END] - noisy_4s[MID_START:MID_END], 84, axis=0,
                    ) * -1,
                })

            pbar.update(1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Background reconstruction mismatch (middle 2s) ──────────")
    for key in ("mismatch_det", "mismatch_mcd_median"):
        arr = np.array(results[key])
        print(f"  {key:30s}  mean={arr.mean():.3f}%  median={np.median(arr):.3f}%  std={arr.std():.3f}%")

    print("\n── Glitch residual mismatch (middle 2s) ─────────────────────")
    for key in ("mismatch_det_gl", "mismatch_mcd_gl_median"):
        arr = np.array(results[key])
        print(f"  {key:30s}  mean={arr.mean():.3f}%  median={np.median(arr):.3f}%  std={arr.std():.3f}%")

    print("\n── Per signal-type breakdown (background mismatch) ──────────")
    for stype in sorted(set(results["signal_type"])):
        idx = [i for i, s in enumerate(results["signal_type"]) if s == stype]
        m_det = np.mean([results["mismatch_det"][i] for i in idx])
        m_mcd = np.mean([results["mismatch_mcd_median"][i] for i in idx])
        print(f"  {stype:20s}  det={m_det:.3f}%  mcd={m_mcd:.3f}%  Δ={m_mcd-m_det:+.3f}%  (n={len(idx)})")

    with open(os.path.join(args.output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    # ── Mismatch distribution plot ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k_det, k_mcd, title in zip(
        axes,
        ("mismatch_det", "mismatch_det_gl"),
        ("mismatch_mcd_median", "mismatch_mcd_gl_median"),
        ("Background mismatch (%)", "Glitch residual mismatch (%)"),
    ):
        ax.hist(results[k_det], bins=40, alpha=0.6, color="steelblue", label="Deterministic")
        ax.hist(results[k_mcd], bins=40, alpha=0.6, color="tomato",    label="MC Dropout (median)")
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
    plt.suptitle("Deterministic vs MC Dropout 4s model")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "mismatch_distributions.png"), dpi=150)
    plt.close()

    # ── Example showcase plots ────────────────────────────────────────────────
    for idx, ex in enumerate(examples):
        t = ex["t_mid"]
        mm_det = _mismatch_pct(ex["true_bg_mid"], ex["recon_bg_det_mid"])
        mm_mcd = _mismatch_pct(ex["true_bg_mid"], ex["mc_median_mid"])
        mm_gl_det = _mismatch_pct(ex["true_gl_mid"], ex["recon_gl_det_mid"])
        mm_gl_mcd = _mismatch_pct(ex["true_gl_mid"], ex["mc_gl_mid"])

        fig, (ax_bg, ax_gl) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(
            f"Example {idx} — {ex['s_type']} | SNR={ex['snr']:.1f} | middle 2s\n"
            f"Background mismatch: det={mm_det:.2f}%  mcd={mm_mcd:.2f}%",
            fontsize=10,
        )

        # Background reconstruction
        ax_bg.fill_between(t, ex["mc_p16_mid"], ex["mc_p84_mid"],
                           color="tomato", alpha=0.25, label="MC 1σ band")
        ax_bg.plot(t, ex["mc_median_mid"], color="tomato", lw=1.1,
                   label=f"MC median  ({mm_mcd:.2f}% mismatch)")
        ax_bg.plot(t, ex["recon_bg_det_mid"], color="steelblue", lw=1.1,
                   label=f"Deterministic  ({mm_det:.2f}% mismatch)")
        ax_bg.plot(t, ex["true_bg_mid"], color="k", lw=1.1, ls="--", label="True background")
        ax_bg.set_ylabel("Strain (scaled)")
        ax_bg.set_title("Background reconstruction")
        ax_bg.legend(fontsize=8, loc="upper right")

        # Glitch residual
        ax_gl.fill_between(t, ex["mc_gl_p16"], ex["mc_gl_p84"],
                           color="tomato", alpha=0.25, label="MC 1σ band")
        ax_gl.plot(t, ex["mc_gl_mid"], color="tomato", lw=1.1,
                   label=f"MC residual  ({mm_gl_mcd:.2f}% mismatch)")
        ax_gl.plot(t, ex["recon_gl_det_mid"], color="steelblue", lw=1.1,
                   label=f"Det. residual  ({mm_gl_det:.2f}% mismatch)")
        ax_gl.plot(t, ex["true_gl_mid"], color="k", lw=1.1, ls="--", label="True glitch")
        ax_gl.set_ylabel("Strain (scaled)")
        ax_gl.set_xlabel("Time (s)")
        ax_gl.set_title("Glitch residual")
        ax_gl.legend(fontsize=8, loc="upper right")

        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, f"showcase_{idx}_{ex['s_type']}.png"), dpi=150
        )
        plt.close()

    print(f"\nPlots and results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
