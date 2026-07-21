#!/usr/bin/env python3
"""
GravitySpy validation of a 4s DeepExtractor transfer-learned model on real glitches
from the same run period (O3, O4, ...).

The thing we care about is the RESIDUAL GLITCH: input − DeepExtractor background.
DeepExtractor separates glitch from noise; we take the glitch component and check
whether its morphology is preserved well enough for GravitySpy to still identify
the class.

Pipeline:
  1. Fetch real glitch strain (GWOSC, or local frames on CIT via --data-source cit)
     → GWpy-whiten (identical params to training)
     → DeepExtractor (MC Dropout) → residual glitch = whitened_input − background.
  2. Inject that residual into a SEPARATE clean noise segment (same GWpy whitening)
     so GravitySpy sees a realistic glitch-in-noise signal, not the original data.
  3. Classify with GravitySpy → confusion matrix.

Outputs (in --output-dir):
    timeseries_{class}.png     — whitened input (grey) + residual glitch (k--)
    qscan_input_{class}.png    — Q=10 Q-scan of whitened input
    qscan_residual_{class}.png — Q=10 Q-scan of extracted residual glitch
    gspy_results.csv           — per-sample GravitySpy classification results
    confusion_matrix.pdf/png   — confusion matrix

Dependencies
------------
Install deepextractor itself plus every runtime dependency GravitySpy needs, then
install GravitySpy separately with --no-deps (its PyPI metadata pins broken
scipy/keras versions) and apply nine small patches for Python 3.11+ / keras 3.x /
gwpy 4.x / matplotlib 3.3+ / numpy>=1.24 / scikit-image compatibility:

    pip install -e ".[gspy]"                   # deepextractor + gravityspy runtime deps
    pip install gravityspy==1.0.0 --no-deps    # gravityspy (skip its broken scipy pin)
    python patch_gspy.py                       # apply the 9 compatibility patches below

`patch_gspy.py` (repo root) applies these fixes directly to the installed package:

    # Fix 1: scipy.misc.imresize removed in scipy 1.3
    from scipy.misc import imresize  →  from skimage.transform import resize as imresize
    # Fix 2: keras 3.x loads old .h5 models incorrectly — use tf_keras (legacy keras 2.x)
    from keras.applications.vgg16 import preprocess_input  →  from tf_keras...
    from keras.models import load_model                    →  from tf_keras...
    from keras import backend as K                         →  from tf_keras...
    # Fix 3: gwpy 4.0 rejects bare truthiness checks on TimeSeries
    if timeseries:  →  if timeseries is not None:
    # Fix 4: gwpy 4.x's Series.crop() dropped the verbose kwarg entirely
    timeseries.crop(start_time, stop_time, verbose=verbose)  →  timeseries.crop(start_time, stop_time)
    # Fix 5: keras.utils.np_utils removed (Keras 1/2-only name). Imported
    # transitively via gravityspy.table.Events even when only classify() is used.
    from keras.utils import np_utils  →  shim built on tensorflow.keras.utils.to_categorical
    # Fix 6: matplotlib >=3.3 renamed LogScale's basex/basey kwargs to base
    ax.set_yscale('log', basey=2)  →  ax.set_yscale('log', base=2)
    # Fix 7: skimage renamed rescale()'s multichannel=bool to channel_axis, and
    # numpy removed the np.int alias entirely (both in ml/read_image.py)
    rescale(..., multichannel=False)  →  rescale(..., channel_axis=None)
    rescale(..., multichannel=True)   →  rescale(..., channel_axis=-1)
    np.int(...)  →  int(...)
    # Fix 8: ml/__init__.py and ml/GS_utils.py import bare keras (Keras 3.x
    # standalone) instead of tensorflow.keras. Both run unconditionally on any
    # gravityspy.ml import — GS_utils via train_classifier's own import (Fix 5).
    from keras import backend as K  →  from tensorflow.keras import backend as K
    (+ regularizers.l2, models.Sequential, layers.{Dense,Dropout,...} in GS_utils.py)
    # Fix 9: predict_proba() removed from Keras models entirely — replaced by
    # predict(), which returns probabilities directly for classification outputs.
    final_model.predict_proba(...)  →  final_model.predict(...)

--data-source cit additionally needs gwdatafind (pip install -e ".[site]"), and only
works on a LIGO Data Grid machine (e.g. CIT) with datafind + frame access configured.

Usage::

    # O3 — extract + reconstruct only (--skip-gspy: no injection, no classification)
    python scripts/validate_gspy_o3.py \\
        --checkpoint checkpoints_4s_o3_mcd/DeepExtractor_257_checkpoints/checkpoint_best_o3_mcd_tl.pth.tar \\
        --scaler     data_4s_real/o3_td/scaler_real.pkl \\
        --n-per-class 3 \\
        --output-dir evaluation/gspy_4s_o3 \\
        --skip-gspy

    # O4 — same, no --clean-gps-h1/-l1 or --gspy-model needed since --skip-gspy
    python scripts/validate_gspy_o3.py \\
        --run-label  O4 \\
        --checkpoint checkpoints_4s_o4_mcd/DeepExtractor_257_checkpoints/checkpoint_best_o4_mcd_tl.pth.tar \\
        --scaler     data_4s_real/o4_td/scaler_real.pkl \\
        --csv        data_o4_high_confidence.csv \\
        --n-per-class 3 \\
        --output-dir evaluation/gspy_4s_o4 \\
        --skip-gspy

    # Full pipeline with GravitySpy re-classification (injects extracted residual
    # into a separate clean noise segment first, since GravitySpy expects realistic
    # strain context, not a bare isolated array). Run on CIT with --data-source cit
    # so every per-glitch fetch reads local frame files via gwdatafind instead of
    # round-tripping to the public GWOSC API:
    python scripts/validate_gspy_o3.py \\
        --run-label     O3 \\
        --checkpoint    checkpoints_4s_o3_mcd/DeepExtractor_257_checkpoints/checkpoint_best_o3_mcd_tl.pth.tar \\
        --scaler        data_4s_real/o3_td/scaler_real.pkl \\
        --gspy-model    /Users/tomdooney/Documents/Work/Projects/glitchgan/models/sidd-cqg-paper-O3-model.h5 \\
        --n-per-class   3 \\
        --data-source   cit \\
        --output-dir    evaluation/gspy_4s_o3
"""

import argparse
import functools
import logging
import os
import pickle
import shutil
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from gwdatafind import find_urls
from gwpy.timeseries import TimeSeries
from tqdm import tqdm

from deepextractor.models.architectures import UNET2D
from deepextractor.utils.checkpoints import load_checkpoint
from deepextractor.utils.mc_dropout import enable_mc_dropout, mc_predict
from deepextractor.utils.stft import apply_istft, apply_stft
from deepextractor.utils.visualization import plot_q_transform

# ── constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 4096
LEN_4S         = 4 * SAMPLE_RATE
N_FFT, HOP_LENGTH, WIN_LENGTH = 512, 32, 64

# Whitening params — must exactly match get_clean_backgrounds.py
FETCH_DUR      = 36        # s fetched per window (same as CONTEXT_DURATION)
MAX_FILTER_DUR = 2         # s trimmed from each edge after whitening
PSD_FFTLENGTH  = 4         # s — fftlength arg to GWpy whiten
PSD_OVERLAP    = 2         # s — overlap arg to GWpy whiten
HIGHPASS_HZ    = 10.0
MAX_AMP        = 30.0      # clip threshold on whitened strain

# CIT-local frame access (--data-source cit) — same channel/frame-type mapping
# as get_clean_backgrounds.py, reads via gwdatafind instead of the public GWOSC API.
FRAME_TYPE     = {"H1": "H1_HOFT_C00", "L1": "L1_HOFT_C00"}

# Default clean O3b GPS centre for H1, only used when --run-label O3, --ifo H1,
# and no --clean-gps-h1 override is given. This segment was vetted quiet for H1
# specifically (source: gspy_classification.ipynb) — it is NOT verified clean for
# L1, so there is no equivalent L1 default. Other run labels (e.g. O4) must pass
# a clean GPS centre explicitly — an O3 quiet segment is not a valid injection
# background for an O4-era model either.
DEFAULT_CLEAN_GPS_H1 = 1262540000

LABEL_ORDER = [
    "Blip", "Fast_Scattering", "Koi_Fish",
    "Low_Frequency_Burst", "Scattered_Light", "Tomte", "Whistle",
]


# ── whitening ─────────────────────────────────────────────────────────────────

def _whiten(ts: TimeSeries) -> np.ndarray:
    """Whiten and trim, using identical params to get_clean_backgrounds.py."""
    w = ts.whiten(fftlength=PSD_FFTLENGTH, overlap=PSD_OVERLAP, highpass=HIGHPASS_HZ)
    pad = MAX_FILTER_DUR * SAMPLE_RATE
    arr = np.array(w, dtype=np.float32)[pad:-pad]
    np.clip(arr, -MAX_AMP, MAX_AMP, out=arr)
    return arr


def _fetch_raw(ifo: str, start: float, end: float, data_source: str) -> TimeSeries:
    """Fetch raw (un-whitened) strain over [start, end) from the chosen source.

    'open'  — TimeSeries.fetch_open_data, the public GWOSC API (works anywhere,
              but is a per-call HTTP round trip — slow for many individual fetches).
    'cit'   — gwdatafind.find_urls + TimeSeries.read, reading local/network frame
              files directly. Only works on LIGO Data Grid machines (e.g. CIT) with
              datafind + frame access configured, same pattern as get_clean_backgrounds.py.
    """
    if data_source == "open":
        return TimeSeries.fetch_open_data(ifo, start, end, sample_rate=SAMPLE_RATE)

    urls = find_urls(ifo[0], FRAME_TYPE[ifo], start, end)
    ts = TimeSeries.read(urls, channel=f"{ifo}:GDS-CALIB_STRAIN", start=start, end=end)
    if ts.sample_rate.value != SAMPLE_RATE:
        ts = ts.resample(SAMPLE_RATE)
    return ts


def _fetch_whitened(ifo: str, gps_center: float, data_source: str = "open"):
    """Fetch FETCH_DUR seconds centred on gps_center and whiten.

    Returns (white_arr, central_4s, t0_gps).
    After trimming, white_arr is ~32 s.  central_4s is the middle 4 s.
    t0_gps is the true GPS start of white_arr (the raw fetch's t0 plus the
    MAX_FILTER_DUR trimmed off the front) — this is what must be used as the
    TimeSeries t0 when wrapping white_arr for GravitySpy, otherwise the injected
    residual ends up offset from where GravitySpy's Q-scan actually looks.
    """
    half = FETCH_DUR // 2
    ts = _fetch_raw(ifo, gps_center - half, gps_center + half, data_source)
    t0_gps = float(ts.t0.value) + MAX_FILTER_DUR
    white = _whiten(ts)

    c, h = len(white) // 2, LEN_4S // 2
    if c - h < 0 or c + h > len(white):
        raise ValueError(f"Whitened segment too short ({len(white)}) after trim")
    return white, white[c - h : c + h].copy(), t0_gps


# ── DeepExtractor inference ───────────────────────────────────────────────────

def _load_model(ckpt_path: str, dropout_p: float, device: str):
    model = UNET2D(in_channels=2, out_channels=2, dropout_p=dropout_p).to(device)
    load_checkpoint(torch.load(ckpt_path, map_location=device), model)
    model.eval()
    enable_mc_dropout(model)
    return model


def _extract_glitch(model, scaler, whitened_4s: np.ndarray,
                    window, istft_fn, device, n_passes: int):
    """Run MC Dropout DeepExtractor and return (background, residual_glitch).

    background = median reconstruction of the noise (glitch removed)
    residual   = whitened_4s − background  (the extracted glitch component)
    Both are in the same GWpy-whitened domain as whitened_4s.
    """
    scaled = scaler.transform(whitened_4s.reshape(-1, 1)).astype(np.float32).reshape(1, -1)
    x_stft = apply_stft(scaled, N_FFT, HOP_LENGTH, WIN_LENGTH, window.cpu()).to(device)

    passes = mc_predict(model, x_stft, n_passes=n_passes, postprocess_fn=istft_fn)
    passes_np = passes.cpu().numpy().squeeze(1)   # (P, L)

    bg_passes = scaler.inverse_transform(
        passes_np.reshape(-1, 1)
    ).reshape(passes_np.shape[0], -1).astype(np.float32)

    background = np.median(bg_passes, axis=0)      # (L,) — cleaned noise, glitch removed
    residual   = whitened_4s - background           # (L,) — extracted glitch
    return background, residual


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_timeseries(whitened_4s, residual, out_path, title):
    """Whitened input in grey, residual glitch in k-- so alignment is visible."""
    t = np.linspace(0, 4, LEN_4S)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, whitened_4s, color="grey", lw=0.6, alpha=0.3, label="Whitened input")
    ax.plot(t, residual,    "k--",        lw=1.0,             label="Residual glitch")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Whitened strain")
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_qscan(arr, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_q_transform(
        arr, srate=SAMPLE_RATE, whiten=False,
        qrange=[10, 10], frange=[10, 1200],
        ax=ax, colourbar=True,
    )
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_confusion(df_res, out_dir, run_label):
    df = df_res[df_res["pred_label"] != "Error"].copy()
    if df.empty:
        print("No valid GravitySpy results — skipping confusion matrix.")
        return

    pred_all  = sorted(df["pred_label"].unique())
    pred_cols = (
        [l for l in LABEL_ORDER if l in pred_all]
        + [l for l in pred_all if l not in LABEL_ORDER]
    )
    count_matrix = pd.DataFrame(0, index=LABEL_ORDER, columns=pred_cols)
    conf_acc     = {(t, p): [] for t in LABEL_ORDER for p in pred_cols}
    for _, row in df.iterrows():
        t, p, c = row["true_label"], row["pred_label"], row["confidence"]
        if t in LABEL_ORDER and p in pred_cols:
            count_matrix.loc[t, p] += 1
            conf_acc[(t, p)].append(c)

    annot = pd.DataFrame("", index=LABEL_ORDER, columns=pred_cols)
    for t in LABEL_ORDER:
        for p in pred_cols:
            n = count_matrix.loc[t, p]
            annot.loc[t, p] = "0" if n == 0 else f"{n}\n({np.mean(conf_acc[(t,p)]):.2f})"

    total = count_matrix.values.sum()
    acc   = np.trace(count_matrix.values) / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(max(10, len(pred_cols) * 1.1), 6))
    sns.heatmap(count_matrix, annot=annot, fmt="", cmap="Blues",
                linewidths=0.5, linecolor="gray",
                annot_kws={"size": 8, "color": "black"}, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"GravitySpy — 4s DeepExtractor {run_label} TL  (accuracy = {acc:.1%})", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"confusion_matrix.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nConfusion matrix saved — accuracy: {acc:.3f}  ({total} samples)")


def _run_gspy(records, clean_bg, args):
    if args.gspy_repo:
        sys.path.insert(0, args.gspy_repo)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from gravityspy.classify import classify as gspy_classify

    gspy_tmp = os.path.join(args.output_dir, "gspy_tmp")
    shutil.rmtree(gspy_tmp, ignore_errors=True)
    os.makedirs(gspy_tmp, exist_ok=True)

    rows = []
    for rec in tqdm(records, desc="GravitySpy classify"):
        ifo      = rec["ifo"]
        residual = rec["residual"]
        bg_white, bg_t0, bg_gps_center = clean_bg[ifo]

        injected = bg_white.copy()
        c, h = len(injected) // 2, LEN_4S // 2
        injected[c - h : c + h] += residual

        ts = TimeSeries(injected, t0=bg_t0, sample_rate=SAMPLE_RATE, name=ifo)
        try:
            result = gspy_classify(
                event_time=bg_gps_center,
                channel_name=f"{ifo}:GDS-CALIB_STRAIN",
                path_to_cnn=args.gspy_model,
                timeseries=ts,
                plot_directory=gspy_tmp,
            )
            pred = result["ml_label"].value[0]
            conf = float(result["ml_confidence"].value[0])
        except Exception as e:
            print(f"  GravitySpy error {ifo} {rec['gps']:.3f}: {e}")
            pred, conf = "Error", 0.0

        rows.append({
            "true_label": rec["true_label"], "pred_label": pred,
            "confidence": conf, "ifo": ifo, "gps": rec["gps"],
        })

    df_res = pd.DataFrame(rows)
    csv_out = os.path.join(args.output_dir, "gspy_results.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")
    _plot_confusion(df_res, args.output_dir, args.run_label)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-label",     default="O3",
                        help="Run period label, e.g. O3 or O4. Used in the confusion-matrix "
                             "title and to pick clean-GPS defaults (O3 only).")
    parser.add_argument("--csv",           nargs="+", default=None,
                        help="High-confidence GravitySpy CSV(s) to sample from. "
                             "Defaults to the O3a+O3b CSVs when --run-label O3, otherwise required.")
    parser.add_argument("--ifo",           choices=["H1", "L1"], default="H1",
                        help="Only sample glitches from this IFO. There is exactly one clean "
                             "injection background per run, vetted quiet for one specific IFO — "
                             "using it for the other IFO's glitches would inject into an "
                             "unverified (possibly not actually clean) segment.")
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--scaler",        required=True)
    parser.add_argument("--gspy-model",    default=None,
                        help="Path to sidd-cqg-paper-O3-model.h5. Required unless --skip-gspy.")
    parser.add_argument("--gspy-repo",     default=None,
                        help="Optional path to GravitySpy repo clone (prepended to sys.path).")
    parser.add_argument("--data-source",   choices=["open", "cit"], default="open",
                        help="'open' = TimeSeries.fetch_open_data (public GWOSC API, works "
                             "anywhere). 'cit' = gwdatafind + TimeSeries.read from local frame "
                             "files (H1_HOFT_C00/L1_HOFT_C00) — much faster per-glitch, but only "
                             "works on LIGO Data Grid machines like CIT.")
    parser.add_argument("--clean-gps-h1",  type=float, default=None,
                        help="GPS centre of a clean H1 noise segment for injection. Required "
                             "when --ifo H1, unless --run-label O3 (uses the vetted H1 default).")
    parser.add_argument("--clean-gps-l1",  type=float, default=None,
                        help="GPS centre of a clean L1 noise segment for injection. Required "
                             "when --ifo L1 — there is no default (no L1 segment has been "
                             "vetted quiet yet).")
    parser.add_argument("--output-dir",    default="evaluation/gspy_4s_o3")
    parser.add_argument("--n-per-class",   type=int,   default=10)
    parser.add_argument("--n-passes",      type=int,   default=50)
    parser.add_argument("--dropout-p",     type=float, default=0.1)
    parser.add_argument("--min-snr",       type=float, default=15.0)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--device",        default=None)
    parser.add_argument("--skip-gspy",     action="store_true",
                        help="Skip GravitySpy classification; just produce DE plots.")
    parser.add_argument("--gspy-only",     action="store_true",
                        help="Skip DE inference; load residuals.npz from --output-dir and classify.")
    args = parser.parse_args()

    if args.run_label == "O3":
        if args.csv is None:
            args.csv = ["data_o3a_high_confidence.csv", "data_o3b_high_confidence.csv"]
        if args.ifo == "H1" and args.clean_gps_h1 is None:
            args.clean_gps_h1 = DEFAULT_CLEAN_GPS_H1
    elif args.csv is None and not args.gspy_only:
        parser.error(f"--run-label {args.run_label} requires --csv (no default outside O3).")

    needs_clean_gps = args.gspy_only or not args.skip_gspy
    if needs_clean_gps:
        clean_gps_arg = "--clean-gps-h1" if args.ifo == "H1" else "--clean-gps-l1"
        clean_gps_val = args.clean_gps_h1 if args.ifo == "H1" else args.clean_gps_l1
        if clean_gps_val is None:
            parser.error(
                f"{clean_gps_arg} is required for --ifo {args.ifo} "
                f"(the only built-in default, DEFAULT_CLEAN_GPS_H1, is H1-only and O3-only)."
            )

    if args.gspy_model is None and not args.skip_gspy:
        parser.error("--gspy-model is required unless --skip-gspy is set.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    warnings.filterwarnings("ignore")
    for _log in ["gwpy", "astropy", "gravityspy", "tensorflow"]:
        logging.getLogger(_log).setLevel(logging.ERROR)

    # ── gspy-only: load saved residuals and skip straight to classification ──
    if args.gspy_only:
        residuals_path = os.path.join(args.output_dir, "residuals.npz")
        if not os.path.exists(residuals_path):
            raise FileNotFoundError(
                f"--gspy-only requires {residuals_path} — run without --gspy-only first."
            )
        data = np.load(residuals_path, allow_pickle=True)
        records = [
            {"true_label": str(t), "ifo": str(i), "gps": float(g), "residual": r}
            for t, i, g, r in zip(
                data["true_label"], data["ifo"], data["gps"], data["residual"]
            )
        ]
        print(f"Loaded {len(records)} residuals from {residuals_path}")
        stale_ifos = {r["ifo"] for r in records} - {args.ifo}
        if stale_ifos:
            raise ValueError(
                f"{residuals_path} contains residuals for {stale_ifos}, but --ifo is "
                f"{args.ifo!r} — the saved clean-GPS backgrounds only apply to one IFO. "
                f"Re-run without --gspy-only to regenerate residuals for --ifo {args.ifo}."
            )
        # Still need the clean background for injection
        print("\nFetching clean injection background...")
        clean_gps = {"H1": args.clean_gps_h1, "L1": args.clean_gps_l1}[args.ifo]
        bg_white, _central_4s, bg_t0 = _fetch_whitened(
            args.ifo, clean_gps, data_source=args.data_source
        )
        clean_bg = {args.ifo: (bg_white, bg_t0, clean_gps)}
        # Jump straight to GravitySpy
        _run_gspy(records, clean_bg, args)
        return

    # ── 1. Sample selection ──────────────────────────────────────────────────
    print("\nLoading CSVs...")
    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    df = df[
        df["label"].isin(LABEL_ORDER)
        & (df["ifo"] == args.ifo)
        & (df["snr"] >= args.min_snr)
    ].reset_index(drop=True)

    samples_by_class = {}
    for cls in LABEL_ORDER:
        sub = df[df["label"] == cls]
        n   = min(args.n_per_class, len(sub))
        if n < args.n_per_class:
            print(f"  WARNING: only {n} samples for {cls}")
        idx = rng.choice(len(sub), size=n, replace=False)
        samples_by_class[cls] = sub.iloc[sorted(idx)].reset_index(drop=True)
        print(f"  {cls}: {n}  {samples_by_class[cls]['ifo'].value_counts().to_dict()}")

    # ── 2. Load model + scaler ───────────────────────────────────────────────
    print(f"\nLoading model: {args.checkpoint}")
    model = _load_model(args.checkpoint, args.dropout_p, device)
    with open(args.scaler, "rb") as fh:
        scaler = pickle.load(fh)
    window   = torch.hann_window(WIN_LENGTH).to(device)
    istft_fn = functools.partial(
        apply_istft, n_fft=N_FFT, hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH, window=window,
    )

    # ── 3. Extract residual glitches + make 1-per-class plots ───────────────
    records   = []   # {true_label, ifo, gps, residual}
    plot_done = set()

    for cls in LABEL_ORDER:
        print(f"\n── {cls} ──")
        for _, row in tqdm(samples_by_class[cls].iterrows(),
                           total=len(samples_by_class[cls]), desc=cls):
            gps = float(row["GPStime"])
            ifo = row["ifo"]

            try:
                _, whitened_4s, _ = _fetch_whitened(ifo, gps, data_source=args.data_source)
            except Exception as e:
                print(f"  SKIP {ifo} GPS={gps:.3f}: {e}")
                continue

            with torch.no_grad():
                background, residual = _extract_glitch(
                    model, scaler, whitened_4s, window, istft_fn, device, args.n_passes
                )

            if cls not in plot_done:
                tag = f"{cls}  |  {ifo}  |  GPS {gps:.3f}"
                # Time domain: input + residual overlay to check alignment
                _plot_timeseries(
                    whitened_4s, residual,
                    os.path.join(args.output_dir, f"timeseries_{cls}.png"), tag,
                )
                # Q-scan input: should show the glitch (yellow excess power)
                _plot_qscan(
                    whitened_4s,
                    os.path.join(args.output_dir, f"qscan_input_{cls}.png"),
                    f"{cls} — whitened input  |  {ifo}  |  GPS {gps:.3f}",
                )
                # Q-scan background: input with glitch removed — yellow parts should be gone
                _plot_qscan(
                    background,
                    os.path.join(args.output_dir, f"qscan_cleaned_{cls}.png"),
                    f"{cls} — DeepExtractor cleaned  |  {ifo}  |  GPS {gps:.3f}",
                )
                plot_done.add(cls)

            records.append({"true_label": cls, "ifo": ifo, "gps": gps, "residual": residual})

    print(f"\n{len(records)} residuals extracted.")

    # Always save residuals so GravitySpy can be re-run without re-fetching
    residuals_path = os.path.join(args.output_dir, "residuals.npz")
    np.savez(
        residuals_path,
        true_label = np.array([r["true_label"] for r in records]),
        ifo        = np.array([r["ifo"]        for r in records]),
        gps        = np.array([r["gps"]        for r in records]),
        residual   = np.stack([r["residual"]   for r in records]),  # (N, LEN_4S)
    )
    print(f"Saved residuals → {residuals_path}")

    # ── 4. GravitySpy: inject residual into clean noise, classify ───────────
    if args.skip_gspy:
        print("Skipping GravitySpy (--skip-gspy).")
        return

    print("\nFetching clean injection background...")
    gps_c = {"H1": args.clean_gps_h1, "L1": args.clean_gps_l1}[args.ifo]
    print(f"  {args.ifo} @ GPS {gps_c}")
    bg_white, _central_4s, bg_t0 = _fetch_whitened(args.ifo, gps_c, data_source=args.data_source)
    clean_bg = {args.ifo: (bg_white, bg_t0, gps_c)}

    _run_gspy(records, clean_bg, args)


if __name__ == "__main__":
    main()
