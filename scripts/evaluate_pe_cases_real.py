"""
PE case evaluation for DeepExtractor on REAL O3/O4 holdout noise.

Same methodology as evaluate_pe_cases.py -- real published GW event
parameters (EVENTS), a gengli glitch injected at merger, PyCBC mismatch,
the same plots -- but the noise comes from the real O3/O4 test-holdout
pool (generate_real_separation_data.py's context-level test split)
instead of bilby-simulated noise. The injected signal is whitened via
GWpy's .whiten() against each draw's matched real PSD (matching how the
real noise itself was whitened), not bilby's own whitening -- the same
reasoning as generate_real_noise_samples.py.

Unlike the training/val/test shards in real_separation_data/, this
script does NOT use SNR-sampling -- each event's signal uses its own
real, published luminosity_distance (and every other parameter)
unchanged, exactly like evaluate_pe_cases.py does for the simulated
case, so results stay comparable between the two.

The test split is reconstructed from the raw real-noise pool rather
than reading the already-generated test_holdout/*.h5 shards, since
those were built with randomly-drawn parameters for training-style
coverage, not these specific real events. --split-seed must match
whatever seed generate_real_separation_data.py was actually run with
(default 1337, its own default) -- otherwise this would reconstruct a
*different* train/val/test partition than what the model was actually
trained on, risking silent train/test leakage.

Usage
-----
    python scripts/evaluate_pe_cases_real.py \\
        --run O3 --backgrounds-dir . --psd-dir segment_psds/ \\
        --checkpoint checkpoints/o3_finetune/checkpoint_best.pth.tar \\
        --scaler scalers/o3_finetune_scaler.pkl \\
        --out PE_results_O3_o3_finetune/ --n-per-event 10
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import bilby
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

from deepextractor.models import UNET1D

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_pe_cases import (  # noqa: E402
    EVENTS, SAMPLE_RATE, T_INJ, TIME_AXIS, WAVEFORM_ARGUMENTS,
    compute_mismatches, inject_gengli_glitch, plot_mismatch_summary, run_separator,
)
from generate_real_noise_samples import MAX_FILTER_DUR, REAL_NOISE_SLICE_START  # noqa: E402
from generate_real_separation_data import load_pooled_run, split_pool_by_context  # noqa: E402

bilby.core.utils.setup_logger(log_level="warning")

LENGTH = int(4.0 * SAMPLE_RATE)
SIGNAL_WINDOW_DURATION = 4.0 + 2 * MAX_FILTER_DUR  # 8.0s, matches the real-noise pipeline

# ── Data generation ───────────────────────────────────────────────────────────

def generate_real_example(event_name: str, params: dict, pool: dict, test_indices: np.ndarray, rng) -> dict:
    """Real O3/O4 holdout noise (H1/L1-pooled, same positional-slot convention
    as generate_real_separation_data.py) + this event's real parameters,
    unmodified -- no SNR rescaling, unlike the training data generator.
    """
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.minimum_frequency = 20

    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=SIGNAL_WINDOW_DURATION, sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=WAVEFORM_ARGUMENTS,
    )

    slot_positions = rng.choice(test_indices, size=2, replace=True)
    noise, asds, gps_used = {}, {}, {}
    for ifo_name, pos in zip(["H1", "L1"], slot_positions):
        raw = pool["samples"][pos]
        noise[ifo_name] = raw[REAL_NOISE_SLICE_START:REAL_NOISE_SLICE_START + LENGTH].astype(np.float64)
        gps = float(pool["gps_starts"][pos])
        asds[ifo_name] = FrequencySeries(
            np.sqrt(pool["psd_by_gps"][gps]), frequencies=pool["psd_freqs"],
        )
        gps_used[ifo_name] = gps

    start_time = params["geocent_time"] - (MAX_FILTER_DUR + T_INJ)
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=SAMPLE_RATE, duration=SIGNAL_WINDOW_DURATION, start_time=start_time,
    )
    # Point each detector's PSD at its own matched real ASD before injecting,
    # so the SNR report below reflects the real noise this signal is embedded
    # in -- same reasoning as generate_real_noise_samples.py's sample_snr mode,
    # just for reporting here rather than rescaling.
    for ifo in ifos:
        asd = asds[ifo.name]
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=np.asarray(asd.frequencies.value, dtype=np.float64),
            asd_array=np.asarray(asd.value, dtype=np.float64),
        )
    ifos.inject_signal(waveform_generator=wfg, parameters=params, raise_error=False)

    pad = int(MAX_FILTER_DUR * SAMPLE_RATE)
    signal = {}
    for ifo in ifos:
        colored = np.asarray(ifo.strain_data.time_domain_strain, dtype=np.float64)
        ts = TimeSeries(colored, sample_rate=SAMPLE_RATE, t0=start_time)
        whitened = ts.whiten(asd=asds[ifo.name], highpass=10.0)
        signal[ifo.name] = np.asarray(whitened.value, dtype=np.float64)[pad:-pad]

    fd_signal = wfg.frequency_domain_strain(params)
    snr = {}
    for ifo in ifos:
        fd_resp = ifo.get_detector_response(waveform_polarizations=fd_signal, parameters=params)
        snr[ifo.name] = float(np.abs(ifo.optimal_snr_squared(fd_resp) ** 0.5))

    whitened_h1 = noise["H1"] + signal["H1"]
    whitened_l1 = noise["L1"] + signal["L1"]

    return {
        "event":            event_name,
        "injection_parameters": params,
        "snr_h1":           snr["H1"],
        "snr_l1":           snr["L1"],
        "whitened_data_h1": whitened_h1,
        "whitened_data_l1": whitened_l1,
        "background_h1":    noise["H1"],
        "background_l1":    noise["L1"],
        "signal_h1":        signal["H1"],
        "signal_l1":        signal["L1"],
        "gps_used_h1":      gps_used["H1"],
        "gps_used_l1":      gps_used["L1"],
    }

# ── Plotting (same as evaluate_pe_cases.py, title adjusted for real noise) ────

def plot_separation_event(ex: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    mm = {
        "sig_h1": ex["mismatch_signal_h1"],
        "sig_l1": ex["mismatch_signal_l1"],
        "glitch": ex["mismatch_glitch"],
    }
    glitch_det = "H1" if ex["inject_h1"] else "L1"

    for col, ifo in enumerate(["H1", "L1"]):
        ifo_l = ifo.lower()
        ax = axes[0, col]
        ax.plot(TIME_AXIS, ex[f"glitchy_{ifo_l}"], color="grey", lw=0.5, alpha=0.8, label="Input")
        ax.plot(TIME_AXIS, ex[f"signal_{ifo_l}"], color="black", lw=0.8, ls="--", alpha=0.7, label="True GW")
        true_g = ex[f"true_glitch_{ifo_l}"]
        if np.any(true_g != 0):
            ax.plot(TIME_AXIS, true_g, color="red", lw=0.8, ls="--", alpha=0.7, label="True glitch")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(
            f"{ifo} -- whitened input  (GPS context {ex[f'gps_used_{ifo_l}']:.0f}, "
            f"GW SNR={ex[f'snr_{ifo_l}']:.1f}, glitch SNR={ex['glitch_snr']:.1f} in {glitch_det})",
            fontsize=9,
        )
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

    for col, ifo in enumerate(["H1", "L1"]):
        ifo_l = ifo.lower()
        ax = axes[1, col]
        ax.plot(TIME_AXIS, ex[f"glitchy_{ifo_l}"], color="grey", lw=0.4, alpha=0.4)
        ax.plot(TIME_AXIS, ex[f"pred_{ifo_l}_sig"], color="royalblue", lw=0.8, label="Predicted signal")
        ax.plot(TIME_AXIS, ex[f"signal_{ifo_l}"], color="black", lw=0.8, ls=":", alpha=0.8, label="True signal")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(f"{ifo} signal -- MM={mm[f'sig_{ifo_l}']:.1f}%", fontsize=9, color="darkred")
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

    for col, ifo in enumerate(["H1", "L1"]):
        ifo_l = ifo.lower()
        ax = axes[2, col]
        has_glitch = (ifo == glitch_det)
        ax.plot(TIME_AXIS, ex[f"glitchy_{ifo_l}"], color="grey", lw=0.4, alpha=0.4)
        ax.plot(TIME_AXIS, ex[f"g_hat_{ifo_l}"], color="tomato", lw=0.8, label="Predicted glitch")
        ax.plot(TIME_AXIS, ex[f"true_glitch_{ifo_l}"], color="black", lw=0.8, ls=":", alpha=0.8, label="True glitch")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        if has_glitch:
            ax.set_title(f"{ifo} glitch -- MM={mm['glitch']:.1f}%", fontsize=9, color="darkred")
        else:
            ax.set_title(f"{ifo} -- no glitch injected", fontsize=9, color="grey")
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(
        f"{ex['event']}  |  real O3/O4 noise + IMRPhenomXPHM + gengli glitch\n"
        f"Red dotted line = merger (t={T_INJ}s)  |  MM = mismatch (%)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, choices=["O3", "O4"])
    p.add_argument("--backgrounds-dir", type=Path, default=Path("."))
    p.add_argument("--psd-dir", type=Path, default=Path("segment_psds"))
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth.tar)")
    p.add_argument("--scaler", required=True, help="Path to the scaler this checkpoint was trained with")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--n-per-event", type=int, default=1)
    p.add_argument("--seed", type=int, default=42, help="Evaluation-time randomness (noise draw, glitch choice)")
    p.add_argument("--split-seed", type=int, default=1337,
                   help="MUST match the seed generate_real_separation_data.py was run with, so the "
                        "reconstructed test split is identical to what the model was actually trained "
                        "against -- otherwise this risks silent train/test leakage.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and pooling {args.run} H1+L1 real noise, reconstructing test split "
          f"(split-seed={args.split_seed}) ...")
    pool = load_pooled_run(args.backgrounds_dir, args.psd_dir, args.run)
    splits = split_pool_by_context(pool["gps_starts"], seed=args.split_seed)
    test_indices = splits["test"]
    print(f"  Test pool: {len(test_indices):,} samples")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config")
    if config is not None:
        detectors, active_detectors = config["detectors"], config["active_detectors"]
        model = UNET1D(
            in_channels=config["in_channels"], out_channels=config["out_channels"],
            features=config["features"], dropout_p=config.get("dropout_p", 0.0),
            norm=config.get("norm", "bn"), num_groups=config.get("num_groups", 8),
        ).to(device)
        use_presence_flags = True
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})  "
              f"detectors={detectors} active={active_detectors}")
    else:
        detectors = active_detectors = ["H1", "L1"]
        model = UNET1D(in_channels=2, out_channels=4, features=[64, 128, 256, 512, 1024, 2048]).to(device)
        use_presence_flags = False
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})  legacy H1/L1, no presence flags")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)
    print(f"Loaded scaler from {args.scaler}")

    results = {ev: [] for ev in EVENTS}
    all_examples = []

    for event_name, params in EVENTS.items():
        print(f"\n{event_name}  ({args.n_per_event} realisations)")
        for i in range(args.n_per_event):
            print(f"  [{i+1}/{args.n_per_event}] generating ...", end=" ", flush=True)
            ex = generate_real_example(event_name, params, pool, test_indices, rng)
            ex = inject_gengli_glitch(ex)
            ex = run_separator(ex, model, scaler, device,
                                detectors=detectors, active_detectors=active_detectors,
                                use_presence_flags=use_presence_flags)
            ex.update(compute_mismatches(ex))

            print(f"MM signal H1={ex['mismatch_signal_h1']:.1f}%  "
                  f"L1={ex['mismatch_signal_l1']:.1f}%  "
                  f"glitch={ex['mismatch_glitch']:.1f}%  "
                  f"(SNR H1={ex['snr_h1']:.1f} L1={ex['snr_l1']:.1f} glitch={ex['glitch_snr']:.1f})")

            results[event_name].append(ex)
            all_examples.append(ex)

    pkl_path = out_dir / f"pe_cases_real_{args.run}_n{args.n_per_event}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {pkl_path}")

    print("\nSaving plots ...")
    sep_dir = out_dir / "separation"
    sep_dir.mkdir(exist_ok=True)
    for ex in all_examples:
        plot_separation_event(ex, sep_dir / f"{ex['event']}.png")
    plot_mismatch_summary(results, out_dir / f"mismatch_summary_real_{args.run}_n{args.n_per_event}.png")


if __name__ == "__main__":
    main()
