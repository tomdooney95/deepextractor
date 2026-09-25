"""
Prototype: generate a few multi-detector (H1/L1) signal+glitch separation
samples using *real* O3/O4 background noise and its matched per-context PSD,
instead of bilby-simulated noise from an analytic design curve. Plots every
component (noise, signal, glitch, combined) for visual sanity-checking before
this logic gets folded into the full-scale dataset generation pipeline.

Key methodological point (see the long PSD-matching discussion this came
out of): the real noise samples in backgrounds_<RUN>.pkl are already
whitened via GWpy's .whiten() -- unit variance, zero mean, by construction.
bilby's own inject_signal()/whitened_time_domain_strain machinery works in
colored (physical strain) units internally and only whitens at the very
end, using whatever PSD the Interferometer object currently has set. Those
two whitening implementations are NOT the same algorithm (GWpy's .whiten()
does FIR inverse-spectrum-truncation; a naive PSD division is a different,
if related, operation) -- so the injected signal is generated as a colored
bilby waveform, then explicitly whitened via GWpy's *own* .whiten(asd=...)
against the matched real PSD for that sample, and only then added to the
already-whitened real noise. This reuses the exact algorithm that whitened
the noise itself, rather than reimplementing whitening math by hand.

Usage
-----
    python scripts/generate_real_noise_samples.py \\
        --backgrounds-dir . --psd-dir segment_psds/ \\
        --runs O3 O4 --n-per-run 3 --out-dir real_noise_samples/
"""

import argparse
import pickle
from pathlib import Path

import bilby
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries

from deepextractor.generation.generate_separation_data import (
    DMAX_MPC, DMIN_MPC, GEOCENT_TIME, MINIMUM_FREQUENCY, REFERENCE_FREQUENCY,
    SAMPLE_RATE, WAVEFORM_APPROXIMANT, _build_dc_dl_lookup, _inject_glitch,
    _random_cbc_parameters,
)

bilby.core.utils.setup_logger(log_level="warning")

# ── Constants ─────────────────────────────────────────────────────────────────

IFOS = ["H1", "L1"]
DURATION = 4.0
LENGTH = int(DURATION * SAMPLE_RATE)                  # 16384 -- final sample length
MAX_FILTER_DUR = 2.0                                   # matches get_clean_backgrounds.py's crop-per-edge
SIGNAL_WINDOW_DURATION = DURATION + 2 * MAX_FILTER_DUR  # 8.0s -- generated pre-crop
T_INJ = 3.5                                            # merger position within the final 4s output
TIME_AXIS = np.linspace(0, DURATION, LENGTH, endpoint=False)

REAL_NOISE_SLICE_START = 8192   # middle 4s of the 8s real-noise samples (32768 -> 16384)

# Default target network-SNR range for --sample-snr mode. 5.0 sits below the
# conventional network SNR=8 confident-detection threshold, deliberately --
# the network still needs marginal/sub-threshold examples, not just clean
# detections. The loudest O3/O4 events reached roughly SNR ~20-30; SNR_MAX is
# pushed well past that deliberately, so the network also sees loud, clean
# examples of true signal structure (helps it recognize the same structure
# when it's buried in noise at low SNR) -- log-uniform(5, 150) puts ~12% of
# draws above SNR=100 and ~20% below SNR=10, so the loud tail is present but
# a minority, not the bulk of the distribution.
SNR_MIN = 5.0
SNR_MAX = 150.0

# Probability a sample gets no astrophysical signal injected at all (pure
# real noise + glitch), so the network sees genuine negative examples rather
# than only "signal present, however faint." Kept small and non-zero -- see
# the --sample-snr discussion: with SNR now sampled directly rather than
# inherited from an uninformative distance prior, most injected signals are
# meaningfully loud, so explicit no-injection cases are needed for the
# network to learn what "nothing there" looks like at all.
NO_INJ_PROB = 0.02

# ── Data loading ──────────────────────────────────────────────────────────────

# O3a/O3b and O4a/O4b are trained as single O3/O4 models (whitening against
# each context's own matched PSD makes them statistically comparable), so
# their backgrounds and PSD tables are pooled here rather than kept separate.
SUBRUNS = {"O3": ["O3a", "O3b"], "O4": ["O4a", "O4b"]}


def _load_subrun(backgrounds_dir: Path, psd_dir: Path, subrun: str, ifo: str):
    with open(backgrounds_dir / f"backgrounds_{subrun}.pkl", "rb") as f:
        bg = pickle.load(f)[subrun][ifo]
    with open(psd_dir / f"{subrun}_{ifo}_psds.pkl", "rb") as f:
        psd_data = pickle.load(f)
    return bg, psd_data


def load_run_data(backgrounds_dir: Path, psd_dir: Path, run: str, ifo: str):
    subruns = SUBRUNS.get(run, [run])
    bgs, psd_datas = zip(*(
        _load_subrun(backgrounds_dir, psd_dir, subrun, ifo) for subrun in subruns
    ))

    bg = {
        "samples": np.concatenate([b["samples"] for b in bgs], axis=0),
        "gps_starts": np.concatenate([b["gps_starts"] for b in bgs], axis=0),
    }
    psd_data = {
        "psd_freqs": psd_datas[0]["psd_freqs"],  # same fftlength/overlap for every subrun -> same grid
        "psd_gps_starts": np.concatenate([p["psd_gps_starts"] for p in psd_datas], axis=0),
        "psds": np.concatenate([p["psds"] for p in psd_datas], axis=0),
    }
    psd_index = {gps: j for j, gps in enumerate(psd_data["psd_gps_starts"])}
    return bg, psd_data, psd_index


def get_real_noise_and_asd(bg, psd_data, psd_index, rng):
    """A random 4s real-noise slice, and its matched ASD as a GWpy FrequencySeries."""
    n = bg["samples"].shape[0]
    idx = int(rng.integers(0, n))
    full = np.asarray(bg["samples"][idx], dtype=np.float64)     # (32768,) 8s whitened real noise
    noise_4s = full[REAL_NOISE_SLICE_START:REAL_NOISE_SLICE_START + LENGTH]

    gps = float(bg["gps_starts"][idx])
    j = psd_index[gps]
    asd = FrequencySeries(
        np.sqrt(np.asarray(psd_data["psds"][j], dtype=np.float64)),
        frequencies=np.asarray(psd_data["psd_freqs"], dtype=np.float64),
    )
    # The recovered PSD has df = 1/PSD_DURATION = 0.25 Hz (Welch sub-segment
    # length from the 36s context it was estimated over in get_segment_psds.py);
    # .whiten() interpolates any ASD it's handed onto the signal's own grid
    # internally, so no manual resampling is needed here.
    return noise_4s, asd, gps

# ── Signal generation ─────────────────────────────────────────────────────────

def generate_whitened_signal(ifos, wfg, rng, dc_grid, dl_grid, asds: dict, start_time: float,
                              sample_snr: bool = False, snr_min: float = SNR_MIN, snr_max: float = SNR_MAX):
    """Coherent bilby CBC signal, projected through H1+L1, each detector's copy
    whitened against its own matched real ASD (not bilby's default PSD).

    sample_snr=False (default): distance is drawn directly from
    _random_cbc_parameters' uniform-in-comoving-volume prior, which -- since
    volume scales as d^3 -- concentrates most draws near DMAX_MPC, i.e. most
    signals end up faint.

    sample_snr=True: draws a target network SNR log-uniformly between
    snr_min/snr_max (mirrors _inject_glitch's log-uniform SNR convention) and
    rescales the already-injected, already-whitened signal to hit it exactly.
    This is exact, not approximate -- strain amplitude is linear in 1/D_L, so
    SNR is linear in strain, so a flat multiplicative rescale of the output
    is equivalent to having injected at the corresponding distance in the
    first place. params['luminosity_distance'] is adjusted to match so it
    stays physically meaningful, and params['network_snr'] records the draw.
    """
    params = _random_cbc_parameters(rng, dc_grid, dl_grid, DMAX_MPC, DMIN_MPC, GEOCENT_TIME)
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=SAMPLE_RATE, duration=SIGNAL_WINDOW_DURATION, start_time=start_time,
    )
    if sample_snr:
        # bilby's optimal_SNR (read below) is computed against whatever PSD is
        # attached to the Interferometer -- by default that's a generic design
        # curve (aLIGO_O4_high_asd.txt for both H1/L1, wrong for O3 and not
        # matched to this specific context either way), completely disconnected
        # from the real ASD this signal is actually whitened against. Point it
        # at the matched real ASD so "target SNR" means the true SNR relative
        # to the real noise this sample gets embedded in.
        for ifo in ifos:
            asd = asds[ifo.name]
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                frequency_array=np.asarray(asd.frequencies.value, dtype=np.float64),
                asd_array=np.asarray(asd.value, dtype=np.float64),
            )
    ifos.inject_signal(waveform_generator=wfg, parameters=params)

    scale = 1.0
    if sample_snr:
        network_snr = float(np.sqrt(sum(ifo.meta_data["optimal_SNR"] ** 2 for ifo in ifos)))
        if np.isfinite(network_snr) and network_snr > 0:
            target_snr = float(np.exp(rng.uniform(np.log(snr_min), np.log(snr_max))))
            scale = target_snr / network_snr
            params["luminosity_distance"] /= scale
            params["network_snr"] = target_snr
        else:
            params["network_snr"] = network_snr  # degenerate draw -- left unscaled, flagged in params

    pad = int(MAX_FILTER_DUR * SAMPLE_RATE)
    signal = {}
    for ifo in ifos:
        colored = np.asarray(ifo.strain_data.time_domain_strain, dtype=np.float64)
        ts = TimeSeries(colored, sample_rate=SAMPLE_RATE, t0=start_time)
        whitened = ts.whiten(asd=asds[ifo.name], highpass=10.0)
        signal[ifo.name] = scale * np.asarray(whitened.value, dtype=np.float64)[pad:-pad]
    return signal, params

# ── Per-sample assembly ───────────────────────────────────────────────────────

def generate_one_sample(run_data: dict, ifos, wfg, rng, dc_grid, dl_grid,
                         sample_snr: bool = False, snr_min: float = SNR_MIN, snr_max: float = SNR_MAX,
                         no_inj_prob: float = NO_INJ_PROB):
    """run_data: {ifo: (bg, psd_data, psd_index)} for H1 and L1."""
    noise, asds, gps_used = {}, {}, {}
    for ifo in IFOS:
        bg, psd_data, psd_index = run_data[ifo]
        noise[ifo], asds[ifo], gps_used[ifo] = get_real_noise_and_asd(bg, psd_data, psd_index, rng)

    # Merger placed at T_INJ within the final 4s output -> within the
    # SIGNAL_WINDOW_DURATION pre-crop window that's (crop + T_INJ) in.
    start_time = GEOCENT_TIME - (MAX_FILTER_DUR + T_INJ)
    if rng.random() < no_inj_prob:
        signal = {ifo: np.zeros(LENGTH, dtype=np.float64) for ifo in IFOS}
        params = {"mass_1": float("nan"), "mass_2": float("nan"), "no_injection": True}
    else:
        signal, params = generate_whitened_signal(ifos, wfg, rng, dc_grid, dl_grid, asds, start_time,
                                                   sample_snr=sample_snr, snr_min=snr_min, snr_max=snr_max)
        params["no_injection"] = False

    noisy = {ifo: noise[ifo] + signal[ifo] for ifo in IFOS}

    # One detector gets a glitch, matching generate_separation_data.py's
    # convention (glitches are instrumental/detector-local, not coherent).
    # whitened_snr_scaling (used inside _inject_glitch) assumes a flat-PSD,
    # unit-variance whitened target -- exactly what GWpy's .whiten() also
    # produces by construction, so it's valid to use unmodified on real noise.
    glitch_ifo = IFOS[int(rng.integers(0, len(IFOS)))]
    noisy_with_glitch = dict(noisy)
    noisy_with_glitch[glitch_ifo] = noisy[glitch_ifo].copy()
    _inject_glitch(noisy_with_glitch[glitch_ifo], rng, sample_rate=SAMPLE_RATE)
    true_glitch = {
        ifo: (noisy_with_glitch[ifo] - noisy[ifo]) if ifo == glitch_ifo
        else np.zeros(LENGTH, dtype=np.float64)
        for ifo in IFOS
    }

    return {
        "noise": noise, "signal": signal, "noisy": noisy_with_glitch,
        "true_glitch": true_glitch, "glitch_ifo": glitch_ifo,
        "gps_used": gps_used, "params": params,
    }

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sample(sample: dict, run: str, idx: int, out_path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    for col, ifo in enumerate(IFOS):
        ax = axes[0, col]
        ax.plot(TIME_AXIS, sample["noisy"][ifo], color="grey", lw=0.5, alpha=0.8, label="Noisy (real noise + signal + glitch)")
        ax.plot(TIME_AXIS, sample["signal"][ifo], color="black", lw=0.8, ls="--", alpha=0.8, label="Signal")
        if ifo == sample["glitch_ifo"]:
            ax.plot(TIME_AXIS, sample["true_glitch"][ifo], color="red", lw=0.8, ls="--", alpha=0.8, label="Glitch")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(f"{ifo} -- input  (GPS of real noise context: {sample['gps_used'][ifo]:.0f})", fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

        ax = axes[1, col]
        ax.plot(TIME_AXIS, sample["noise"][ifo], color="grey", lw=0.4, alpha=0.6, label="Real noise")
        ax.plot(TIME_AXIS, sample["signal"][ifo], color="royalblue", lw=0.8, label="Injected signal (whitened vs. matched PSD)")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(f"{ifo} signal", fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

        ax = axes[2, col]
        has_glitch = ifo == sample["glitch_ifo"]
        ax.plot(TIME_AXIS, sample["noisy"][ifo], color="grey", lw=0.4, alpha=0.4)
        if has_glitch:
            ax.plot(TIME_AXIS, sample["true_glitch"][ifo], color="tomato", lw=0.8, label="Glitch")
            ax.set_title(f"{ifo} glitch", fontsize=9, color="darkred")
        else:
            ax.set_title(f"{ifo} -- no glitch injected", fontsize=9, color="grey")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Time (s)", fontsize=8)

    if sample["params"].get("no_injection", False):
        signal_info = "no signal injected (negative example)"
    else:
        m1, m2 = sample["params"]["mass_1"], sample["params"]["mass_2"]
        snr_info = ""
        if "network_snr" in sample["params"]:
            snr_info = f"  |  target network SNR={sample['params']['network_snr']:.1f}"
        signal_info = f"bilby {WAVEFORM_APPROXIMANT} signal (m1={m1:.1f}, m2={m2:.1f} Msun){snr_info}"
    fig.suptitle(f"{run}  sample {idx}  |  real O3/O4 noise + {signal_info}  |  "
                 f"glitch in {sample['glitch_ifo']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backgrounds-dir", type=Path, default=Path("."))
    p.add_argument("--psd-dir", type=Path, default=Path("segment_psds"))
    p.add_argument("--runs", nargs="+", default=["O3", "O4"],
                   help="O3/O4 pool their a/b sub-runs together (see SUBRUNS); "
                        "pass a bare sub-run name (e.g. O3a) to sample it alone.")
    p.add_argument("--n-per-run", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("real_noise_samples"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-snr", action="store_true",
                    help="Draw a target network SNR log-uniformly and rescale the injected "
                         "signal to hit it exactly, instead of drawing distance directly from "
                         "the uniform-in-comoving-volume prior (which skews heavily towards "
                         "faint, large-distance signals since volume scales as d^3).")
    p.add_argument("--snr-min", type=float, default=SNR_MIN)
    p.add_argument("--snr-max", type=float, default=SNR_MAX)
    p.add_argument("--no-inj-prob", type=float, default=NO_INJ_PROB,
                    help="Probability a sample gets no astrophysical signal injected at all "
                         "(pure real noise + glitch), for genuine negative examples.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ifos = bilby.gw.detector.InterferometerList(IFOS)
    for ifo in ifos:
        ifo.minimum_frequency = MINIMUM_FREQUENCY
    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=SIGNAL_WINDOW_DURATION, sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=dict(
            waveform_approximant=WAVEFORM_APPROXIMANT,
            reference_frequency=REFERENCE_FREQUENCY,
            minimum_frequency=MINIMUM_FREQUENCY,
        ),
    )
    dc_grid, dl_grid = _build_dc_dl_lookup(DMAX_MPC)

    for run in args.runs:
        print(f"\n{'=' * 60}\n{run}\n{'=' * 60}")
        run_data = {ifo: load_run_data(args.backgrounds_dir, args.psd_dir, run, ifo) for ifo in IFOS}

        tag = "_snr" if args.sample_snr else "_dist"
        for idx in range(args.n_per_run):
            print(f"  [{idx + 1}/{args.n_per_run}] generating ...", end=" ", flush=True)
            sample = generate_one_sample(run_data, ifos, wfg, rng, dc_grid, dl_grid,
                                          sample_snr=args.sample_snr,
                                          snr_min=args.snr_min, snr_max=args.snr_max,
                                          no_inj_prob=args.no_inj_prob)
            if sample["params"].get("no_injection", False):
                print(f"no injection (negative example)  glitch_ifo={sample['glitch_ifo']}")
            else:
                snr_msg = f" network_snr={sample['params']['network_snr']:.1f}" if args.sample_snr else ""
                print(f"m1={sample['params']['mass_1']:.1f} m2={sample['params']['mass_2']:.1f} "
                      f"glitch_ifo={sample['glitch_ifo']}{snr_msg}")
            plot_sample(sample, run, idx, args.out_dir / f"{run}_sample{idx}{tag}.png")


if __name__ == "__main__":
    main()
