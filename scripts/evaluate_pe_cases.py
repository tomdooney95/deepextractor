"""
PE case evaluation pipeline for DeepExtractor signal+glitch separation.

For each GW event, generates N independent realisations of:
  Case 1 — ground truth  (bilby noise + IMRPhenomXPHM signal)
  Case 2 — glitchy input (+ gengli glitch injected at merger)
  Case 3 — DeepExtractor separation (predicted signal / background / glitch)

Also saves `deglitched_h1/l1 = glitchy - predicted_glitch` for your
colleague's PE runs.

Pickle output structure
-----------------------
  results[event_name] = list of N dicts, each containing all three cases
  plus per-sample mismatch values.

Usage
-----
    python scripts/evaluate_pe_cases.py \\
        --checkpoint checkpoints/best.pth.tar \\
        --scaler data/standard_scaler.pkl \\
        --out PE_results/ \\
        --n-per-event 10

    # Single realisation per event (notebook-equivalent):
    python scripts/evaluate_pe_cases.py \\
        --checkpoint checkpoints/best.pth.tar \\
        --scaler data/standard_scaler.pkl \\
        --out PE_results/ \\
        --n-per-event 1
"""

import argparse
import pickle
import random
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import bilby
from pycbc.filter.matchedfilter import match
from pycbc.types import TimeSeries as PyCBCTimeSeries

from deepextractor.models import UNET1D
from deepextractor.utils.signal import whitened_snr_scaling

bilby.core.utils.setup_logger(log_level="warning")

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 4096
T           = 4.0
LENGTH      = int(T * SAMPLE_RATE)   # 16384
T_INJ       = 3.5                    # merger time within the 4s window
TIME_AXIS   = np.linspace(0, T, LENGTH, endpoint=False)

# bilby's whitened_time_domain_strain is unit-variance; whitened_snr_scaling's
# flat-PSD formula assumes the SAMPLE_RATE/2-variance convention. Previously
# corrected with a fitted constant (31.970149253731343) that under-corrected
# by ~40%; replaced with the exact analytic conversion.
BILBY_SNR_NORM = np.sqrt(SAMPLE_RATE / 2)

EVENTS = {
    # O1
    "GW150914": dict(
        mass_1=36.0, mass_2=29.0,
        a_1=0.11, a_2=0.09,
        tilt_1=1.40, tilt_2=1.30,
        phi_12=1.00, phi_jl=2.20,
        luminosity_distance=450,
        theta_jn=0.50, psi=0.00, phase=0.00,
        geocent_time=1126259642.413,
        ra=1.50, dec=0.50,
    ),
    # O3a
    "GW190412": dict(
        # Asymmetric mass ratio; first clear detection of higher modes (Abbott+2020 PRL)
        mass_1=30.1, mass_2=8.3,
        a_1=0.28, a_2=0.02,
        tilt_1=0.56, tilt_2=1.50,
        phi_12=1.00, phi_jl=0.20,
        luminosity_distance=740,
        theta_jn=0.60, psi=0.30, phase=0.50,
        geocent_time=1239082262.2,
        ra=1.71, dec=0.73,
    ),
    "GW190521": dict(
        # Highest-mass BBH; possible IMR signal (Abbott+2020 PRL)
        mass_1=85.0, mass_2=66.0,
        a_1=0.69, a_2=0.73,
        tilt_1=0.52, tilt_2=1.06,
        phi_12=5.56, phi_jl=0.41,
        luminosity_distance=3931,
        theta_jn=0.63, psi=1.79, phase=0.71,
        geocent_time=1242442967.44,
        ra=4.37, dec=0.91,
    ),
    "GW190828": dict(
        # Comparable-mass BBH, moderate distance (GWTC-2)
        mass_1=32.7, mass_2=26.4,
        a_1=0.07, a_2=0.05,
        tilt_1=1.40, tilt_2=1.50,
        phi_12=1.00, phi_jl=1.00,
        luminosity_distance=1813,
        theta_jn=0.90, psi=0.70, phase=1.50,
        geocent_time=1251009263.0,
        ra=0.50, dec=-0.70,
    ),
    # O3b
    "GW191204": dict(
        # Clean, moderate-mass BBH (GWTC-3)
        mass_1=11.9, mass_2=8.2,
        a_1=0.10, a_2=0.05,
        tilt_1=1.40, tilt_2=1.50,
        phi_12=1.00, phi_jl=1.00,
        luminosity_distance=630,
        theta_jn=0.50, psi=0.50, phase=1.00,
        geocent_time=1259514990.0,
        ra=0.79, dec=-0.34,
    ),
    "GW200129": dict(
        # Precessing BBH with strong in-band glitch in L1 (GWTC-3)
        mass_1=34.5, mass_2=29.0,
        a_1=0.90, a_2=0.75,
        tilt_1=0.79, tilt_2=1.26,
        phi_12=2.56, phi_jl=0.28,
        luminosity_distance=985,
        theta_jn=2.51, psi=0.37, phase=3.00,
        geocent_time=1264316116.44,
        ra=2.13, dec=-1.23,
    ),
    "GW200224": dict(
        # High-mass comparable-mass BBH (GWTC-3)
        mass_1=40.0, mass_2=33.0,
        a_1=0.05, a_2=0.31,
        tilt_1=1.05, tilt_2=2.00,
        phi_12=0.83, phi_jl=2.37,
        luminosity_distance=1575,
        theta_jn=0.55, psi=1.75, phase=4.52,
        geocent_time=1269363521.74,
        ra=3.68, dec=0.72,
    ),
    "GW191109": dict(
        # High-mass precessing BBH with anti-aligned spins (GWTC-3)
        mass_1=65.0, mass_2=47.0,
        a_1=0.90, a_2=0.76,
        tilt_1=1.55, tilt_2=0.93,
        phi_12=0.40, phi_jl=4.10,
        luminosity_distance=1800,
        theta_jn=0.70, psi=0.50, phase=1.40,
        geocent_time=1257296855.0,
        ra=3.52, dec=0.55,
    ),
    "GW200225": dict(
        # Moderate-mass BBH, low-spin (GWTC-3)
        mass_1=19.3, mass_2=13.8,
        a_1=0.05, a_2=0.03,
        tilt_1=1.40, tilt_2=1.50,
        phi_12=1.00, phi_jl=1.00,
        luminosity_distance=1170,
        theta_jn=0.80, psi=0.60, phase=2.00,
        geocent_time=1269391228.0,
        ra=3.00, dec=-0.50,
    ),
    # O4a
    "GW231028": dict(
        # High-mass BBH (GWTC-4); m1=95, m2=58 → ~0.1s in-band
        mass_1=95.0, mass_2=58.0,
        a_1=0.05, a_2=0.03,
        tilt_1=1.50, tilt_2=1.50,
        phi_12=1.00, phi_jl=1.00,
        luminosity_distance=4100,
        theta_jn=0.80, psi=0.50, phase=1.00,
        geocent_time=1382542404.0,
        ra=1.80, dec=0.30,
    ),
    "GW231226": dict(
        # Comparable-mass BBH, chi_eff=-0.09 (GWTC-4)
        mass_1=40.1, mass_2=35.0,
        a_1=0.10, a_2=0.05,
        tilt_1=2.50, tilt_2=2.50,
        phi_12=1.00, phi_jl=1.00,
        luminosity_distance=1180,
        theta_jn=0.90, psi=0.70, phase=2.00,
        geocent_time=1387620938.3,
        ra=2.50, dec=-0.20,
    ),
    "GW240104": dict(
        # Comparable-mass BBH, chi_eff=0.09 (GWTC-4)
        mass_1=42.3, mass_2=32.1,
        a_1=0.10, a_2=0.05,
        tilt_1=0.50, tilt_2=0.50,
        phi_12=1.00, phi_jl=1.00,
        luminosity_distance=1910,
        theta_jn=0.70, psi=0.40, phase=1.50,
        geocent_time=1388422190.6,
        ra=1.20, dec=0.60,
    ),
    # O4b
    "GW241110": dict(
        # Asymmetric-mass BBH, chi_eff=-0.28 (GWTC-4)
        mass_1=17.2, mass_2=7.7,
        a_1=0.40, a_2=0.10,
        tilt_1=2.80, tilt_2=2.00,
        phi_12=1.00, phi_jl=1.00,
        luminosity_distance=736,
        theta_jn=1.00, psi=0.80, phase=0.50,
        geocent_time=1415277701.7,
        ra=3.10, dec=-0.40,
    ),
}

WAVEFORM_ARGUMENTS = dict(
    waveform_approximant="IMRPhenomXPHM",
    reference_frequency=50,
    minimum_frequency=20,
)

# ── Data generation ───────────────────────────────────────────────────────────

def generate_bilby_example(event_name: str, params: dict) -> dict:
    """Generate whitened bilby noise + injected GW signal for one event."""
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.sampling_frequency = SAMPLE_RATE
        ifo.duration = T

    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=T,
        sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameters=params,
        waveform_arguments=WAVEFORM_ARGUMENTS,
    )

    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLE_RATE,
        duration=T,
        start_time=params["geocent_time"] - T_INJ,
    )

    whitened_noise_h1 = ifos[0].whitened_time_domain_strain.copy()
    whitened_noise_l1 = ifos[1].whitened_time_domain_strain.copy()

    ifos.inject_signal(waveform_generator=wfg, parameters=params, raise_error=False)

    whitened_h1 = ifos[0].whitened_time_domain_strain.copy()
    whitened_l1 = ifos[1].whitened_time_domain_strain.copy()

    signal_h1 = whitened_h1 - whitened_noise_h1
    signal_l1 = whitened_l1 - whitened_noise_l1

    fd_signal = wfg.frequency_domain_strain(params)
    fd_h1 = ifos[0].get_detector_response(waveform_polarizations=fd_signal, parameters=params)
    fd_l1 = ifos[1].get_detector_response(waveform_polarizations=fd_signal, parameters=params)
    snr_h1 = float(np.abs(ifos[0].optimal_snr_squared(fd_h1) ** 0.5))
    snr_l1 = float(np.abs(ifos[1].optimal_snr_squared(fd_l1) ** 0.5))

    return {
        "event":            event_name,
        "injection_parameters": params,
        "snr_h1":           snr_h1,
        "snr_l1":           snr_l1,
        "whitened_data_h1": whitened_h1,       # signal + noise
        "whitened_data_l1": whitened_l1,
        "background_h1":    whitened_noise_h1, # noise only
        "background_l1":    whitened_noise_l1,
        "signal_h1":        signal_h1,         # signal only
        "signal_l1":        signal_l1,
    }


def inject_gengli_glitch(ex: dict, inject_h1: bool | None = None) -> dict:
    """Inject a gengli glitch at the SNR of the louder detector."""
    import gengli

    if inject_h1 is None:
        inject_h1 = random.random() < 0.5

    ifo_str = "H1" if inject_h1 else "L1"
    g = gengli.glitch_generator(ifo_str)
    raw_glitch = np.array(g.get_glitch()).squeeze()
    raw_glitch = raw_glitch - raw_glitch.mean()

    glitch_snr = max(ex["snr_h1"], ex["snr_l1"]) / BILBY_SNR_NORM
    glitch = whitened_snr_scaling(raw_glitch, snr=glitch_snr)

    glitchy_h1 = ex["whitened_data_h1"].copy()
    glitchy_l1 = ex["whitened_data_l1"].copy()
    target = glitchy_h1 if inject_h1 else glitchy_l1

    len_g   = len(glitch)
    id_start = int(T_INJ * SAMPLE_RATE) - len_g // 2
    id_start = max(0, min(id_start, LENGTH - len_g))
    target[id_start:id_start + len_g] += glitch

    return {
        **ex,
        "inject_h1":      inject_h1,
        "glitchy_h1":     glitchy_h1,
        "glitchy_l1":     glitchy_l1,
        "true_glitch_h1": glitchy_h1 - ex["whitened_data_h1"],
        "true_glitch_l1": glitchy_l1 - ex["whitened_data_l1"],
        "glitch_snr":     glitch_snr * BILBY_SNR_NORM,
    }


# ── Model inference ───────────────────────────────────────────────────────────

def run_separator(ex: dict, model, scaler, device) -> dict:
    """Run DeepExtractor and compute all three output cases."""
    x = np.stack([ex["glitchy_h1"], ex["glitchy_l1"]])[np.newaxis].astype(np.float32)
    x_scaled = scaler.transform(x.reshape(-1, 1)).reshape(x.shape)

    with torch.no_grad():
        out = model(torch.tensor(x_scaled).to(device))[0].cpu().numpy()  # (4, T)

    pred_h1_bg  = out[0]
    pred_l1_bg  = out[1]
    pred_h1_sig = out[2]
    pred_l1_sig = out[3]

    g_hat_h1 = ex["glitchy_h1"] - pred_h1_bg - pred_h1_sig
    g_hat_l1 = ex["glitchy_l1"] - pred_l1_bg - pred_l1_sig

    # Deglitched strain = input minus predicted glitch → for PE
    deglitched_h1 = ex["glitchy_h1"] - g_hat_h1
    deglitched_l1 = ex["glitchy_l1"] - g_hat_l1

    return {
        **ex,
        "pred_h1_bg":    pred_h1_bg,
        "pred_l1_bg":    pred_l1_bg,
        "pred_h1_sig":   pred_h1_sig,
        "pred_l1_sig":   pred_l1_sig,
        "g_hat_h1":      g_hat_h1,
        "g_hat_l1":      g_hat_l1,
        "deglitched_h1": deglitched_h1,  # for PE: glitchy - predicted_glitch
        "deglitched_l1": deglitched_l1,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def calc_match(true: np.ndarray, pred: np.ndarray) -> float:
    dt = 1.0 / SAMPLE_RATE
    try:
        return float(match(
            PyCBCTimeSeries(true.astype(np.float64), delta_t=dt),
            PyCBCTimeSeries(pred.astype(np.float64), delta_t=dt),
        )[0])
    except Exception:
        return float("nan")


def compute_mismatches(ex: dict) -> dict:
    m_sig_h1 = calc_match(ex["signal_h1"], ex["pred_h1_sig"])
    m_sig_l1 = calc_match(ex["signal_l1"], ex["pred_l1_sig"])
    glitchy_ifo = "h1" if ex["inject_h1"] else "l1"
    m_glitch = calc_match(ex[f"true_glitch_{glitchy_ifo}"], ex[f"g_hat_{glitchy_ifo}"])
    return {
        "match_signal_h1":    m_sig_h1,
        "match_signal_l1":    m_sig_l1,
        "match_glitch":       m_glitch,
        "mismatch_signal_h1": (1 - m_sig_h1) * 100,
        "mismatch_signal_l1": (1 - m_sig_l1) * 100,
        "mismatch_glitch":    (1 - m_glitch)  * 100,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_separation_event(ex: dict, out_path: Path) -> None:
    """One figure per event: 3 rows × 2 cols (input / signal / glitch)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

    mm = {
        "sig_h1": ex["mismatch_signal_h1"],
        "sig_l1": ex["mismatch_signal_l1"],
        "glitch": ex["mismatch_glitch"],
    }
    glitch_det = "H1" if ex["inject_h1"] else "L1"

    # Row 0 — whitened inputs
    for col, ifo in enumerate(["H1", "L1"]):
        ifo_l = ifo.lower()
        ax = axes[0, col]
        ax.plot(TIME_AXIS, ex[f"glitchy_{ifo_l}"], color="grey",  lw=0.5, alpha=0.8, label="Input")
        ax.plot(TIME_AXIS, ex[f"signal_{ifo_l}"],  color="black", lw=0.8, ls="--", alpha=0.7, label="True GW")
        true_g = ex[f"true_glitch_{ifo_l}"]
        if np.any(true_g != 0):
            ax.plot(TIME_AXIS, true_g, color="red", lw=0.8, ls="--", alpha=0.7, label="True glitch")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(
            f"{ifo} — whitened input  "
            f"(GW SNR={ex[f'snr_{ifo_l}']:.1f},  glitch SNR={ex['glitch_snr']:.1f} in {glitch_det})",
            fontsize=9,
        )
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

    # Row 1 — reconstructed GW signal
    for col, ifo in enumerate(["H1", "L1"]):
        ifo_l = ifo.lower()
        ax = axes[1, col]
        ax.plot(TIME_AXIS, ex[f"glitchy_{ifo_l}"],  color="grey",      lw=0.4, alpha=0.4)
        ax.plot(TIME_AXIS, ex[f"pred_{ifo_l}_sig"], color="royalblue", lw=0.8, label="Predicted signal")
        ax.plot(TIME_AXIS, ex[f"signal_{ifo_l}"],   color="black",     lw=0.8, ls=":", alpha=0.8, label="True signal")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(f"{ifo} signal — MM={mm[f'sig_{ifo_l}']:.1f}%", fontsize=9, color="darkred")
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

    # Row 2 — reconstructed glitch
    for col, ifo in enumerate(["H1", "L1"]):
        ifo_l = ifo.lower()
        ax = axes[2, col]
        has_glitch = (ifo == glitch_det)
        ax.plot(TIME_AXIS, ex[f"glitchy_{ifo_l}"],     color="grey",   lw=0.4, alpha=0.4)
        ax.plot(TIME_AXIS, ex[f"g_hat_{ifo_l}"],       color="tomato", lw=0.8, label="Predicted glitch")
        ax.plot(TIME_AXIS, ex[f"true_glitch_{ifo_l}"], color="black",  lw=0.8, ls=":", alpha=0.8, label="True glitch")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        if has_glitch:
            ax.set_title(f"{ifo} glitch — MM={mm['glitch']:.1f}%", fontsize=9, color="darkred")
        else:
            ax.set_title(f"{ifo} — no glitch injected", fontsize=9, color="grey")
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(
        f"{ex['event']}  |  bilby noise + IMRPhenomXPHM + gengli glitch\n"
        f"Red dotted line = merger (t={T_INJ}s)  |  MM = mismatch (%)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_mismatch_summary(results: dict, out_path: Path) -> None:
    """Mismatch distribution per event — violin plot for N>1, scatter for N=1."""
    events = list(results.keys())
    metrics = ["mismatch_signal_h1", "mismatch_signal_l1", "mismatch_glitch"]
    labels  = ["Signal MM H1 (%)", "Signal MM L1 (%)", "Glitch MM (%)"]
    colors  = ["royalblue", "seagreen", "tomato"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        data = [[ex[metric] for ex in results[ev]] for ev in events]
        n_per = len(data[0])

        if n_per > 1:
            parts = ax.violinplot(data, positions=range(len(events)), showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.5)
        else:
            ax.scatter(range(len(events)), [d[0] for d in data], color=color, s=60, zorder=3)

        ax.set_xticks(range(len(events)))
        ax.set_xticklabels(events, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=10)
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"DeepExtractor mismatch summary  (N={len(results[events[0]])} per event)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",  required=True,   help="Path to model checkpoint (.pth.tar)")
    p.add_argument("--scaler",      default=None,    help="Path to StandardScaler pickle")
    p.add_argument("--out",         required=True,   help="Output directory")
    p.add_argument("--n-per-event", type=int, default=1,
                   help="Number of independent realisations per event (default: 1)")
    p.add_argument("--features",    nargs="+", type=int,
                   default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default=None)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device  = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = UNET1D(in_channels=2, out_channels=4, features=args.features).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded checkpoint  (epoch {ckpt.get('epoch', '?')})")

    # Load scaler
    if args.scaler:
        with open(args.scaler, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from {args.scaler}")
    else:
        warnings.warn("No --scaler provided — results will be unreliable.")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_  = np.array([0.0])
        scaler.scale_ = np.array([1.0])

    # ── Generate ──────────────────────────────────────────────────────────────
    results = {ev: [] for ev in EVENTS}
    all_examples = []  # for the time-series separation plot

    for event_name, params in EVENTS.items():
        print(f"\n{event_name}  ({args.n_per_event} realisations)")
        for i in range(args.n_per_event):
            print(f"  [{i+1}/{args.n_per_event}] generating ...", end=" ", flush=True)
            ex = generate_bilby_example(event_name, params)
            ex = inject_gengli_glitch(ex)
            ex = run_separator(ex, model, scaler, device)
            ex.update(compute_mismatches(ex))

            print(f"MM signal H1={ex['mismatch_signal_h1']:.1f}%  "
                  f"L1={ex['mismatch_signal_l1']:.1f}%  "
                  f"glitch={ex['mismatch_glitch']:.1f}%")

            results[event_name].append(ex)
            all_examples.append(ex)

    # ── Save pickle ───────────────────────────────────────────────────────────
    pkl_path = out_dir / f"pe_cases_n{args.n_per_event}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {pkl_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nSaving plots ...")
    sep_dir = out_dir / "separation"
    sep_dir.mkdir(exist_ok=True)
    for ex in all_examples:
        plot_separation_event(ex, sep_dir / f"{ex['event']}.png")
    plot_mismatch_summary(results, out_dir / f"mismatch_summary_n{args.n_per_event}.png")


if __name__ == "__main__":
    main()
