"""Regenerate the input/signal/glitch separation plot for one event from an
existing evaluate_pe_cases.py pickle, without rerunning the model.

The pickle (pe_cases_n10.pkl) already stores everything needed -- the
glitchy input, true signal/glitch, DeepExtractor's predicted signal/glitch,
and mismatch percentages -- for every realisation of every event. This
just re-draws the same figure evaluate_pe_cases.py itself saves (one
example per event), so it can be regenerated directly on CIT, confirming
it matches whatever realisation run.py's PE actually used -- without
needing the full deepextractor/torch/gengli environment, just numpy and
matplotlib.

Usage
-----
    python scripts/pe/plot_separation_check.py \\
        --pickle PE_results/simulated_separation_v1/pe_cases_n10.pkl \\
        --event GW150914 --realization 0 \\
        --out GW150914_separation_check.png

    # All realisations at once, one PNG each, plus a mismatch summary table
    # sorted best-to-worst (useful for picking a good-separation realisation
    # to use as a should-work-well PE sanity check):
    python scripts/pe/plot_separation_check.py \\
        --pickle PE_results/simulated_separation_v1/pe_cases_n10.pkl \\
        --event GW150914 --all --out-dir GW150914_realizations/
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAMPLE_RATE = 4096
T = 4.0
LENGTH = int(T * SAMPLE_RATE)
T_INJ = 3.5
TIME_AXIS = np.linspace(0, T, LENGTH, endpoint=False)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pickle", required=True, help="pe_cases_n*.pkl from evaluate_pe_cases.py")
    p.add_argument("--event", required=True, help="Event name, e.g. GW150914")
    p.add_argument("--realization", type=int, default=0,
                    help="Which realisation to plot (default 0 -- same one run.py's PE uses). Ignored with --all.")
    p.add_argument("--out", help="Output image path (single-realisation mode)")
    p.add_argument("--all", action="store_true",
                    help="Plot every realisation for --event, one PNG each, plus a sorted mismatch summary table")
    p.add_argument("--out-dir", help="Output directory for --all mode")
    args = p.parse_args()
    if args.all and not args.out_dir:
        p.error("--all requires --out-dir")
    if not args.all and not args.out:
        p.error("--out is required unless --all is given")
    return args


def plot_separation_event(ex: dict, out_path: Path) -> None:
    """One figure per event: 3 rows x 2 cols (input / signal / glitch).
    Identical to evaluate_pe_cases.py's own function of the same name."""
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
            f"{ifo} — whitened input  "
            f"(GW SNR={ex[f'snr_{ifo_l}']:.1f},  glitch SNR={ex['glitch_snr']:.1f} in {glitch_det})",
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
        ax.set_title(f"{ifo} signal — MM={mm[f'sig_{ifo_l}']:.1f}%", fontsize=9, color="darkred")
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
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    with open(args.pickle, "rb") as f:
        results = pickle.load(f)

    if args.event not in results:
        raise SystemExit(f"Event {args.event!r} not found. Available: {list(results.keys())}")

    realizations = results[args.event]

    if not args.all:
        if args.realization >= len(realizations):
            raise SystemExit(f"Only {len(realizations)} realisations for {args.event}, asked for index {args.realization}")
        plot_separation_event(realizations[args.realization], Path(args.out))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, ex in enumerate(realizations):
        plot_separation_event(ex, out_dir / f"{args.event}_r{i}.png")
        glitch_det = "H1" if ex["inject_h1"] else "L1"
        mean_mm = np.mean([ex["mismatch_signal_h1"], ex["mismatch_signal_l1"], ex["mismatch_glitch"]])
        rows.append((i, ex["mismatch_signal_h1"], ex["mismatch_signal_l1"], ex["mismatch_glitch"], glitch_det, mean_mm))

    rows.sort(key=lambda r: r[-1])  # best (lowest mean mismatch) first

    print(f"\n{args.event} -- {len(realizations)} realisations, sorted best (lowest mean mismatch) to worst:")
    print(f"{'r':>3}  {'sig H1 %':>9}  {'sig L1 %':>9}  {'glitch %':>9}  {'glitch in':>9}  {'mean %':>7}")
    for i, mm_h1, mm_l1, mm_g, det, mean_mm in rows:
        print(f"{i:>3}  {mm_h1:>9.1f}  {mm_l1:>9.1f}  {mm_g:>9.1f}  {det:>9}  {mean_mm:>7.1f}")
    print(f"\nBest candidate: realization {rows[0][0]} (mean mismatch {rows[0][-1]:.1f}%)")


if __name__ == "__main__":
    main()
