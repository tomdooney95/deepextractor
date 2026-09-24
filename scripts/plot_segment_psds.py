"""
Sanity-check the backfilled per-context PSDs (segment_psds/<RUN>_<IFO>_psds.pkl,
produced by get_segment_psds.py) by plotting a random sample of them as ASDs.

Overlays a rough aLIGO design-sensitivity curve per run for visual reference
only (O3 -> aLIGO_late, O4 -> aLIGO_O4_high) -- these are generic design
curves, not a fit to the specific run, so don't expect an exact match; they're
there to catch anything grossly wrong (wrong units, a flat/zero spectrum, a
bucket in the wrong place), not to validate absolute normalization.

Usage
-----
    python scripts/plot_segment_psds.py \\
        --psd-dir segment_psds/ --runs O3 O4 --n-samples 15 \\
        --out-dir psd_sanity_plots/
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# O3a/O3b and O4a/O4b are pooled the same way generate_real_noise_samples.py
# pools them, so "make sure the PSDs look sane" is checked at the same
# granularity they'll actually be sampled from during training.
SUBRUNS = {"O3": ["O3a", "O3b"], "O4": ["O4a", "O4b"]}

DESIGN_CURVE = {"O3": "aLIGO_late_asd.txt", "O4": "aLIGO_O4_high_asd.txt"}

IFOS = ["H1", "L1"]


def load_psd_data(psd_dir: Path, run: str, ifo: str) -> dict:
    subruns = SUBRUNS.get(run, [run])
    freqs = None
    psds = []
    gps_starts = []
    for subrun in subruns:
        with open(psd_dir / f"{subrun}_{ifo}_psds.pkl", "rb") as f:
            d = pickle.load(f)
        if freqs is None:
            freqs = np.asarray(d["psd_freqs"], dtype=np.float64)
        psds.append(np.asarray(d["psds"], dtype=np.float64))
        gps_starts.append(np.asarray(d["psd_gps_starts"], dtype=np.float64))
    return {
        "psd_freqs": freqs,
        "psds": np.concatenate(psds, axis=0),
        "psd_gps_starts": np.concatenate(gps_starts, axis=0),
    }


def load_design_asd(run: str):
    try:
        import bilby
    except ImportError:
        return None, None
    fname = DESIGN_CURVE.get(run)
    if fname is None:
        return None, None
    try:
        psd = bilby.gw.detector.PowerSpectralDensity(asd_file=fname)
        return psd.frequency_array, psd.asd_array
    except Exception:
        return None, None


def plot_run(psd_dir: Path, run: str, n_samples: int, rng: np.random.Generator, out_dir: Path):
    design_f, design_asd = load_design_asd(run)

    fig, axes = plt.subplots(1, len(IFOS), figsize=(14, 5.5))
    for ax, ifo in zip(np.atleast_1d(axes), IFOS):
        data = load_psd_data(psd_dir, run, ifo)
        freqs = data["psd_freqs"]
        psds = data["psds"]
        n_total = psds.shape[0]

        finite = np.isfinite(psds).all()
        nonpos = int(np.sum(psds <= 0))
        print(f"  {run} {ifo}: {n_total} contexts  |  all finite: {finite}  |  "
              f"non-positive values: {nonpos}  |  df={freqs[1] - freqs[0]:.3f} Hz")

        idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
        for i in idx:
            asd = np.sqrt(np.clip(psds[i], 0, None))
            ax.loglog(freqs, asd, lw=0.6, alpha=0.35, color="steelblue")
        median_asd = np.sqrt(np.median(np.clip(psds[idx], 0, None), axis=0))
        ax.loglog(freqs, median_asd, lw=1.6, color="navy",
                   label=f"median of {len(idx)} sampled contexts")

        if design_f is not None:
            ax.loglog(design_f, design_asd, lw=1.2, color="crimson", ls="--",
                       label=f"{DESIGN_CURVE[run]} (rough reference only)")

        ax.set_xlim(10, 2048)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"ASD (strain / $\sqrt{\mathrm{Hz}}$)")
        ax.set_title(f"{ifo}  --  {n_total} contexts total, {len(idx)} shown")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"{run}: sampled backfilled per-context PSDs", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"{run}_psd_sample.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--psd-dir", type=Path, default=Path("segment_psds"))
    p.add_argument("--runs", nargs="+", default=["O3", "O4"],
                   help="O3/O4 pool their a/b sub-runs (see SUBRUNS); pass a bare "
                        "sub-run name (e.g. O3a) to sample it alone.")
    p.add_argument("--n-samples", type=int, default=15,
                   help="Number of randomly sampled PSDs to plot per run/IFO panel")
    p.add_argument("--out-dir", type=Path, default=Path("psd_sanity_plots"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for run in args.runs:
        print(f"\n{'=' * 60}\n{run}\n{'=' * 60}")
        plot_run(args.psd_dir, run, args.n_samples, rng, args.out_dir)


if __name__ == "__main__":
    main()
