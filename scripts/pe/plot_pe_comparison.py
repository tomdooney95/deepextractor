"""Overlay control/dirty/prediction posteriors from run.py on one corner plot.

Reads bilby *_result.json files (one per --type run.py was given) and uses
bilby.result.plot_multiple to combine them into a single corner plot, so
the effect of the glitch -- and DeepExtractor's removal of it -- on
parameter recovery is directly visible in one figure. Any subset of
control/dirty/prediction (at least 2) can be given, e.g. to check progress
before all three have finished.

Meant to run interactively on ldas-pcdev2 (not via Condor): the execute
pool's sandboxed HOME is what broke run.py's own corner-plot step (see the
job.sub fix), but an interactive login shell has a real, writable HOME, so
this needs no such workaround. Also run with PYTHONNOUSERSITE=1 if using
the CVMFS python -- otherwise a personal ~/.local numpy install can shadow
the env's own and break astropy's cosmology import.

Usage
-----
    python scripts/pe/plot_pe_comparison.py \\
        --control GW150914_control_outdir/GW150914_control_result.json \\
        --dirty GW150914_dirty_outdir/GW150914_dirty_result.json \\
        --prediction GW150914_prediction_outdir/GW150914_prediction_result.json \\
        --out GW150914_pe_comparison_corner.png

    # Or just a subset, e.g. before prediction has finished:
    python scripts/pe/plot_pe_comparison.py \\
        --control GW150914_control_outdir/GW150914_control_result.json \\
        --dirty GW150914_dirty_outdir/GW150914_dirty_result.json \\
        --out GW150914_control_vs_dirty_corner.png
"""

import argparse

import bilby

DEFAULT_PARAMETERS = ["chirp_mass", "mass_ratio", "luminosity_distance", "geocent_time"]

RUNS = [
    ("control", "Control (no glitch)"),
    ("dirty", "Dirty (with glitch)"),
    ("prediction", "Prediction (DeepExtractor)"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", help="control run's *_result.json")
    p.add_argument("--dirty", help="dirty run's *_result.json")
    p.add_argument("--prediction", help="prediction run's *_result.json")
    p.add_argument("--out", required=True, help="Output image path")
    p.add_argument("--parameters", nargs="+", default=DEFAULT_PARAMETERS,
                    help=f"Parameters to plot (default: {DEFAULT_PARAMETERS}). "
                         "Pass --parameters all 15 free params for the full corner.")
    return p.parse_args()


def main():
    args = parse_args()

    given = [(name, label, getattr(args, name)) for name, label in RUNS if getattr(args, name)]
    if len(given) < 2:
        raise SystemExit("Need at least two of --control/--dirty/--prediction to compare.")

    results = [bilby.result.read_in_result(filename=path) for _, _, path in given]
    labels = [label for _, label, _ in given]

    bilby.result.plot_multiple(
        results,
        filename=args.out,
        labels=labels,
        parameters=args.parameters,
        save=True,
    )
    print(f"Saved {args.out} ({', '.join(labels)})")


if __name__ == "__main__":
    main()
