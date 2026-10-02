"""Overlay control/dirty/prediction posteriors from run.py on one corner plot.

Reads three bilby *_result.json files (one per --type run.py was given) and
uses bilby.result.plot_multiple to combine them into a single corner plot,
so the effect of the glitch -- and DeepExtractor's removal of it -- on
parameter recovery is directly visible in one figure.

Meant to run interactively on ldas-pcdev2 (not via Condor): the execute
pool's sandboxed HOME is what broke run.py's own corner-plot step (see the
job.sub fix), but an interactive login shell has a real, writable HOME, so
this needs no such workaround.

Usage
-----
    python scripts/pe/plot_pe_comparison.py \\
        --control GW150914_control_outdir/GW150914_control_result.json \\
        --dirty GW150914_dirty_outdir/GW150914_dirty_result.json \\
        --prediction GW150914_prediction_outdir/GW150914_prediction_result.json \\
        --out GW150914_pe_comparison_corner.png
"""

import argparse

import bilby

DEFAULT_PARAMETERS = ["chirp_mass", "mass_ratio", "luminosity_distance", "geocent_time"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--control", required=True, help="control run's *_result.json")
    p.add_argument("--dirty", required=True, help="dirty run's *_result.json")
    p.add_argument("--prediction", required=True, help="prediction run's *_result.json")
    p.add_argument("--out", required=True, help="Output image path")
    p.add_argument("--parameters", nargs="+", default=DEFAULT_PARAMETERS,
                    help=f"Parameters to plot (default: {DEFAULT_PARAMETERS}). "
                         "Pass --parameters all 15 free params for the full corner.")
    return p.parse_args()


def main():
    args = parse_args()

    results = [
        bilby.result.read_in_result(filename=args.control),
        bilby.result.read_in_result(filename=args.dirty),
        bilby.result.read_in_result(filename=args.prediction),
    ]
    labels = ["Control (no glitch)", "Dirty (with glitch)", "Prediction (DeepExtractor)"]

    bilby.result.plot_multiple(
        results,
        filename=args.out,
        labels=labels,
        parameters=args.parameters,
        save=True,
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
