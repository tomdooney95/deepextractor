"""Overlay control/dirty/prediction posteriors from run.py on one corner plot.

Reads bilby *_result.json files (one per --type run.py was given) and
overlays their posteriors on a single corner plot via the `corner` package
directly (reusing one Figure across calls, one call per result, each a
different colour) -- bilby.result.plot_multiple's own truths-forwarding
turned out not to reliably render reference lines, so this calls corner.
corner() explicitly instead, where `truths` is well-defined and documented.
Any subset of control/dirty/prediction (at least 2) can be given, e.g. to
check progress before all three have finished.

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

Default parameters: chirp_mass, mass_ratio, luminosity_distance,
geocent_time, ra, dec, theta_jn (inclination, as theta_jn -- the angle
between total angular momentum and line of sight -- since priors here are
precessing-spin; pass --parameters to override, e.g. for the full 15-dim
corner or a different subset).
"""

import argparse

import bilby
import corner
import matplotlib.lines as mlines
import numpy as np

DEFAULT_PARAMETERS = [
    "chirp_mass", "mass_ratio", "luminosity_distance", "geocent_time",
    "ra", "dec", "theta_jn",
]

COLOURS = ["tab:blue", "tab:orange", "tab:green"]

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
    colours = COLOURS[:len(results)]

    # All runs share the same injected event parameters (run.py passes
    # injection_parameters into bilby.run_sampler(), which stores them on
    # the Result) -- pull truth values from whichever result has them.
    injection_parameters = next(
        (r.injection_parameters for r in results if r.injection_parameters), None
    )
    if injection_parameters is None:
        print("WARNING: no injection_parameters found on any result -- plotting without truth markers")
        truths = None
    else:
        missing = [p for p in args.parameters if p not in injection_parameters]
        if missing:
            print(f"WARNING: no injected value for {missing} -- leaving those truth markers blank")
        truths = [injection_parameters.get(p) for p in args.parameters]

    fig = None
    for i, (result, colour) in enumerate(zip(results, colours)):
        samples = np.array([result.posterior[p].values for p in args.parameters]).T
        fig = corner.corner(
            samples,
            labels=args.parameters,
            fig=fig,
            color=colour,
            truths=truths if i == 0 else None,
            truth_color="black",
            plot_datapoints=False,
            plot_density=False,
            levels=(0.68, 0.95),
            hist_kwargs=dict(density=True),
        )

    handles = [mlines.Line2D([], [], color=c, label=l) for c, l in zip(colours, labels)]
    if truths is not None:
        handles.append(mlines.Line2D([], [], color="black", label="Injected (truth)"))
    fig.legend(handles=handles, loc="upper right", fontsize=10)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out} ({', '.join(labels)})")


if __name__ == "__main__":
    main()
