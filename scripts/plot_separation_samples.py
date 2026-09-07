"""Plot example samples from generate_separation_data.py's sharded HDF5 output.

Sanity-checks the raw generated data (no model involved) before committing to
a full-scale generation run: for each of N samples, plots per detector the
noisy input, background vs. signal-only, and the recovered glitch
(noisy - background - signal_only) so you can visually confirm the signal is
coherently present across detectors and the glitch is confined to exactly one.

Reads through HDF5SeparationDataset itself, so this also exercises the same
loader that training will use.

Example:
    conda run -n deepextractor python scripts/plot_separation_samples.py \\
        --shard-dir ~/deex_timing_test --split train --n 6 --out evaluation/separation_samples
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deepextractor.data import HDF5SeparationDataset


def plot_sample(x, y, detectors, index, out_dir):
    n_det = len(detectors)
    background = y[:n_det]
    signal_only = y[n_det:]
    glitch = x - background - signal_only  # recovered by subtraction

    fig, axes = plt.subplots(n_det, 4, figsize=(19, 3 * n_det), squeeze=False)
    for row, det in enumerate(detectors):
        axes[row, 0].plot(x[row], linewidth=0.5, color="k")
        axes[row, 0].set_ylabel(det, fontsize=12, fontweight="bold")
        if row == 0:
            axes[row, 0].set_title("Noisy input (background + signal + glitch)")

        axes[row, 1].plot(background[row], linewidth=0.4, color="gray", alpha=0.7)
        if row == 0:
            axes[row, 1].set_title("Background only")

        axes[row, 2].plot(signal_only[row], linewidth=0.8, color="tab:blue")
        sig_energy = float(np.sum(signal_only[row] ** 2))
        axes[row, 2].text(
            0.02, 0.92, f"energy={sig_energy:.2e}", transform=axes[row, 2].transAxes, fontsize=8,
        )
        if row == 0:
            axes[row, 2].set_title("Signal only (own y-scale)")

        axes[row, 3].plot(glitch[row], linewidth=0.6, color="tab:red")
        glitch_energy = float(np.sum(glitch[row] ** 2))
        axes[row, 3].text(
            0.02, 0.92, f"energy={glitch_energy:.2e}", transform=axes[row, 3].transAxes, fontsize=8,
        )
        if row == 0:
            axes[row, 3].set_title("Recovered glitch (noisy - background - signal)")

    fig.suptitle(f"Sample {index}", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / f"separation_sample_{index:03d}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--detectors", nargs="+", default=["H1", "L1", "V1"])
    parser.add_argument("--n", type=int, default=6, help="Number of samples to plot.")
    parser.add_argument("--out", type=str, default="evaluation/separation_samples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = HDF5SeparationDataset(args.shard_dir, split=args.split, detectors=args.detectors)
    print(f"Dataset has {len(ds)} samples in split={args.split!r}")

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    for idx in indices:
        x, y = ds[int(idx)]
        path = plot_sample(x.numpy(), y.numpy(), args.detectors, int(idx), out_dir)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
