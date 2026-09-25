"""
Plot a few examples directly from the generated real-noise separation
dataset shards (output of generate_real_separation_data.py) -- reads what's
already on disk, no regeneration, for visual sanity-checking of the actual
data before it gets rsynced and trained on.

"Glitch" isn't stored as its own array (see HDF5SeparationDataset's
targets: background + signal_only); it's recovered the same way training
does, as noisy - background - signal_only.

Usage
-----
    python scripts/plot_generated_separation_samples.py \\
        --shard-dir real_separation_data/O3 --splits train val test \\
        --n-samples 5 --out-dir generated_sample_plots/O3/
"""

import argparse
import glob
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IFOS = ["H1", "L1"]
SAMPLE_RATE = 4096
DURATION = 4.0
LENGTH = int(DURATION * SAMPLE_RATE)
T_INJ = 3.5  # merger position within the window, fixed at generation time
TIME_AXIS = np.linspace(0, DURATION, LENGTH, endpoint=False)


def load_random_samples(shard_dir: Path, split: str, n_samples: int, rng: np.random.Generator):
    """Reads n_samples random rows for `split`. Tries shards in random order
    until enough non-empty ones are found -- shards straddling the train/val
    boundary can have zero rows for one side."""
    if split == "test":
        pattern = str(shard_dir / "test_holdout" / "separation_shard_*.h5")
    else:
        pattern = str(shard_dir / "separation_shard_*.h5")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No shards found matching {pattern}")

    suffix = "_test" if split == "test" else f"_{split}"
    for path in rng.permutation(np.array(paths, dtype=object)):
        path = Path(path)
        with h5py.File(path, "r") as f:
            n_rows = f[f"noisy_{IFOS[0]}{suffix}"].shape[0]
            if n_rows == 0:
                continue
            idx = np.sort(rng.choice(n_rows, size=min(n_samples, n_rows), replace=False))
            samples = []
            for i in idx:
                sample = {}
                for det in IFOS:
                    noisy = f[f"noisy_{det}{suffix}"][int(i)]
                    background = f[f"background_{det}{suffix}"][int(i)]
                    signal_only = f[f"signal_only_{det}{suffix}"][int(i)]
                    glitch = noisy - background - signal_only
                    sample[det] = dict(noisy=noisy, background=background,
                                        signal_only=signal_only, glitch=glitch)
                samples.append(sample)
            if len(samples) >= n_samples:
                return samples[:n_samples], path
    raise RuntimeError(f"Could not find {n_samples} non-empty '{split}' rows across {len(paths)} shard(s)")


def plot_sample(sample: dict, split: str, idx: int, shard_name: str, out_path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    glitch_threshold = 1e-6

    for col, det in enumerate(IFOS):
        d = sample[det]
        has_glitch = np.abs(d["glitch"]).max() > glitch_threshold

        ax = axes[0, col]
        ax.plot(TIME_AXIS, d["noisy"], color="grey", lw=0.5, alpha=0.8, label="Noisy (input)")
        ax.plot(TIME_AXIS, d["signal_only"], color="black", lw=0.8, ls="--", alpha=0.8, label="Signal")
        if has_glitch:
            ax.plot(TIME_AXIS, d["glitch"], color="red", lw=0.8, ls="--", alpha=0.8, label="Glitch")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(f"{det} -- input", fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

        ax = axes[1, col]
        ax.plot(TIME_AXIS, d["background"], color="grey", lw=0.4, alpha=0.6, label="Background (real noise)")
        ax.plot(TIME_AXIS, d["signal_only"], color="royalblue", lw=0.8, label="Signal target")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(f"{det} signal", fontsize=9)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

        ax = axes[2, col]
        ax.plot(TIME_AXIS, d["noisy"], color="grey", lw=0.4, alpha=0.4)
        if has_glitch:
            ax.plot(TIME_AXIS, d["glitch"], color="tomato", lw=0.8, label="Glitch target")
            ax.set_title(f"{det} glitch", fontsize=9, color="darkred")
        else:
            ax.set_title(f"{det} -- no glitch", fontsize=9, color="grey")
        ax.axvline(T_INJ, color="red", lw=0.8, ls=":", alpha=0.5)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Time (s)", fontsize=8)

    fig.suptitle(f"{split}  sample {idx}  |  from {shard_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shard-dir", type=Path, required=True,
                   help="e.g. real_separation_data/O3 (train/val shards + test_holdout/ subdir)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=Path("generated_sample_plots"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    run_label = args.shard_dir.name

    for split in args.splits:
        print(f"\n{split}:")
        samples, shard_path = load_random_samples(args.shard_dir, split, args.n_samples, rng)
        for i, sample in enumerate(samples):
            out_path = args.out_dir / f"{run_label}_{split}_sample{i}.png"
            plot_sample(sample, split, i, shard_path.name, out_path)


if __name__ == "__main__":
    main()
