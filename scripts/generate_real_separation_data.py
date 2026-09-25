"""
Full-scale O3/O4 real-noise signal+glitch separation dataset generator.

See docs/real_noise_dataset_methodology.md for the complete design
rationale (why H1/L1 are pooled, why the split is context-level with a
separate chronological test holdout, why sign-flip/time-reversal are
applied to noise only, why SNR is sampled instead of distance, etc.) --
this docstring covers usage and implementation only.

Output layout
-------------
    <out-dir>/separation_shard_NNNN.h5   -- train+val together, matching
                                             HDF5SeparationDataset's format
                                             exactly (glob-matched filename,
                                             per-shard "{key}_{det}_train"/
                                             "{key}_{det}_val" datasets).
    <out-dir>/test_holdout/separation_shard_NNNN.h5
                                          -- chronologically-last 10%,
                                             "{key}_{det}_test" datasets.
                                             Separate subdirectory so the
                                             standard training loader
                                             (scoped to <out-dir> only)
                                             never discovers it.

Run once per observing run -- O3 and O4 are independent invocations with
independent --out-dir values.

Usage
-----
    python scripts/generate_real_separation_data.py \\
        --run O3 --backgrounds-dir . --psd-dir segment_psds/ \\
        --out-dir real_separation_data/O3/ \\
        --duplication 10 --shard-size 5000 --num-workers 4
"""

import argparse
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import bilby
import h5py
import numpy as np
from gwpy.frequencyseries import FrequencySeries

from deepextractor.generation.generate_separation_data import (
    DMAX_MPC, GEOCENT_TIME, MINIMUM_FREQUENCY, REFERENCE_FREQUENCY,
    SAMPLE_RATE, WAVEFORM_APPROXIMANT, _build_dc_dl_lookup, _inject_glitch,
)
from generate_real_noise_samples import (
    IFOS, LENGTH, MAX_FILTER_DUR, NO_INJ_PROB, SIGNAL_WINDOW_DURATION,
    SNR_MAX, SNR_MIN, SUBRUNS, T_INJ, generate_whitened_signal,
)

bilby.core.utils.setup_logger(log_level="warning")

RAW_LENGTH = 32768  # 8s @ 4096 Hz -- length of a stored real-noise sample
SLIDE_MAX_OFFSET = RAW_LENGTH - LENGTH  # 16384 -- 4s of slack, see methodology doc

# ── Pool loading ─────────────────────────────────────────────────────────────

def load_pooled_run(backgrounds_dir: Path, psd_dir: Path, run: str) -> dict:
    """Pool H1+L1 real-noise samples for a run. PSDs are kept deduplicated,
    keyed by context GPS time (not one row per sample) -- a context's 13
    overlapping samples all share one PSD, so duplicating it per-sample
    would multiply memory use ~13x for no reason.
    """
    all_samples, all_gps = [], []
    psd_by_gps: dict[float, np.ndarray] = {}
    psd_freqs = None
    subruns = SUBRUNS.get(run, [run])

    for ifo in IFOS:
        for subrun in subruns:
            with open(backgrounds_dir / f"backgrounds_{subrun}.pkl", "rb") as f:
                bg = pickle.load(f)[subrun][ifo]
            with open(psd_dir / f"{subrun}_{ifo}_psds.pkl", "rb") as f:
                psd_data = pickle.load(f)
            if psd_freqs is None:
                psd_freqs = np.asarray(psd_data["psd_freqs"], dtype=np.float64)
            for g, psd_row in zip(psd_data["psd_gps_starts"], psd_data["psds"]):
                psd_by_gps[float(g)] = np.asarray(psd_row, dtype=np.float64)

            all_samples.append(np.asarray(bg["samples"], dtype=np.float32))
            all_gps.append(np.asarray(bg["gps_starts"], dtype=np.float64))

    return {
        "samples": np.concatenate(all_samples, axis=0),
        "gps_starts": np.concatenate(all_gps, axis=0),
        "psd_by_gps": psd_by_gps,
        "psd_freqs": psd_freqs,
    }


def split_pool_by_context(gps_starts: np.ndarray, seed: int,
                           train_frac: float = 0.8, val_frac: float = 0.1,
                           test_frac: float = 0.1,
                           time_block_seconds: float = 7 * 86400) -> dict:
    """Context-level (not sample-level) train/val/test split.

    test = chronologically-last test_frac of contexts, held out entirely.
    train/val = stratified by time block across the remaining (chronologically
    first) contexts, so both get representative coverage of that range rather
    than a raw random shuffle that could still cluster unevenly.

    A context's `gps_starts` value is shared by every sample sliced from it,
    so grouping by unique gps_starts automatically keeps a context's
    correlated samples together in one split -- no separate bookkeeping
    needed to prevent leakage across the boundary.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    rng = np.random.default_rng(seed)

    unique_gps = np.unique(gps_starts)
    unique_gps.sort()

    n_test = max(1, int(round(len(unique_gps) * test_frac)))
    test_contexts = set(unique_gps[-n_test:].tolist())
    remaining = unique_gps[:-n_test]

    t0 = remaining.min()
    block_id = np.floor((remaining - t0) / time_block_seconds).astype(int)
    val_share = val_frac / (train_frac + val_frac)

    train_contexts, val_contexts = set(), set()
    for b in np.unique(block_id):
        block_ctx = remaining[block_id == b]
        shuffled = rng.permutation(block_ctx)
        n_val = int(round(len(shuffled) * val_share))
        val_contexts.update(shuffled[:n_val].tolist())
        train_contexts.update(shuffled[n_val:].tolist())

    context_to_split = {}
    context_to_split.update({g: "train" for g in train_contexts})
    context_to_split.update({g: "val" for g in val_contexts})
    context_to_split.update({g: "test" for g in test_contexts})

    split_indices = {"train": [], "val": [], "test": []}
    for i, g in enumerate(gps_starts):
        split_indices[context_to_split[float(g)]].append(i)

    return {k: np.array(v, dtype=np.int64) for k, v in split_indices.items()}

# ── Per-example generation ───────────────────────────────────────────────────

def slide_and_augment(raw_8s: np.ndarray, rng: np.random.Generator, allow_reversal: bool) -> np.ndarray:
    """Random 4s sub-window from an already-clean 8s real-noise sample, then
    sign-flip (always) and time-reversal (train split only)."""
    start = int(rng.integers(0, SLIDE_MAX_OFFSET + 1))
    window = raw_8s[start:start + LENGTH].astype(np.float64, copy=True)
    if rng.random() < 0.5:
        window = -window
    if allow_reversal and rng.random() < 0.5:
        window = np.ascontiguousarray(window[::-1])
    return window


def generate_one_example(pool: dict, split_indices: np.ndarray, rng, ifos, wfg,
                          dc_grid, dl_grid, allow_reversal: bool) -> dict:
    """Draw two independent positions from the (H1+L1-pooled) split pool to
    fill the two output channels -- positional slots, not tied to which
    physical detector the noise actually came from (see methodology doc:
    H1/L1 pooling). Returns {"H1": (noisy, background, signal_only), "L1": ...}.
    """
    slot_positions = rng.choice(split_indices, size=2, replace=True)
    noise, asds = {}, {}
    for slot_ifo, pos in zip(IFOS, slot_positions):
        raw = pool["samples"][pos]
        noise[slot_ifo] = slide_and_augment(raw, rng, allow_reversal)
        gps = float(pool["gps_starts"][pos])
        asds[slot_ifo] = FrequencySeries(
            np.sqrt(pool["psd_by_gps"][gps]), frequencies=pool["psd_freqs"],
        )

    start_time = GEOCENT_TIME - (MAX_FILTER_DUR + T_INJ)
    if rng.random() < NO_INJ_PROB:
        signal = {ifo: np.zeros(LENGTH, dtype=np.float64) for ifo in IFOS}
    else:
        signal, _ = generate_whitened_signal(
            ifos, wfg, rng, dc_grid, dl_grid, asds, start_time,
            sample_snr=True, snr_min=SNR_MIN, snr_max=SNR_MAX,
        )

    background = noise
    noisy = {ifo: noise[ifo] + signal[ifo] for ifo in IFOS}
    glitch_ifo = IFOS[int(rng.integers(0, len(IFOS)))]
    _inject_glitch(noisy[glitch_ifo], rng, sample_rate=SAMPLE_RATE)

    return {ifo: (noisy[ifo], background[ifo], signal[ifo]) for ifo in IFOS}

# ── Worker process ────────────────────────────────────────────────────────────
# Each worker loads and pools the run's data once at startup (via the
# ProcessPoolExecutor initializer), not once per shard -- avoids repeatedly
# pickling multi-GB arrays through the task queue. Keep --num-workers modest:
# every worker holds its own full copy of the pooled run (~1-8 GiB depending
# on run/detector combo; see methodology doc).

_WORKER_POOL = None
_WORKER_SPLITS = None


def _worker_init(backgrounds_dir: str, psd_dir: str, run: str, seed: int):
    global _WORKER_POOL, _WORKER_SPLITS
    _WORKER_POOL = load_pooled_run(Path(backgrounds_dir), Path(psd_dir), run)
    _WORKER_SPLITS = split_pool_by_context(_WORKER_POOL["gps_starts"], seed=seed)


def _build_ifos_and_wfg():
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
    return ifos, wfg


def _worker_shard_train_val(shard_id, start, stop, train_cut, out_dir, seed):
    rng = np.random.default_rng(seed)
    ifos, wfg = _build_ifos_and_wfg()
    dc_grid, dl_grid = _build_dc_dl_lookup(DMAX_MPC)

    n_total = stop - start
    n_train = max(0, min(train_cut - start, n_total)) if start < train_cut else 0
    n_val = n_total - n_train

    shard_path = os.path.join(out_dir, f"separation_shard_{shard_id:04d}.h5")
    chunks_train = (min(2048, n_train), LENGTH) if n_train > 0 else None
    chunks_val = (min(2048, n_val), LENGTH) if n_val > 0 else None

    with h5py.File(shard_path, "w") as f:
        datasets = {}
        for det in IFOS:
            for key in ("noisy", "background", "signal_only"):
                datasets[(det, key, "train")] = f.create_dataset(
                    f"{key}_{det}_train", shape=(n_train, LENGTH), dtype=np.float32, chunks=chunks_train,
                )
                datasets[(det, key, "val")] = f.create_dataset(
                    f"{key}_{det}_val", shape=(n_val, LENGTH), dtype=np.float32, chunks=chunks_val,
                )

        wptr_train = wptr_val = 0
        for idx in range(start, stop):
            split_name = "train" if idx < train_cut else "val"
            sample = generate_one_example(
                _WORKER_POOL, _WORKER_SPLITS[split_name], rng, ifos, wfg, dc_grid, dl_grid,
                allow_reversal=(split_name == "train"),
            )
            wptr = wptr_train if split_name == "train" else wptr_val
            for det in IFOS:
                noisy, background, signal_only = sample[det]
                datasets[(det, "noisy", split_name)][wptr] = noisy.astype(np.float32, copy=False)
                datasets[(det, "background", split_name)][wptr] = background.astype(np.float32, copy=False)
                datasets[(det, "signal_only", split_name)][wptr] = signal_only.astype(np.float32, copy=False)
            if split_name == "train":
                wptr_train += 1
            else:
                wptr_val += 1

    return shard_path, n_train, n_val


def _worker_shard_test(shard_id, start, stop, out_dir, seed):
    rng = np.random.default_rng(seed)
    ifos, wfg = _build_ifos_and_wfg()
    dc_grid, dl_grid = _build_dc_dl_lookup(DMAX_MPC)

    n = stop - start
    shard_path = os.path.join(out_dir, f"separation_shard_{shard_id:04d}.h5")
    with h5py.File(shard_path, "w") as f:
        datasets = {}
        for det in IFOS:
            for key in ("noisy", "background", "signal_only"):
                datasets[(det, key)] = f.create_dataset(
                    f"{key}_{det}_test", shape=(n, LENGTH), dtype=np.float32, chunks=(min(2048, n), LENGTH),
                )
        for i, idx in enumerate(range(start, stop)):
            sample = generate_one_example(
                _WORKER_POOL, _WORKER_SPLITS["test"], rng, ifos, wfg, dc_grid, dl_grid, allow_reversal=False,
            )
            for det in IFOS:
                noisy, background, signal_only = sample[det]
                datasets[(det, "noisy")][i] = noisy.astype(np.float32, copy=False)
                datasets[(det, "background")][i] = background.astype(np.float32, copy=False)
                datasets[(det, "signal_only")][i] = signal_only.astype(np.float32, copy=False)

    return shard_path, n

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, choices=["O3", "O4"])
    p.add_argument("--backgrounds-dir", type=Path, default=Path("."))
    p.add_argument("--psd-dir", type=Path, default=Path("segment_psds"))
    p.add_argument("--out-dir", type=Path, required=True,
                   help="train+val shards written here as separation_shard_NNNN.h5")
    p.add_argument("--test-out-dir", type=Path, default=None,
                   help="Defaults to <out-dir>/test_holdout/")
    p.add_argument("--duplication", type=int, default=10,
                   help="Generated examples per pooled real-noise sample, per split.")
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=4,
                   help="Keep modest: each worker independently loads and holds the full "
                        "pooled H1+L1 run in memory (~1-8 GiB depending on run). See "
                        "docs/real_noise_dataset_methodology.md.")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir = args.test_out_dir or (args.out_dir / "test_holdout")
    test_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and pooling {args.run} H1+L1 real noise ...")
    pool = load_pooled_run(args.backgrounds_dir, args.psd_dir, args.run)
    splits = split_pool_by_context(pool["gps_starts"], seed=args.seed)

    bytes_per_sample = len(IFOS) * 3 * LENGTH * 4
    n_examples = {name: len(splits[name]) * args.duplication for name in ("train", "val", "test")}
    for name, n in n_examples.items():
        print(f"  {name}: {len(splits[name]):,} pooled samples x {args.duplication} -> "
              f"{n:,} examples (~{n * bytes_per_sample / 1024**3:.1f} GiB)")
    del pool  # each worker loads its own copy via the initializer

    # --- train + val: combined shards, matching HDF5SeparationDataset's format ---
    n_total = n_examples["train"] + n_examples["val"]
    train_cut = n_examples["train"]
    tasks = []
    shard_id = 0
    for start in range(0, n_total, args.shard_size):
        stop = min(start + args.shard_size, n_total)
        tasks.append((shard_id, start, stop, train_cut, str(args.out_dir), args.seed + start))
        shard_id += 1

    print(f"\nGenerating {args.run} train+val: {n_total:,} examples across {len(tasks)} shards ...")
    with ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=_worker_init,
        initargs=(str(args.backgrounds_dir), str(args.psd_dir), args.run, args.seed),
    ) as ex:
        futures = [ex.submit(_worker_shard_train_val, *t) for t in tasks]
        total_train = total_val = 0
        for fut in as_completed(futures):
            shard_path, n_tr, n_v = fut.result()
            total_train += n_tr
            total_val += n_v
            print(f"  {os.path.basename(shard_path)}: train {n_tr:,} val {n_v:,}")
    print(f"{args.run} train+val done: {total_train:,} train / {total_val:,} val -> {args.out_dir}")

    # --- test: separate shards, separate subdirectory (never seen by the training loader) ---
    tasks = []
    shard_id = 0
    for start in range(0, n_examples["test"], args.shard_size):
        stop = min(start + args.shard_size, n_examples["test"])
        tasks.append((shard_id, start, stop, str(test_out_dir), args.seed + 10_000_000 + start))
        shard_id += 1

    if tasks:
        print(f"\nGenerating {args.run} test (chronological holdout): {n_examples['test']:,} examples ...")
        with ProcessPoolExecutor(
            max_workers=args.num_workers, initializer=_worker_init,
            initargs=(str(args.backgrounds_dir), str(args.psd_dir), args.run, args.seed),
        ) as ex:
            futures = [ex.submit(_worker_shard_test, *t) for t in tasks]
            total_test = 0
            for fut in as_completed(futures):
                shard_path, n = fut.result()
                total_test += n
                print(f"  {os.path.basename(shard_path)}: test {n:,}")
        print(f"{args.run} test done: {total_test:,} -> {test_out_dir}")


if __name__ == "__main__":
    main()
