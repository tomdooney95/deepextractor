"""
Generate synthetic time-domain training data for multi-detector (H1/L1/V1)
signal + glitch separation.

For every sample, per active detector:

  1. Draw simultaneous whitened Gaussian noise from bilby's PSDs (one
     ``set_strain_data_from_power_spectral_densities`` call sets strain for
     the whole network at once, so per-detector noise is independent but
     drawn at the same instant).
  2. With probability ``1 - no_inj_prob``, inject one physical CBC signal
     (IMRPhenomXPHM BBH) via ``InterferometerList.inject_signal()`` — this
     projects a single physical waveform through each detector's actual
     antenna response and arrival-time delay, so the signal is coherent but
     not identical across detectors.
  3. Independently, choose exactly one detector at random and add 1-30
     analytic glitch injections (chirp/sine/sine_gaussian/gaussian_pulse/
     ringdown) at random SNR/duration/offset. Only one detector is
     glitch-contaminated per sample — glitches are instrumental and
     detector-local, unlike the coherent signal.

Targets written per detector: ``noisy`` (background + signal + glitch),
``background`` (noise only), ``signal_only`` (signal only). The glitch is
not given its own target — it's recoverable as
``noisy - background - signal_only``, and reconstructing it directly (with
a consistency loss instead of a direct target) is left as a documented
follow-up (see the module docstring on training use).

Adapted from a validated two-detector (H1/L1) Snellius pipeline
(``DeepExtractor_v2_CIT/generate_bilby_signals_volume_parallel_h5.py`` +
``generate_training_data_bilby_hdf5_td_no_scaling_parallel.py``), extended
to three detectors and fused into a single streaming pass — the original
pipeline wrote an intermediate raw ``strain_data.npy`` and then re-split it
into final HDF5 shards, roughly doubling storage; this version writes
straight to final per-shard files. It also fixes the old pipeline's
``SNR_SCALING_FACTOR_BILBY = 31.97`` constant (empirically under-corrected)
in favour of the analytically correct ``sqrt(sample_rate / 2)`` factor
already used in ``generate_timeseries.py``.

Each worker process writes its own shard file (no concurrent multi-writer
access to a shared HDF5 file), matching the pattern already used for
spectrogram generation on CIT (see ``generate_spectrograms.py``).

Usage::

    # Local
    deepextractor-generate-separation --output-dir data_separation/ \\
        --num-train 1550000 --num-val 170000 --detectors H1 L1 V1

    # Snellius SLURM (no editing needed — all config is CLI args)
    deepextractor-generate-separation --output-dir $TMPDIR/data_separation/ \\
        --num-train 1550000 --num-val 170000 --detectors H1 L1 V1 \\
        --num-workers 32 --shard-size 20000

Storage: at the defaults (4s, 4096 Hz, 3 detectors, float32), one sample
(noisy + background + signal_only, all 3 detectors) costs 576 KiB. The
--num-train/--num-val defaults below (1,550,000 / 170,000 = 1.72M total)
total ~945 GiB, sized for a ~950GB budget on a 1TB volume. Pass
smaller/larger counts directly if your actual free space differs.
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import astropy.units as u
import bilby
import h5py
import numpy as np
from astropy.cosmology import Planck18
from tqdm import tqdm

from deepextractor.generation.glitch_functions import (
    generate_chirp,
    generate_gaussian_pulse,
    generate_sine,
    generate_sine_gaussian,
    ringdown,
)
from deepextractor.utils.signal import whitened_snr_scaling

bilby.core.utils.setup_logger(log_level="warning")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Physical / signal-processing constants -------------------------------
SAMPLE_RATE = 4096
DURATION = 4.0
LENGTH = int(DURATION * SAMPLE_RATE)
GLITCH_T_MIN, GLITCH_T_MAX = 0.125, 2.0
SNR_MIN, SNR_MAX = 1, 250
NO_INJ_PROB = 0.05
DMAX_MPC = 3000.0
MINIMUM_FREQUENCY = 20.0
REFERENCE_FREQUENCY = 50.0
WAVEFORM_APPROXIMANT = "IMRPhenomXPHM"
# Arbitrary reference epoch (GW150914's GPS time). Only fixes the instant at
# which noise/antenna-response are evaluated -- sky position, distance, and
# every other CBC parameter are redrawn per sample, so this introduces no
# bias. Segment start offset (-3.5s within a 4s window) matches the
# reference Snellius pipeline.
GEOCENT_TIME = 1126259642.413
MAX_INJECTION_RETRIES = 10
TRAIN_FRAC = 0.9

ALL_DETECTORS = ["H1", "L1", "V1"]

SIGNAL_TYPES = ["chirp", "sine", "sine_gaussian", "gaussian_pulse", "ringdown"]
SIGNAL_FUNCTION_MAP = {
    "chirp": generate_chirp,
    "sine": generate_sine,
    "sine_gaussian": generate_sine_gaussian,
    "gaussian_pulse": generate_gaussian_pulse,
    "ringdown": ringdown,
}

DTYPE = np.float32


# ---------------------------------------------------------------------------
# Luminosity distance: uniform-in-comoving-volume sampling out to dmax_mpc
# ---------------------------------------------------------------------------

def _build_dc_dl_lookup(dmax_mpc: float, n_z: int = 8000):
    z_max = 1.5
    while Planck18.comoving_distance(z_max).to_value(u.Mpc) < dmax_mpc and z_max < 10:
        z_max *= 1.2
    z = np.linspace(0, z_max, n_z)
    dc = Planck18.comoving_distance(z).to_value(u.Mpc)
    dl = (1.0 + z) * dc
    return dc.astype(np.float64), dl.astype(np.float64)


def _sample_luminosity_distance(rng, dc_grid, dl_grid, dmax_mpc: float):
    u_rand = rng.random()
    dc = dmax_mpc * u_rand ** (1 / 3)
    return float(np.interp(dc, dc_grid, dl_grid))


def _random_cbc_parameters(rng, dc_grid, dl_grid, dmax_mpc, geocent_time):
    m1 = rng.uniform(5, 200)
    m2 = rng.uniform(5, 200)
    if m1 < m2:
        m1, m2 = m2, m1
    return dict(
        mass_1=m1, mass_2=m2,
        a_1=rng.uniform(0, 0.99), a_2=rng.uniform(0, 0.99),
        tilt_1=rng.uniform(0, np.pi), tilt_2=rng.uniform(0, np.pi),
        phi_12=rng.uniform(0, 2 * np.pi), phi_jl=rng.uniform(0, 2 * np.pi),
        luminosity_distance=_sample_luminosity_distance(rng, dc_grid, dl_grid, dmax_mpc),
        theta_jn=np.arccos(rng.uniform(-1.0, 1.0)),
        psi=rng.uniform(0, np.pi),
        phase=rng.uniform(0, 2 * np.pi),
        geocent_time=geocent_time,
        ra=rng.uniform(0, 2 * np.pi),
        dec=np.arcsin(rng.uniform(-1.0, 1.0)),
    )


# ---------------------------------------------------------------------------
# Per-sample generation
# ---------------------------------------------------------------------------

def _inject_glitch(noisy: np.ndarray, rng, sample_rate: int = SAMPLE_RATE,
                    t_min: float = GLITCH_T_MIN, t_max: float = GLITCH_T_MAX,
                    snr_min: float = SNR_MIN, snr_max: float = SNR_MAX) -> None:
    """Add 1-30 random analytic glitch morphologies to ``noisy`` in place.

    Requested SNR is drawn in bilby's unit-variance-whitened convention and
    converted to the flat-PSD convention ``whitened_snr_scaling`` assumes via
    the analytically correct ``sqrt(sample_rate / 2)`` factor (replaces the
    empirically-wrong fitted constant used in the original pipeline).
    """
    length = noisy.shape[0]
    n_injs = int(rng.integers(1, 30))
    for _ in range(n_injs):
        snr_to_scale = rng.uniform(snr_min, snr_max) / np.sqrt(sample_rate / 2)
        duration = rng.uniform(t_min, t_max)
        s_type = SIGNAL_TYPES[rng.integers(0, len(SIGNAL_TYPES))]
        _, waveform = SIGNAL_FUNCTION_MAP[s_type](duration, sample_rate=sample_rate)
        waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)
        if waveform.size == 0 or np.isnan(waveform).any():
            continue

        glitch = whitened_snr_scaling(waveform - waveform.mean(), snr=snr_to_scale, srate=sample_rate)
        len_glitch = glitch.shape[0]

        centre = length // 2 - len_glitch // 2
        left = max(-centre, -(len_glitch - 1))
        right = max(1, length - centre - len_glitch)
        shift = int(rng.integers(left, right))
        start = centre + shift
        if start < 0 or start + len_glitch > length:
            continue
        noisy[start:start + len_glitch] += glitch


def _generate_sample(ifos, wfg, rng, dc_grid, dl_grid, dmax_mpc, no_inj_prob,
                      geocent_time, duration):
    """Generate one multi-detector sample.

    Returns ``{detector_name: (noisy, background, signal_only)}``, each a
    length-``LENGTH`` float64 array.
    """
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLE_RATE, duration=duration,
        start_time=geocent_time - (duration - 0.5),
    )
    background = {
        ifo.name: np.asarray(ifo.whitened_time_domain_strain, dtype=np.float64).copy()
        for ifo in ifos
    }
    signal_only = {name: np.zeros_like(arr) for name, arr in background.items()}

    if rng.random() >= no_inj_prob:
        for _ in range(MAX_INJECTION_RETRIES):
            try:
                params = _random_cbc_parameters(rng, dc_grid, dl_grid, dmax_mpc, geocent_time)
                ifos.inject_signal(waveform_generator=wfg, parameters=params)
                for ifo in ifos:
                    strained = np.asarray(ifo.whitened_time_domain_strain, dtype=np.float64)
                    signal_only[ifo.name] = strained - background[ifo.name]
                break
            except Exception:
                continue  # falls back to signal_only=0 (noise-only sample) if all retries fail

    noisy = {name: background[name] + signal_only[name] for name in background}

    glitch_detector = ifos[int(rng.integers(0, len(ifos)))].name
    _inject_glitch(noisy[glitch_detector], rng, sample_rate=SAMPLE_RATE)

    return {
        name: (noisy[name], background[name], signal_only[name])
        for name in background
    }


# ---------------------------------------------------------------------------
# Worker: builds its own InterferometerList/WaveformGenerator, writes its own
# shard file (no concurrent access to a shared HDF5 file across processes).
# ---------------------------------------------------------------------------

def _worker_shard(shard_id, start, stop, train_cut, detectors, out_dir, duration,
                   no_inj_prob, dmax_mpc, geocent_time, seed):
    rng = np.random.default_rng(seed)

    ifos = bilby.gw.detector.InterferometerList(list(detectors))
    for ifo in ifos:
        ifo.minimum_frequency = MINIMUM_FREQUENCY

    wfg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration, sampling_frequency=SAMPLE_RATE,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        waveform_arguments=dict(
            waveform_approximant=WAVEFORM_APPROXIMANT,
            reference_frequency=REFERENCE_FREQUENCY,
            minimum_frequency=MINIMUM_FREQUENCY,
        ),
    )

    length = int(duration * SAMPLE_RATE)
    dc_grid, dl_grid = _build_dc_dl_lookup(dmax_mpc)

    n_total = stop - start
    n_train = max(0, min(train_cut - start, n_total)) if start < train_cut else 0
    n_val = n_total - n_train

    shard_path = os.path.join(out_dir, f"separation_shard_{shard_id:04d}.h5")
    # h5py rejects a chunk shape whose first dim exceeds the dataset's first
    # dim, which happens for a 0-length split (e.g. a shard entirely inside
    # train has n_val == 0) -- fall back to an unchunked dataset there.
    chunks_train = (min(2048, n_train), length) if n_train > 0 else None
    chunks_val = (min(2048, n_val), length) if n_val > 0 else None

    with h5py.File(shard_path, "w") as f:
        datasets = {}
        for det in detectors:
            for key in ("noisy", "background", "signal_only"):
                datasets[(det, key, "train")] = f.create_dataset(
                    f"{key}_{det}_train", shape=(n_train, length), dtype=DTYPE, chunks=chunks_train,
                )
                datasets[(det, key, "val")] = f.create_dataset(
                    f"{key}_{det}_val", shape=(n_val, length), dtype=DTYPE, chunks=chunks_val,
                )

        wptr_train = 0
        wptr_val = 0
        for idx in range(start, stop):
            sample = _generate_sample(
                ifos, wfg, rng, dc_grid, dl_grid, dmax_mpc, no_inj_prob, geocent_time, duration,
            )
            split, wptr = ("train", wptr_train) if idx < train_cut else ("val", wptr_val)
            for det in detectors:
                noisy, background, signal_only = sample[det]
                datasets[(det, "noisy", split)][wptr] = noisy.astype(DTYPE, copy=False)
                datasets[(det, "background", split)][wptr] = background.astype(DTYPE, copy=False)
                datasets[(det, "signal_only", split)][wptr] = signal_only.astype(DTYPE, copy=False)
            if split == "train":
                wptr_train += 1
            else:
                wptr_val += 1

    return shard_path, n_train, n_val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-detector (H1/L1/V1) signal+glitch separation training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_separation/"))
    parser.add_argument("--detectors", nargs="+", default=ALL_DETECTORS,
                         help="Detectors to simulate (must be in bilby's built-in database).")
    # Sized for ~945 GiB total at 4s/4096Hz/3-detector (576 KiB/sample) --
    # see module docstring. Override directly once you've checked real free space.
    parser.add_argument("--num-train", type=int, default=1_550_000)
    parser.add_argument("--num-val", type=int, default=170_000)
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--no-inj-prob", type=float, default=NO_INJ_PROB,
                         help="Fraction of samples with no signal injected (pure noise, still glitch-eligible).")
    parser.add_argument("--dmax-mpc", type=float, default=DMAX_MPC)
    parser.add_argument("--shard-size", type=int, default=20_000,
                         help="Samples per shard file (~11.25GB/shard at 3 detectors, 4s).")
    parser.add_argument("--num-workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_total = args.num_train + args.num_val
    train_cut = args.num_train

    bytes_per_sample = len(args.detectors) * 3 * int(args.duration * SAMPLE_RATE) * 4
    logger.info(f"Detectors: {args.detectors}")
    logger.info(f"Total samples: {n_total:,}  (train {args.num_train:,} / val {args.num_val:,})")
    logger.info(f"Estimated size: {n_total * bytes_per_sample / 1024**3:.1f} GiB "
                f"({bytes_per_sample / 1024:.0f} KiB/sample)")
    logger.info(f"Writing shards to: {args.output_dir}")
    logger.info(f"Workers: {args.num_workers} | shard size: {args.shard_size:,}")

    tasks = []
    shard_id = 0
    for start in range(0, n_total, args.shard_size):
        stop = min(start + args.shard_size, n_total)
        tasks.append((
            shard_id, start, stop, train_cut, args.detectors, str(args.output_dir),
            args.duration, args.no_inj_prob, args.dmax_mpc, GEOCENT_TIME, args.seed + start,
        ))
        shard_id += 1

    total_train = total_val = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(_worker_shard, *t) for t in tasks]
        with tqdm(total=len(futures), desc="Shards", unit="shard") as pbar:
            for fut in as_completed(futures):
                shard_path, n_train, n_val = fut.result()
                total_train += n_train
                total_val += n_val
                logger.info(f"Shard written: {os.path.basename(shard_path)} | train {n_train:,} val {n_val:,}")
                pbar.update(1)

    logger.info(f"Done. Total train: {total_train:,} | total val: {total_val:,}")
    logger.info(f"{shard_id} shard files written to {args.output_dir}")


if __name__ == "__main__":
    main()
