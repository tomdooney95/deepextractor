"""
Generate synthetic time-domain training data.

Usage::

    # Original pycbc white noise (fast, no PSD)
    deepextractor-generate --output-dir data/ --num-train 250000

    # Bilby PSD-coloured noise (default realistic option)
    deepextractor-generate --output-dir data_4s/ --num-train 250000 \\
        --noise-backend bilby --detector L1 \\
        --duration 4.0 --minimum-frequency 10.0

    # gwmock-noise coloured noise — requires a PSD file and gwmock-noise installed
    # (pip install deepextractor[gwmock])
    deepextractor-generate --output-dir data_ET1/ --num-train 250000 \\
        --noise-backend gwmock --detector ET1 \\
        --psd-file psds/ET_D_sensitivity.txt \\
        --duration 4.0 --minimum-frequency 10.0 --format hdf5

    # Supported detectors vary by backend:
    #   bilby  — any interferometer in bilby's built-in database:
    #             H1, L1, V1, K1, GEO600, ET, CE, NEMO, A1
    #             (bilby treats ET as a single entity, not individual arms)
    #   gwmock — any string label; noise is coloured by --psd-file:
    #             ET1, ET2, ET3, ET null stream, CE1, custom, ...
    #
    # NOTE: the ET null stream ((ET1+ET2+ET3)/sqrt(6)) cancels GW signals by
    # design. Generating it correctly requires *correlated* ET arm noise; this
    # implementation generates each arm independently and is therefore an
    # approximation. Proper correlated noise is planned for a future update.

"""

import argparse
import os
import pickle
import random

import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from deepextractor.generation.glitch_functions import (
    generate_chirp,
    generate_gaussian_pulse,
    generate_sine,
    generate_sine_gaussian,
    ringdown,
)
from deepextractor.utils.signal import whitened_snr_scaling

SAMPLE_RATE = 4096
T = 2.0
T_INJ = T / 2
LENGTH = int(T * SAMPLE_RATE)
SNR_MIN, SNR_MAX = 1, 250
MINIMUM_FREQUENCY = 20.0

# Earlier versions of this module scaled bilby-noise injections by a fitted
# constant, SNR_SCALING_FACTOR_BILBY = 31.970149253731343, to compensate for
# bilby's whitened_time_domain_strain being unit-variance rather than the
# SAMPLE_RATE/2-variance convention snr_scaling()'s flat-PSD formula assumes.
# That constant under-corrected (verified empirically: a requested SNR of 20
# was actually realized at SNR~28.6) -- harmless for training given the wide
# SNR sampling range used, but not analytically correct. The exact,
# duration/cutoff-independent conversion is sqrt(SAMPLE_RATE / 2), computed
# inline in generate_synthetic_data() below.

SIGNAL_TYPES = ["chirp", "sine", "sine_gaussian", "gaussian_pulse", "ringdown"]
SIGNAL_FUNCTION_MAP = {
    "chirp": generate_chirp,
    "sine": generate_sine,
    "sine_gaussian": generate_sine_gaussian,
    "gaussian_pulse": generate_gaussian_pulse,
    "ringdown": ringdown,
}


def generate_gaussian_noise(mean, std_dev, num_samples, sample_shape, bilby_noise=False,
                            sample_rate=SAMPLE_RATE, duration=T,
                            minimum_frequency=MINIMUM_FREQUENCY, detector="L1",
                            show_progress=True):
    """Generate Gaussian noise samples (pycbc or bilby)."""
    if bilby_noise:
        try:
            import bilby
        except ImportError as e:
            raise ImportError(
                "bilby is required for bilby noise generation. "
                "Install it with: pip install deepextractor[generative]"
            ) from e
        # Build the interferometer once — set_strain_data_from_power_spectral_densities
        # draws a fresh independent noise realisation on every call, so the ifo object
        # can and should be reused across all samples. Constructing InterferometerList
        # per-sample loads PSD calibration files from disk each time, which is
        # catastrophically slow on networked cluster filesystems (NFS/ZFS-over-NFS).
        ifos = bilby.gw.detector.InterferometerList([detector])
        for ifo in ifos:
            ifo.minimum_frequency = minimum_frequency

        gaussian_noise_samples = []
        iterator = range(num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating bilby noise...")
        for i in iterator:
            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=sample_rate,
                duration=duration,
                start_time=0,
            )
            white_time_domain_strain = list(ifos[0].whitened_time_domain_strain)
            gaussian_noise_samples.append(white_time_domain_strain)
        return np.asarray(gaussian_noise_samples)
    else:
        if show_progress:
            print("Generating pycbc noise...")
        return np.random.normal(loc=mean, scale=std_dev, size=(num_samples, *sample_shape))


def generate_detector_noise(
    n_samples: int,
    detector: str = "L1",
    duration: float = T,
    sample_rate: float = SAMPLE_RATE,
    backend: str = "bilby",
    minimum_frequency: float = MINIMUM_FREQUENCY,
    psd_file: str | None = None,
    show_progress: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Generate whitened detector noise samples for any GW interferometer.

    Returns an ``(n_samples, int(duration * sample_rate))`` float32 array of
    unit-variance whitened noise — the same convention as bilby's
    ``whitened_time_domain_strain`` — so the samples are directly compatible
    with :func:`generate_synthetic_data` when ``bilby_noise=True``.

    Args:
        n_samples:          Number of noise realisations to generate.
        detector:           Interferometer name.  Bilby backend: any name in
                            bilby's built-in database — H1, L1, V1, K1,
                            GEO600, ET, CE, NEMO, A1.  Note bilby treats ET
                            as a single entity; for independent arm noise
                            (ET1/ET2/ET3) use the gwmock backend with per-arm
                            PSD files.  gwmock backend: any string label —
                            noise is coloured by the --psd-file you supply.
        duration:           Window length in seconds.
        sample_rate:        Sampling rate in Hz.
        backend:            ``"bilby"`` (default) or ``"gwmock"``.
        minimum_frequency:  Low-frequency cutoff in Hz.
        psd_file:           Path to a two-column (frequency, PSD) ASCII file.
                            Required for the gwmock backend; optional for bilby
                            (overrides the built-in design-sensitivity curve).
        show_progress:      Show a tqdm progress bar.
        seed:               Optional random seed for reproducibility.

    Note:
        ET null stream ((ET1+ET2+ET3)/sqrt(6)) requires *correlated* ET arm
        noise to be physically accurate; this function generates independent
        arms.  Use the output for pilot experiments only.
    """
    length = int(duration * sample_rate)

    if backend == "bilby":
        try:
            import bilby
        except ImportError as exc:
            raise ImportError(
                "bilby is required for the bilby backend. "
                "Install with: pip install deepextractor[generative]"
            ) from exc

        bilby.core.utils.logger.setLevel("ERROR")
        ifos = bilby.gw.detector.InterferometerList([detector])
        for ifo in ifos:
            ifo.minimum_frequency = minimum_frequency
            if psd_file is not None:
                ifo.power_spectral_density = (
                    bilby.gw.detector.PowerSpectralDensity(psd_file=psd_file)
                )
        if seed is not None:
            bilby.core.utils.random.seed(seed)

        samples = []
        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating bilby noise [{detector}]...")
        for _ in iterator:
            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=sample_rate, duration=duration, start_time=0,
            )
            samples.append(np.asarray(ifos[0].whitened_time_domain_strain, dtype=np.float32))
        return np.array(samples, dtype=np.float32)

    if backend == "gwmock":
        if psd_file is None:
            raise ValueError(
                "gwmock backend requires --psd-file.  Provide a two-column "
                "(frequency, PSD) file for the target detector."
            )
        try:
            from gwmock_noise import ColoredNoiseSimulator
        except ImportError as exc:
            raise ImportError(
                "gwmock-noise is required for the gwmock backend. "
                "Install with: pip install deepextractor[gwmock]"
            ) from exc

        sim = ColoredNoiseSimulator(
            psd_file=psd_file,
            detectors=[detector],
            sampling_frequency=sample_rate,
            duration=duration,
            low_frequency_cutoff=minimum_frequency,
            seed=seed,
        )

        # Load PSD for manual whitening.
        # gwmock generates coloured strain — whiten here using the same PSD,
        # matching the convention of bilby's whitened_time_domain_strain so
        # that the output variance is ~1 and the same SNR correction applies.
        psd_data = np.loadtxt(psd_file)
        if psd_data.ndim == 1:
            raise ValueError(f"PSD file {psd_file} must have two columns: frequency and PSD.")
        f_psd, psd_vals = psd_data[:, 0], psd_data[:, 1]
        f_grid = np.fft.rfftfreq(length, d=1.0 / sample_rate)
        psd_interp = np.interp(f_grid, f_psd, psd_vals, left=np.inf, right=np.inf)
        psd_interp[f_grid < minimum_frequency] = np.inf  # zero content below cutoff

        samples = []
        iterator = range(n_samples)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating gwmock noise [{detector}]...")
        _seed = seed
        for _ in iterator:
            colored = sim.generate(duration, sample_rate, [detector], seed=_seed)[detector]
            _seed = None  # subsequent calls advance the simulator's own state
            # Whiten: divide FD strain by sqrt(PSD), normalise to unit variance.
            fd = np.fft.rfft(colored) / sample_rate
            white_fd = np.where(np.isfinite(psd_interp) & (psd_interp > 0),
                                fd / np.sqrt(psd_interp), 0.0)
            white_td = np.fft.irfft(white_fd * sample_rate, n=length) / np.sqrt(sample_rate / 2)
            samples.append(white_td.astype(np.float32))
        return np.array(samples, dtype=np.float32)

    raise ValueError(f"Unknown backend '{backend}'. Choose 'bilby' or 'gwmock'.")


def generate_synthetic_data(gaussian_noise_samples, bilby_noise=False, phase="train",
                             t_min=0.125, t_max=None, snr_min=SNR_MIN, snr_max=SNR_MAX,
                             sample_rate=SAMPLE_RATE, show_progress=True):
    """Generate synthetic noisy glitch and background data arrays.

    ``t_max`` defaults to the window duration implied by
    ``gaussian_noise_samples`` (its length / ``sample_rate``), so injected
    glitches can span up to the full window regardless of duration.
    """
    if t_max is None:
        t_max = gaussian_noise_samples.shape[-1] / sample_rate

    noisy_glitch_ts = []
    pure_noise_ts = []

    iterator = range(len(gaussian_noise_samples))
    if show_progress:
        iterator = tqdm(iterator, desc=f"Generating Synthetic {phase.capitalize()} Data")
    for i in iterator:
        background = gaussian_noise_samples[i]
        noisy_glitch = background.copy()
        n_injs = np.random.randint(1, 30)
        for _ in range(n_injs):
            snr_to_scale = np.random.uniform(snr_min, snr_max)
            if bilby_noise:
                # bilby's whitened_time_domain_strain is unit-variance; convert
                # to the sample_rate/2-variance convention whitened_snr_scaling
                # assumes (see module-level comment above).
                snr_to_scale = snr_to_scale / np.sqrt(sample_rate / 2)
            duration = np.random.uniform(t_min, t_max)
            s_type = random.choice(SIGNAL_TYPES)
            _, signal_injection = SIGNAL_FUNCTION_MAP[s_type](duration)
            len_glitch = len(signal_injection)
            id_start = len(background) // 2 - len_glitch // 2  # centre of the window
            glitch = signal_injection - np.mean(signal_injection)
            glitch = whitened_snr_scaling(glitch, snr=snr_to_scale, srate=sample_rate)
            shift_int = np.random.randint(-id_start, len(background) - id_start - len_glitch)
            noisy_glitch[id_start + shift_int:id_start + len_glitch + shift_int] += glitch

        noisy_glitch_ts.append(noisy_glitch)
        pure_noise_ts.append(background)

    noisy_glitch_ts = np.asarray(noisy_glitch_ts)
    pure_noise_ts = np.asarray(pure_noise_ts)

    mask = ~np.any(
        np.isnan(noisy_glitch_ts) | np.isinf(noisy_glitch_ts)
        | (np.abs(noisy_glitch_ts) > np.finfo(np.float64).max),
        axis=1,
    )
    return noisy_glitch_ts[mask], pure_noise_ts[mask]


def _generate_hdf5(args):
    """Chunked generation + streaming-scaled HDF5 output.

    Avoids ever holding a full train/val array in memory: each batch is
    generated, injected, and written straight to disk, and the
    StandardScaler is fit via ``partial_fit`` over the same chunks rather
    than on an in-memory array. Mirrors the pattern already validated for
    the two-detector separation pipeline (chunked write, explicit
    batch-aligned ``chunks=``, ``compression=None`` for read speed,
    ``float32`` throughout).
    """
    import h5py

    backend = args.noise_backend       # "pycbc" | "bilby" | "gwmock"
    bilby_noise = backend != "pycbc"   # unit-variance whitened noise flag for injection
    duration = args.duration
    length = int(duration * SAMPLE_RATE)
    batch_size = args.batch_size

    noise_ext = f"{backend}_noise_hdf5/"
    noise_type_path = os.path.join(args.output_dir, noise_ext)
    domain_path = os.path.join(noise_type_path, "time_domain")
    os.makedirs(domain_path, exist_ok=True)
    if duration != T and args.output_dir == "data/":
        print(
            f"NOTE: --duration {duration} differs from the {T}s default but "
            f"--output-dir was not overridden — this will write to the same "
            f"path ({domain_path}) as a {T}s run. Pass a distinct --output-dir "
            f"to keep them separate."
        )

    raw_h5_path = os.path.join(domain_path, "strain_data.h5")
    scaled_h5_path = os.path.join(domain_path, "strain_data_scaled.h5")
    scaler_path = os.path.join(noise_type_path, "scaler.pkl")

    mean = 0
    std_dev = np.sqrt(SAMPLE_RATE / 2)  # PyCBC convention: variance = SAMPLE_RATE / 2

    # ── 1) Generate + write raw (unscaled) datasets in batches ──────────────
    with h5py.File(raw_h5_path, "w") as f:
        for split, n_total in [("train", args.num_train), ("val", args.num_val)]:
            chunks = (min(batch_size, n_total), length)
            glitch_ds = f.create_dataset(
                f"noisy_glitch_{split}", shape=(0, length), maxshape=(None, length),
                dtype=np.float32, chunks=chunks,
            )
            bg_ds = f.create_dataset(
                f"background_{split}", shape=(0, length), maxshape=(None, length),
                dtype=np.float32, chunks=chunks,
            )

            wptr = 0
            for start in tqdm(range(0, n_total, batch_size), desc=f"Generating {split}"):
                bs = min(batch_size, n_total - start)
                if backend == "pycbc":
                    noise_batch = generate_gaussian_noise(
                        mean, std_dev, bs, (length,), bilby_noise=False,
                        duration=duration, minimum_frequency=args.minimum_frequency,
                        show_progress=False,
                    )
                else:
                    noise_batch = generate_detector_noise(
                        bs, detector=args.detector,
                        duration=duration, sample_rate=SAMPLE_RATE,
                        backend=backend,
                        minimum_frequency=args.minimum_frequency,
                        psd_file=args.psd_file,
                        show_progress=False,
                    )
                glitch_batch, bg_batch = generate_synthetic_data(
                    noise_batch, bilby_noise, split, sample_rate=SAMPLE_RATE,
                    show_progress=False,
                )
                n_written = glitch_batch.shape[0]  # may be < bs if NaN/Inf samples were dropped
                glitch_ds.resize(wptr + n_written, axis=0)
                bg_ds.resize(wptr + n_written, axis=0)
                glitch_ds[wptr:wptr + n_written] = glitch_batch.astype(np.float32)
                bg_ds[wptr:wptr + n_written] = bg_batch.astype(np.float32)
                wptr += n_written

            if wptr < n_total:
                print(f"NOTE: {split} got {wptr}/{n_total} samples "
                      f"({n_total - wptr} dropped as NaN/Inf).")

    # ── 2) Stream-fit StandardScaler on noisy_glitch_train ──────────────────
    scaler = StandardScaler()
    with h5py.File(raw_h5_path, "r") as f:
        ds = f["noisy_glitch_train"]
        for start in tqdm(range(0, ds.shape[0], batch_size), desc="Fitting scaler"):
            end = min(start + batch_size, ds.shape[0])
            scaler.partial_fit(ds[start:end].reshape(-1, 1))

    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)

    # ── 3) Chunked transform → scaled HDF5 ───────────────────────────────────
    with h5py.File(raw_h5_path, "r") as fin, h5py.File(scaled_h5_path, "w") as fout:
        for key in fin.keys():
            shape = fin[key].shape
            chunks = (min(batch_size, shape[0]), *shape[1:])
            out_ds = fout.create_dataset(key, shape=shape, dtype=np.float32, chunks=chunks)
            for start in tqdm(range(0, shape[0], batch_size), desc=f"Scaling {key}"):
                end = min(start + batch_size, shape[0])
                chunk = fin[key][start:end]
                scaled = scaler.transform(chunk.reshape(-1, 1)).reshape(chunk.shape)
                out_ds[start:end] = scaled.astype(np.float32)

    print(f"Saved raw HDF5:    {raw_h5_path}")
    print(f"Saved scaled HDF5: {scaled_h5_path}")
    print(f"Saved scaler:      {scaler_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic time-domain training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default="data/", help="Root output directory.")
    parser.add_argument("--num-train", type=int, default=250000)
    parser.add_argument("--num-val", type=int, default=25000)
    parser.add_argument(
        "--bilby-noise", action="store_true",
        help="Legacy flag: equivalent to --noise-backend bilby.",
    )
    parser.add_argument(
        "--noise-backend", choices=["pycbc", "bilby", "gwmock"], default=None,
        help="Noise source. 'pycbc' = fast white Gaussian noise; 'bilby' = "
             "PSD-coloured via bilby (any detector in bilby's database); "
             "'gwmock' = coloured via gwmock-noise (requires --psd-file). "
             "Defaults to 'bilby' if --bilby-noise is set, else 'pycbc'.",
    )
    parser.add_argument(
        "--detector", type=str, default="L1",
        help="Interferometer name for bilby/gwmock backends. "
             "bilby: H1, L1, V1, ET1, ET2, ET3, CE, KAGRA, … "
             "gwmock: any label; noise is shaped by --psd-file.",
    )
    parser.add_argument(
        "--psd-file", type=str, default=None,
        help="Path to a two-column (frequency, PSD) ASCII file. "
             "Required for gwmock backend; overrides bilby's built-in curve if set.",
    )
    parser.add_argument(
        "--duration", type=float, default=T, help="Window duration in seconds."
    )
    parser.add_argument(
        "--minimum-frequency", type=float, default=MINIMUM_FREQUENCY,
        help="Low-frequency cutoff in Hz (bilby/gwmock backends).",
    )
    parser.add_argument(
        "--format", choices=["npy", "hdf5"], default="npy",
        help="Output format. 'hdf5' streams generation/scaling in chunks "
             "(--batch-size) and never holds the full dataset in memory — "
             "use for large/long-duration datasets.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048,
        help="Chunk size for --format hdf5 generation, scaler fitting, and writes.",
    )
    args = parser.parse_args()

    # Resolve noise backend: explicit --noise-backend takes priority;
    # fall back to bilby if --bilby-noise set, else pycbc.
    if args.noise_backend is None:
        args.noise_backend = "bilby" if args.bilby_noise else "pycbc"

    if args.format == "hdf5":
        _generate_hdf5(args)
        return

    # ── Legacy npy path ────────────────────────────────────────────────────────
    backend = args.noise_backend
    unit_variance = backend != "pycbc"  # bilby + gwmock both produce unit-variance noise
    duration = args.duration
    length = int(duration * SAMPLE_RATE)
    noise_type_path = os.path.join(args.output_dir, f"{backend}_noise/")
    domain_path = os.path.join(noise_type_path, "time_domain")
    os.makedirs(domain_path, exist_ok=True)
    if duration != T and args.output_dir == "data/":
        print(
            f"NOTE: --duration {duration} differs from the {T}s default but "
            f"--output-dir was not overridden. Pass a distinct --output-dir "
            f"to keep them separate."
        )

    mean = 0
    std_dev = np.sqrt(SAMPLE_RATE / 2)  # PyCBC convention: variance = SAMPLE_RATE / 2

    if backend == "pycbc":
        train_noise = generate_gaussian_noise(
            mean, std_dev, args.num_train, (length,), bilby_noise=False,
            duration=duration, minimum_frequency=args.minimum_frequency,
        )
        val_noise = generate_gaussian_noise(
            mean, std_dev, args.num_val, (length,), bilby_noise=False,
            duration=duration, minimum_frequency=args.minimum_frequency,
        )
    else:
        train_noise = generate_detector_noise(
            args.num_train, detector=args.detector, duration=duration,
            sample_rate=SAMPLE_RATE, backend=backend,
            minimum_frequency=args.minimum_frequency, psd_file=args.psd_file,
        )
        val_noise = generate_detector_noise(
            args.num_val, detector=args.detector, duration=duration,
            sample_rate=SAMPLE_RATE, backend=backend,
            minimum_frequency=args.minimum_frequency, psd_file=args.psd_file,
        )

    glitch_train, bg_train = generate_synthetic_data(train_noise, unit_variance, "train", sample_rate=SAMPLE_RATE)
    glitch_val, bg_val = generate_synthetic_data(val_noise, unit_variance, "val", sample_rate=SAMPLE_RATE)

    scaler = StandardScaler()
    glitch_train_scaled = scaler.fit_transform(glitch_train.reshape(-1, 1)).reshape(glitch_train.shape)
    bg_train_scaled = scaler.transform(bg_train.reshape(-1, 1)).reshape(bg_train.shape)
    glitch_val_scaled = scaler.transform(glitch_val.reshape(-1, 1)).reshape(glitch_val.shape)
    bg_val_scaled = scaler.transform(bg_val.reshape(-1, 1)).reshape(bg_val.shape)

    with open(os.path.join(noise_type_path, f"scaler_{backend}.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    np.save(os.path.join(domain_path, "glitch_train_scaled"), glitch_train_scaled)
    np.save(os.path.join(domain_path, "background_train_scaled"), bg_train_scaled)
    np.save(os.path.join(domain_path, "glitch_val_scaled"), glitch_val_scaled)
    np.save(os.path.join(domain_path, "background_val_scaled"), bg_val_scaled)

    print("Done. Data saved to", domain_path)


if __name__ == "__main__":
    main()
