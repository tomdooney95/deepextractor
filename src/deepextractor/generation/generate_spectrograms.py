"""
Convert time-domain arrays to STFT spectrograms (magnitude + phase).

Also provides a utility to concatenate chunked spectrogram files.

Usage::

    deepextractor-specgen --input-dir data/pycbc_noise/time_domain/ --output-dir data/pycbc_noise/spectrogram_domain/

    # HDF5 mode -- reads an HDF5 produced by generate_timeseries.py --format hdf5
    # and writes one spectrogram HDF5 file per dataset key into --output-dir.
    # Splitting across files avoids the ~512 GB per-file hard limit on LIGO CIT.
    # Safe to re-run after a partial failure — completed files are skipped.
    deepextractor-specgen --format hdf5 \\
        --input-h5 data_4s/bilby_noise_hdf5/time_domain/strain_data_scaled.h5 \\
        --output-dir data_4s/bilby_noise_hdf5/spectrogram_domain/

"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from deepextractor.utils.stft import apply_stft


# Default STFT parameters (257x257 output shape)
DEFAULT_N_FFT = 256 * 2
DEFAULT_WIN_LENGTH = DEFAULT_N_FFT // 8
DEFAULT_HOP_LENGTH = DEFAULT_WIN_LENGTH // 2


def apply_stft_and_save(
    array_path, save_path, n_fft, hop_length, win_length, window, chunk_size=5000
):
    """Apply STFT to a .npy array in chunks and save the result."""
    array = np.load(array_path)
    print(f"Loaded {array_path}, shape: {array.shape}")

    total_chunks = array.shape[0] // chunk_size
    stft_list = []

    for i in range(0, array.shape[0], chunk_size):
        chunk = array[i : i + chunk_size]
        tensor = torch.tensor(chunk, dtype=torch.float32)
        stft_result = torch.stft(
            tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        magnitude = torch.abs(stft_result)
        phase = torch.angle(stft_result)
        stft_mag_phase = torch.stack([magnitude, phase], dim=1)
        stft_list.append(stft_mag_phase)

        del tensor, stft_result, magnitude, phase
        torch.cuda.empty_cache()

        print(f"Processed chunk {i // chunk_size + 1}/{max(total_chunks, 1)}")

    stft_final = torch.cat(stft_list, dim=0)
    stft_numpy = stft_final.cpu().numpy()
    np.save(save_path, stft_numpy)
    print(f"STFT saved to {save_path}.npy, final shape: {stft_numpy.shape}")

    del array, stft_list, stft_final, stft_numpy
    torch.cuda.empty_cache()


def _generate_hdf5_spectrograms(args):
    """Chunked HDF5 STFT generation — one file per dataset key.

    Writes each key from the input HDF5 to its own output file
    ({output_dir}/{key}.h5) to avoid filesystem per-file size limits
    (~512 GB on LIGO CIT). Each output file contains a single "data"
    dataset of shape (N, 2, F, T). Skips keys whose output file already
    exists and is complete — safe to re-run after a partial failure.
    """
    import h5py

    window = torch.hann_window(args.win_length)
    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.input_h5, "r") as fin:
        for key in fin.keys():
            out_path = os.path.join(args.output_dir, f"{key}.h5")
            in_ds = fin[key]
            n = in_ds.shape[0]

            # Resume support: skip if output file already exists and is complete.
            if os.path.exists(out_path):
                try:
                    with h5py.File(out_path, "r") as fcheck:
                        if "data" in fcheck and fcheck["data"].shape[0] == n:
                            print(f"Skipping {key} — {out_path} already complete ({n} samples).")
                            continue
                except Exception:
                    pass
                print(f"Re-generating {key} — {out_path} exists but is incomplete or corrupt.")

            # Probe output shape from a single dummy sample.
            probe = apply_stft(
                np.zeros((1, in_ds.shape[1]), dtype=np.float32),
                args.n_fft, args.hop_length, args.win_length, window,
            )
            out_shape = (n, *probe.shape[1:])
            # HDF5 hard limit: 4 GB per chunk. At 4s defaults each spectrogram
            # sample is ~1 MB, so cap chunk_n so chunks stay under 2 GB.
            per_sample_bytes = int(np.prod(probe.shape[1:])) * 4
            max_chunk_n = max(1, (2 * 1024 ** 3) // per_sample_bytes)
            chunk_n = min(args.chunk_size, n, max_chunk_n)
            chunks = (chunk_n, *probe.shape[1:])

            with h5py.File(out_path, "w") as fout:
                out_ds = fout.create_dataset("data", shape=out_shape, dtype=np.float32, chunks=chunks)
                for start in tqdm(range(0, n, args.chunk_size), desc=f"STFT {key}"):
                    end = min(start + args.chunk_size, n)
                    stft_chunk = apply_stft(
                        in_ds[start:end], args.n_fft, args.hop_length, args.win_length, window,
                    )
                    out_ds[start:end] = stft_chunk.numpy().astype(np.float32)

            print(f"Saved {out_path}  shape={out_shape}")


def load_and_concatenate_chunks(data_dir, base_filename, total_chunks):
    """Load and concatenate chunked numpy arrays saved as ``{base}_chunk_{i}.npy``."""
    stft_list = []
    for i in range(total_chunks):
        chunk_filename = f"{base_filename}_chunk_{i}.npy"
        chunk_path = os.path.join(data_dir, chunk_filename)
        if os.path.exists(chunk_path):
            print(f"Loading {chunk_filename}...")
            stft_list.append(np.load(chunk_path))
        else:
            print(f"Chunk {chunk_filename} not found. Skipping.")
    print("Concatenating chunks...")
    return np.concatenate(stft_list, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Convert time-domain .npy arrays to STFT spectrogram arrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--format", choices=["npy", "hdf5"], default="npy",
        help="'hdf5' reads/writes HDF5 in chunks, never holding the full "
             "dataset in memory -- use for large/long-duration datasets.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing the time-domain .npy files (--format npy).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the spectrogram .npy files (--format npy).",
    )
    parser.add_argument(
        "--input-h5", type=str,
        help="Path to the time-domain HDF5 file, e.g. strain_data_scaled.h5 (--format hdf5).",
    )
    parser.add_argument("--n-fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--win-length", type=int, default=DEFAULT_WIN_LENGTH)
    parser.add_argument("--hop-length", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--combine-chunks",
        action="store_true",
        help="Combine pre-existing chunk files instead of generating new spectrograms.",
    )
    parser.add_argument(
        "--chunks-glitch-train", type=int, default=16,
        help="Number of chunks for glitch_train (used with --combine-chunks).",
    )
    parser.add_argument(
        "--chunks-background-train", type=int, default=16,
        help="Number of chunks for background_train (used with --combine-chunks).",
    )
    parser.add_argument(
        "--chunks-glitch-val", type=int, default=2,
        help="Number of chunks for glitch_val (used with --combine-chunks).",
    )
    parser.add_argument(
        "--chunks-background-val", type=int, default=2,
        help="Number of chunks for background_val (used with --combine-chunks).",
    )
    args = parser.parse_args()

    if args.format == "hdf5":
        if not args.input_h5 or not args.output_dir:
            parser.error("--format hdf5 requires --input-h5 and --output-dir")
        _generate_hdf5_spectrograms(args)
        return

    if not args.input_dir or not args.output_dir:
        parser.error("--format npy requires --input-dir and --output-dir")

    os.makedirs(args.output_dir, exist_ok=True)
    window = torch.hann_window(args.win_length)

    if args.combine_chunks:
        for base, n_chunks in [
            ("glitch_train_scaled_mag_phase", args.chunks_glitch_train),
            ("background_train_scaled_mag_phase", args.chunks_background_train),
            ("glitch_val_scaled_mag_phase", args.chunks_glitch_val),
            ("background_val_scaled_mag_phase", args.chunks_background_val),
        ]:
            combined = load_and_concatenate_chunks(args.output_dir, base, n_chunks)
            out_path = os.path.join(args.output_dir, f"{base}_combined.npy")
            np.save(out_path, combined)
            print(f"Saved combined {base} to {out_path}")
        print("All combined datasets saved.")
    else:
        datasets = [
            ("glitch_train_scaled.npy", "glitch_train_scaled_mag_phase"),
            ("background_train_scaled.npy", "background_train_scaled_mag_phase"),
            ("glitch_val_scaled.npy", "glitch_val_scaled_mag_phase"),
            ("background_val_scaled.npy", "background_val_scaled_mag_phase"),
        ]
        for in_name, out_name in datasets:
            in_path = os.path.join(args.input_dir, in_name)
            out_path = os.path.join(args.output_dir, out_name)
            apply_stft_and_save(
                in_path, out_path,
                args.n_fft, args.hop_length, args.win_length, window,
                args.chunk_size,
            )
        print("All STFT results saved.")


if __name__ == "__main__":
    main()
