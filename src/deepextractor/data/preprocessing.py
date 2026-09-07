import numpy as np
import h5py


class ChannelStandardScaler:
    """Per-channel standard scaler for (N, C, T) time-series data.

    Fits one mean and std per channel across all samples and time points.
    The mean_ and scale_ attributes (shape (C,)) are compatible with the
    HDF5Dataset input_scaler interface.

    Compatible with joblib.dump / pickle for serialisation.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_channels_ = None

    def fit(self, X: np.ndarray) -> "ChannelStandardScaler":
        """Fit on array X of shape (N, C, T).

        For large datasets use fit_from_hdf5 instead to avoid loading
        everything into memory.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3-D array (N, C, T), got shape {X.shape}")
        # Mean and std over axes 0 (samples) and 2 (time), per channel
        self.mean_ = X.mean(axis=(0, 2))        # (C,)
        self.scale_ = X.std(axis=(0, 2))        # (C,)
        self.scale_[self.scale_ == 0] = 1.0     # guard against zero-variance channels
        self.n_channels_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        mean = self.mean_[np.newaxis, :, np.newaxis]    # (1, C, 1)
        scale = self.scale_[np.newaxis, :, np.newaxis]  # (1, C, 1)
        return (X - mean) / scale

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        mean = self.mean_[np.newaxis, :, np.newaxis]
        scale = self.scale_[np.newaxis, :, np.newaxis]
        return X * scale + mean

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def fit_from_hdf5(
        self, hdf5_path: str, key: str, chunk_size: int = 2048
    ) -> "ChannelStandardScaler":
        """Fit on a dataset too large to load at once.

        Computes per-channel mean and variance in two online passes over the
        HDF5 dataset — first pass for the mean, second for the variance.
        Memory usage is O(chunk_size * C * T) rather than O(N * C * T).

        Args:
            hdf5_path: Path to the HDF5 file.
            key: Dataset key with shape (N, C, T).
            chunk_size: Number of samples to process at a time.
        """
        with h5py.File(hdf5_path, "r") as f:
            ds = f[key]
            n, c, t = ds.shape

            # Pass 1: mean per channel
            sum_ = np.zeros(c, dtype=np.float64)
            count = np.float64(n * t)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = ds[start:end].astype(np.float64)  # (batch, C, T)
                sum_ += chunk.sum(axis=(0, 2))
            mean_ = sum_ / count

            # Pass 2: variance per channel
            sq_sum = np.zeros(c, dtype=np.float64)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk = ds[start:end].astype(np.float64)
                diff = chunk - mean_[np.newaxis, :, np.newaxis]
                sq_sum += (diff ** 2).sum(axis=(0, 2))
            var_ = sq_sum / count

        self.mean_ = mean_.astype(np.float32)
        self.scale_ = np.sqrt(var_).astype(np.float32)
        self.scale_[self.scale_ == 0] = 1.0
        self.n_channels_ = c
        return self

    def fit_from_separation_shards(
        self, shard_dir: str, split: str, detectors=("H1", "L1", "V1"),
        chunk_size: int = 2048, per_channel: bool = False,
    ) -> "ChannelStandardScaler":
        """Fit on the sharded output of ``generate_separation_data.py``.

        Same two-pass online algorithm as :meth:`fit_from_hdf5`, but reads
        per-detector ``noisy_{det}_{split}`` datasets across every
        ``separation_shard_*.h5`` file in ``shard_dir`` instead of one
        ``(N, C, T)`` array in a single file.

        Args:
            shard_dir: Directory containing ``separation_shard_*.h5`` files.
            split: ``"train"`` or ``"val"``.
            detectors: Detector names, in channel order.
            chunk_size: Number of samples to process at a time, per shard.
            per_channel: If False (default), fit one scalar mean/std across
                all detector channels combined and broadcast it to every
                channel — matches the validated prior pipeline (all detector
                channels flattened into one ``StandardScaler().fit(x.reshape(-1, 1))``
                column; targets are never scaled). If True, fit a separate
                mean/std per detector channel instead. ``mean_``/``scale_``
                are always returned shaped ``(len(detectors),)`` either way,
                so downstream code (:class:`HDF5SeparationDataset`) doesn't
                need to know which mode was used.
        """
        import glob
        import os

        shard_paths = sorted(glob.glob(os.path.join(shard_dir, "separation_shard_*.h5")))
        if not shard_paths:
            raise FileNotFoundError(f"No separation_shard_*.h5 files found in {shard_dir}")

        c = len(detectors)
        keys = [f"noisy_{det}_{split}" for det in detectors]
        reduce_axes = (0, 1, 2) if not per_channel else (0, 2)

        # Pass 1: mean (per channel, or one combined scalar broadcast to all channels)
        sum_ = np.zeros(c, dtype=np.float64)
        count = np.float64(0)
        for path in shard_paths:
            with h5py.File(path, "r") as f:
                n = f[keys[0]].shape[0]
                if n == 0:
                    continue
                t = f[keys[0]].shape[1]
                count += n * t if per_channel else n * t * c
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    chunk = np.stack(
                        [f[key][start:end] for key in keys], axis=1,
                    ).astype(np.float64)  # (batch, C, T)
                    sum_ += np.broadcast_to(chunk.sum(axis=reduce_axes), (c,))
        mean_ = sum_ / count

        # Pass 2: variance (per channel, or one combined scalar broadcast to all channels)
        sq_sum = np.zeros(c, dtype=np.float64)
        for path in shard_paths:
            with h5py.File(path, "r") as f:
                n = f[keys[0]].shape[0]
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    chunk = np.stack(
                        [f[key][start:end] for key in keys], axis=1,
                    ).astype(np.float64)
                    diff = chunk - mean_[np.newaxis, :, np.newaxis]
                    sq_sum += np.broadcast_to((diff ** 2).sum(axis=reduce_axes), (c,))
        var_ = sq_sum / count

        self.mean_ = mean_.astype(np.float32)
        self.scale_ = np.sqrt(var_).astype(np.float32)
        self.scale_[self.scale_ == 0] = 1.0
        self.n_channels_ = c
        return self

    def _check_fitted(self):
        if self.mean_ is None:
            raise RuntimeError("Scaler is not fitted. Call fit() or fit_from_hdf5() first.")
