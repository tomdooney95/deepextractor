import numpy as np
import torch
import h5py
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, input_npy, target_npy, transform=None):
        self.inputs = np.load(input_npy)
        self.targets = np.load(target_npy)

        if self.inputs.ndim == 2:
            self.inputs = np.expand_dims(self.inputs, axis=1)
        if self.targets.ndim == 2:
            self.targets = np.expand_dims(self.targets, axis=1)

        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ts = torch.tensor(self.inputs[index], dtype=torch.float32)
        target_ts = torch.tensor(self.targets[index], dtype=torch.float32)

        if self.transform is not None:
            augmentations = self.transform(input_ts=input_ts, target_ts=target_ts)
            input_ts = augmentations["input_ts"]
            target_ts = augmentations["target_ts"]

        return input_ts, target_ts


class SpectrogramDataset(Dataset):
    def __init__(self, input_npy, target_npy, transform=None):
        self.input_path = input_npy
        self.target_path = target_npy

        self.input_shape = np.load(input_npy, mmap_mode="r").shape
        self.target_shape = np.load(target_npy, mmap_mode="r").shape

        self.input_channels_needed = len(self.input_shape) == 3
        self.target_channels_needed = len(self.target_shape) == 3

        self.transform = transform

    def __len__(self):
        return self.input_shape[0]

    def __getitem__(self, index):
        input_ts = np.load(self.input_path, mmap_mode="r")[index]
        target_ts = np.load(self.target_path, mmap_mode="r")[index]

        if self.input_channels_needed:
            input_ts = np.expand_dims(input_ts, axis=0)
        if self.target_channels_needed:
            target_ts = np.expand_dims(target_ts, axis=0)

        input_ts = torch.tensor(input_ts, dtype=torch.float32)
        target_ts = torch.tensor(target_ts, dtype=torch.float32)

        if self.transform is not None:
            augmentations = self.transform(input_ts=input_ts, target_ts=target_ts)
            input_ts = augmentations["input_ts"]
            target_ts = augmentations["target_ts"]

        return input_ts, target_ts


class HDF5ReconstructionDataset(Dataset):
    """HDF5-backed dataset for single-detector glitch reconstruction.

    Supports two layouts produced by ``deepextractor-specgen --format hdf5``:

    * **Per-key files** (default): ``input_h5`` and ``target_h5`` are separate
      ``.h5`` files each containing a single ``"data"`` dataset — the layout
      written by the current spectrogram pipeline to avoid filesystem per-file
      size limits on LIGO CIT.
    * **Single-file** (legacy): pass ``hdf5_path`` plus ``input_key`` /
      ``target_key`` to read two datasets from one file.

    Use ``shuffle=False`` in DataLoader for purely-simulated data — every
    sample is an independent random draw, so storage order carries no
    structure to shuffle away.  For real background data, shuffle once at
    generation time instead (adjacent real samples can be time-correlated).

    Args:
        input_h5:   Path to the input HDF5 file (``"data"`` key).
        target_h5:  Path to the target HDF5 file (``"data"`` key).
        hdf5_path:  Legacy — single file containing both datasets.
        input_key:  Legacy — dataset key for input within ``hdf5_path``.
        target_key: Legacy — dataset key for target within ``hdf5_path``.
        transform:  Optional callable ``transform(input_ts=..., target_ts=...) → dict``.
    """

    def __init__(self, input_h5=None, target_h5=None,
                 hdf5_path=None, input_key="data", target_key="data",
                 transform=None):
        if input_h5 is not None and target_h5 is not None:
            # Per-key file layout (new default)
            self._input_path  = input_h5
            self._target_path = target_h5
            self._input_key   = "data"
            self._target_key  = "data"
            self._shared_file = False
        elif hdf5_path is not None:
            # Legacy single-file layout
            self._input_path  = hdf5_path
            self._target_path = hdf5_path
            self._input_key   = input_key
            self._target_key  = target_key
            self._shared_file = True
        else:
            raise ValueError("Provide either (input_h5, target_h5) or hdf5_path.")

        self.transform = transform
        self._in_file = self._tgt_file = None
        self._input = self._target = self._len = None

    def _ensure_open(self):
        if self._in_file is None:
            self._in_file  = h5py.File(self._input_path,  "r", swmr=True, libver="latest")
            self._tgt_file = (self._in_file if self._shared_file
                              else h5py.File(self._target_path, "r", swmr=True, libver="latest"))
            self._input  = self._in_file[self._input_key]
            self._target = self._tgt_file[self._target_key]
            self._len    = self._input.shape[0]

    def __len__(self):
        if self._len is None:
            self._ensure_open()
        return self._len

    def __getitem__(self, index):
        self._ensure_open()
        x = torch.tensor(self._input[index],  dtype=torch.float32)
        y = torch.tensor(self._target[index], dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        if self.transform is not None:
            aug = self.transform(input_ts=x, target_ts=y)
            x, y = aug["input_ts"], aug["target_ts"]
        return x, y

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_in_file"] = state["_tgt_file"] = state["_input"] = state["_target"] = None
        return state

    def __del__(self):
        try:
            if self._in_file is not None:
                self._in_file.close()
            if not self._shared_file and self._tgt_file is not None:
                self._tgt_file.close()
        except Exception:
            pass


class HDF5Dataset(Dataset):
    """HDF5-backed dataset for time-domain two-detector signal/glitch separation.

    Lazy-opens the HDF5 file per worker process. Use shuffle=False in DataLoader
    — data is pre-shuffled at generation time; random HDF5 seeks are expensive.

    Args:
        hdf5_path: Path to the HDF5 file.
        input_key: Dataset key for the 2-channel (H1+L1) strain inputs.
        background_key: Dataset key for the background (noise) targets.
        signal_key: Dataset key for the signal targets.
        input_scaler: Optional sklearn-compatible scaler (must expose mean_ and
            scale_ attributes, shaped (n_channels,)). Applied to inputs only;
            targets are assumed to be whitened already.
        target_signal_only: If True, return only the signal targets (2-channel).
            If False (default), concatenate [background, signal] → 4-channel target.
        transform: Optional callable with signature
            transform(input_ts=..., target_ts=...) → dict with same keys.
    """

    def __init__(self, hdf5_path, input_key, background_key, signal_key,
                 input_scaler=None, target_signal_only=False, transform=None):
        self.hdf5_path = hdf5_path
        self.input_key = input_key
        self.background_key = background_key
        self.signal_key = signal_key
        self.input_scaler = input_scaler
        self.target_signal_only = target_signal_only
        self.transform = transform
        self._file = self._input = self._bg = self._sig = self._len = None

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, "r", swmr=True, libver="latest")
            self._input = self._file[self.input_key]
            self._bg = self._file[self.background_key]
            self._sig = self._file[self.signal_key]
            self._len = self._input.shape[0]

    def __len__(self):
        if self._len is None:
            self._ensure_open()
        return self._len

    def __getitem__(self, index):
        self._ensure_open()
        x = torch.tensor(self._input[index], dtype=torch.float32)
        bg = torch.tensor(self._bg[index], dtype=torch.float32)
        sig = torch.tensor(self._sig[index], dtype=torch.float32)

        if self.input_scaler is not None:
            mean = torch.tensor(self.input_scaler.mean_, dtype=torch.float32).view(-1, 1)
            scale = torch.tensor(self.input_scaler.scale_, dtype=torch.float32).view(-1, 1)
            x = (x - mean) / scale

        y = sig if self.target_signal_only else torch.cat([bg, sig], dim=0)

        if self.transform is not None:
            aug = self.transform(input_ts=x, target_ts=y)
            x, y = aug["input_ts"], aug["target_ts"]

        return x, y

    def __getstate__(self):
        # HDF5 file handles cannot be pickled — close and reopen per worker
        state = self.__dict__.copy()
        state["_file"] = state["_input"] = state["_bg"] = state["_sig"] = None
        return state

    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass
