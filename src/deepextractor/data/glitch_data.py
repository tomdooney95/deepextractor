"""Utilities for downloading DeepExtractor glitch reconstructions from HuggingFace."""

from __future__ import annotations

from pathlib import Path

REPO_ID = "tomdooney/deepextractor-glitch-reconstructions"

_FILES = {
    "samples": "glitch_GAN_samples_scaled_balanced.npy",
    "labels": "glitch_GAN_labels_balanced.npy",
    "label_order": "glitch_GAN_label_order.npy",
    "derivatives": "glitch_GAN_deriv_samples_balanced.npy",
}


def download_glitch_data(
    data_dir: str | Path = "data/",
    include_derivatives: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Download DeepExtractor glitch reconstructions from HuggingFace.

    Downloads time-domain reconstructions of seven LIGO O3 glitch classes
    (35,000 samples total: Blip, Fast Scattering, Koi Fish, Low Frequency
    Burst, Scattered Light, Tomte, Whistle) from
    ``tomdooney/deepextractor-glitch-reconstructions`` on HuggingFace.

    These reconstructions were produced by DeepExtractor and are suitable
    for training generative models such as GlitchGAN.

    Parameters
    ----------
    data_dir:
        Directory to save downloaded files. Created if it does not exist.
    include_derivatives:
        If ``True``, also download the first-order time-derivative array
        required for cDVGAN training (adds ~2.1 GB).
    force:
        If ``True``, re-download files even if they already exist locally.

    Returns
    -------
    dict[str, Path]
        Mapping of ``{"samples", "labels", "label_order"}`` (and
        ``"derivatives"`` if requested) to their local file paths.

    Examples
    --------
    >>> from deepextractor.data import download_glitch_data
    >>> paths = download_glitch_data("data/glitches/")
    >>> import numpy as np
    >>> X = np.load(paths["samples"])          # (35000, 8192)
    >>> y = np.load(paths["labels"])           # (35000, 7)
    >>> classes = np.load(paths["label_order"], allow_pickle=True)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download data. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    keys = ["samples", "labels", "label_order"]
    if include_derivatives:
        keys.append("derivatives")

    paths: dict[str, Path] = {}
    for key in keys:
        filename = _FILES[key]
        dest = data_dir / filename
        if dest.exists() and not force:
            print(f"  {filename} already exists, skipping.")
            paths[key] = dest
            continue
        print(f"  Downloading {filename} ...")
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(data_dir),
        )
        paths[key] = Path(local)
        print(f"  Saved to {paths[key]}")

    return paths
