"""
Simulated evaluation of DeepExtractor models.

Usage::

    deepextractor-evaluate --model DeepExtractor_257 --checkpoint-dir checkpoints/ \\
        --data-dir data/ --output-dir evaluation/

"""

import argparse
import logging
import os
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

from deepextractor.generation.glitch_functions import (
    generate_cdvgan_glitch,
    generate_chirp,
    generate_gaussian_pulse,
    generate_gengli_glitch,
    generate_sine,
    generate_sine_gaussian,
    ringdown,
)
from deepextractor.models.architectures import (
    Autoencoder1D,
    Autoencoder2D,
    DnCNN1D,
    ModifiedAutoencoder2D,
    UNET1D,
    UNET2D,
)
from deepextractor.utils.checkpoints import CHECKPOINT_BILBY, load_torch_model
from deepextractor.utils.io import load_tf_model
from deepextractor.utils.signal import generate_gaussian_noise, whitened_snr_scaling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 4096
NYQUIST_FREQ = SAMPLE_RATE // 2
T_MIN, T_MAX = 0.125, 2
T = 2.0
T_INJ = T / 2
LENGTH = int(T * SAMPLE_RATE)

MODEL_REGISTRY = {
    "UNET1D": UNET1D(in_channels=1, out_channels=1),
    "UNET1D_glitch": UNET1D(in_channels=1, out_channels=1),
    "UNET1D_diff": UNET1D(in_channels=1, out_channels=2),
    "UNET1D_2channel": UNET1D(in_channels=1, out_channels=2),
    "DnCNN1D": DnCNN1D(),
    "Autoencoder1D": Autoencoder1D(in_channels=1, out_channels=1),
    "Autoencoder2D": Autoencoder2D(in_channels=2, out_channels=2),
    "ModifiedAutoencoder2D": ModifiedAutoencoder2D(in_channels=2, out_channels=2),
    "DeepExtractor_65": UNET2D(in_channels=2, out_channels=2),
    "DeepExtractor_129": UNET2D(in_channels=2, out_channels=2),
    "DeepExtractor_257": UNET2D(in_channels=2, out_channels=2),
}


def _get_stft_params(model_name):
    if model_name in ("UNET2D_noise", "ModifiedAutoencoder2D"):
        n_fft = 256
        win_length = n_fft // 2
        hop_length = win_length // 2
    elif model_name == "UNET2D_65_noise":
        n_fft = 256 // 2
        win_length = n_fft
        hop_length = win_length // 2
    else:
        n_fft = 256 * 2
        win_length = n_fft // 8
        hop_length = win_length // 2
    window = torch.hann_window(win_length)
    return n_fft, hop_length, win_length, window


from deepextractor.utils.stft import apply_stft, apply_istft  # noqa: F401


def prepare_data_for_stft(data, scaler, n_fft, hop_length, win_length, window):
    noisy_glitch_ts = np.asarray(data["noisy_glitch_ts"])
    pure_noise_ts = np.asarray(data["pure_noise_ts"])
    noisy_glitch_scaled = scaler.transform(
        noisy_glitch_ts.reshape(-1, 1)
    ).reshape(noisy_glitch_ts.shape)
    pure_noise_scaled = scaler.transform(
        pure_noise_ts.reshape(-1, 1)
    ).reshape(pure_noise_ts.shape)
    noisy_stft = apply_stft(noisy_glitch_scaled, n_fft, hop_length, win_length, window)
    return noisy_stft, noisy_glitch_scaled, pure_noise_scaled


def calculate_match(true_signal, predicted_signal, sample_rate=SAMPLE_RATE):
    from pycbc.filter.matchedfilter import match
    from pycbc.types import TimeSeries as TimeSeries_pycbc

    true_ts = TimeSeries_pycbc(true_signal, delta_t=1.0 / sample_rate)
    pred_ts = TimeSeries_pycbc(predicted_signal, delta_t=1.0 / sample_rate, dtype="double")
    return match(true_ts, pred_ts)[0]


def generate_glitch_data(signal_type, gaussian_noise_samples, signal_function_map,
                          snr_min=7.5, snr_max=100, bilby_noise=False):
    glitches_ts, clean_glitch_subtract, noisy_glitch_ts, pure_noise_ts, snrs_inj = (
        [], [], [], [], []
    )
    for noise_sample in tqdm(gaussian_noise_samples):
        snr_to_scale = np.random.uniform(snr_min, snr_max)
        background = noise_sample.copy()
        noisy_glitch = background.copy()

        if signal_type in ("chirp", "sine", "sine_gaussian", "gaussian_pulse", "ringdown"):
            duration = np.random.uniform(T_MIN, T_MAX)
            _, signal_injection = signal_function_map[signal_type](duration)
        else:
            _, signal_injection = signal_function_map[signal_type]()

        signal_injection = signal_injection.squeeze()
        if np.isnan(signal_injection).any():
            continue

        len_glitch = len(signal_injection)
        id_start = int((T_INJ * SAMPLE_RATE / LENGTH) * len(noise_sample)) - len_glitch // 2
        glitch = signal_injection - np.mean(signal_injection)
        # bilby's whitened_time_domain_strain is unit-variance; convert to the
        # SAMPLE_RATE/2-variance convention whitened_snr_scaling assumes.
        effective_snr = snr_to_scale / np.sqrt(SAMPLE_RATE / 2) if bilby_noise else snr_to_scale
        glitch = whitened_snr_scaling(glitch, snr=effective_snr)
        noisy_glitch[id_start : id_start + len_glitch] += glitch
        clean_glitch = noisy_glitch - background

        glitches_ts.append(glitch)
        clean_glitch_subtract.append(clean_glitch)
        noisy_glitch_ts.append(noisy_glitch)
        pure_noise_ts.append(background)
        snrs_inj.append(snr_to_scale)

    return {
        "glitches_ts": glitches_ts,
        "clean_glitch_subtract": clean_glitch_subtract,
        "noisy_glitch_ts": noisy_glitch_ts,
        "pure_noise_ts": pure_noise_ts,
        "snr": snrs_inj,
    }


def generate_hybrid_glitch_data(gaussian_noise_samples, signal_function_map,
                                 snr_min=7.5, snr_max=100):
    hybrid_signals = list(signal_function_map.keys())
    clean_glitch_subtract, noisy_glitch_ts, pure_noise_ts = [], [], []

    for noise_sample in tqdm(gaussian_noise_samples):
        background = noise_sample.copy()
        noisy_glitch = background.copy()
        n_injs = np.random.randint(2, 8)

        for _ in range(n_injs):
            snr_to_scale = np.random.uniform(snr_min, snr_max)
            s_type = random.choice(hybrid_signals)
            if s_type in ("chirp", "sine", "sine_gaussian", "gaussian_pulse", "ringdown"):
                duration = np.random.uniform(T_MIN, 1)
                _, signal_injection = signal_function_map[s_type](duration)
            else:
                _, signal_injection = signal_function_map[s_type]()

            signal_injection = signal_injection.squeeze()
            len_glitch = len(signal_injection)
            id_start = int((T_INJ * SAMPLE_RATE / LENGTH) * len(background)) - len_glitch // 2
            glitch = signal_injection - np.mean(signal_injection)
            glitch = whitened_snr_scaling(glitch, snr=snr_to_scale)
            shift_int = np.random.randint(-id_start, len(background) - id_start - len_glitch)
            noisy_glitch[id_start + shift_int : id_start + len_glitch + shift_int] += glitch

        clean_glitch = noisy_glitch - background
        clean_glitch_subtract.append(clean_glitch)
        noisy_glitch_ts.append(noisy_glitch)
        pure_noise_ts.append(background)

    return {
        "glitches_ts": [],
        "clean_glitch_subtract": clean_glitch_subtract,
        "noisy_glitch_ts": noisy_glitch_ts,
        "pure_noise_ts": pure_noise_ts,
        "snr": [],
    }


def evaluate_model(model_name, model_registry, scaler, glitch_data, output_path,
                   checkpoint_dir, device, batch_size=8, checkpoint_filename=CHECKPOINT_BILBY):
    model = load_torch_model(model_name, model_registry, checkpoint_dir, device,
                             checkpoint_filename=checkpoint_filename)
    n_fft, hop_length, win_length, window = _get_stft_params(model_name)

    model_data_dict = {}
    for signal_type, data in glitch_data.items():
        noisy_stft, _, _ = prepare_data_for_stft(
            data, scaler, n_fft, hop_length, win_length, window
        )
        noisy_glitch_ts = np.asarray(data["noisy_glitch_ts"])
        pure_noise_ts = np.asarray(data["pure_noise_ts"])

        metrics_dict = {"match_background": [], "match_glitch": [], "mismatch_glitch": []}
        extracted_signals, background_output = [], []

        for m in range(0, len(noisy_stft), batch_size):
            batch = noisy_stft[m : m + batch_size].to(device)
            noisy_batch = noisy_glitch_ts[m : m + batch_size]
            pure_batch = pure_noise_ts[m : m + batch_size]
            clean_batch = data["clean_glitch_subtract"][m : m + batch_size]

            with torch.no_grad():
                output_val = model(batch.squeeze())

            output_istft = apply_istft(output_val, n_fft, hop_length, win_length, window)
            output_np = output_istft.cpu().numpy().squeeze()
            backgrounds_inv = scaler.inverse_transform(
                output_np.reshape(-1, output_np.shape[-1])
            ).reshape(output_np.shape)
            diff = noisy_batch - backgrounds_inv

            for k in range(len(diff)):
                extracted_signals.append(diff[k])
                background_output.append(backgrounds_inv[k])
                match_bg = calculate_match(pure_batch[k], backgrounds_inv[k])
                match_gl = calculate_match(clean_batch[k], diff[k])
                metrics_dict["match_background"].append(match_bg)
                metrics_dict["match_glitch"].append(match_gl)
                metrics_dict["mismatch_glitch"].append((1 - match_gl) * 100)

        model_data_dict[signal_type] = {
            "metrics": metrics_dict,
            "time_series": {
                "extracted_glitches": [ts.tolist() for ts in extracted_signals],
                "background_outputs": [ts.tolist() for ts in background_output],
            },
        }

    return model_data_dict


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DeepExtractor models on simulated glitch data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["DeepExtractor_257"],
        help="One or more model names to evaluate.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Local checkpoint directory. Falls back to Hugging Face Hub if not set.")
    parser.add_argument("--checkpoint-filename", type=str, default=CHECKPOINT_BILBY,
                        help="Checkpoint file name within the model subdirectory.")
    parser.add_argument("--assets-dir", type=str, default="assets/",
                        help="Directory containing scaler .pkl files.")
    parser.add_argument("--scaler-path", type=str, default=None,
                        help="Path to scaler .pkl. Defaults to <assets-dir>/scaler_bilby.pkl.")
    parser.add_argument("--data-dir", type=str, default="data/")
    parser.add_argument("--output-dir", type=str, default="evaluation/")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--snr-min", type=float, default=7.5)
    parser.add_argument("--snr-max", type=float, default=100.0)
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use. Auto-detected if not set.",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load scaler — default to bilby scaler (matches simulated evaluation)
    scaler_path = args.scaler_path or os.path.join(args.assets_dir, "scaler_bilby.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load CDVGAN generator (optional)
    try:
        generator = load_tf_model(args.data_dir, "cdvgan")
    except Exception as e:
        logger.warning(f"Could not load CDVGAN generator: {e}. CDVGAN signals will be unavailable.")
        generator = None

    signal_function_map = {
        "chirp": generate_chirp,
        "sine": generate_sine,
        "sine_gaussian": generate_sine_gaussian,
        "gaussian_pulse": generate_gaussian_pulse,
        "ringdown": ringdown,
        "gengli_H1": lambda: generate_gengli_glitch(ifo="H1"),
        "gengli_L1": lambda: generate_gengli_glitch(ifo="L1"),
    }
    if generator is not None:
        signal_function_map.update({
            "cdvgan_blip": lambda: generate_cdvgan_glitch("blip", generator),
            "cdvgan_tomte": lambda: generate_cdvgan_glitch("tomte", generator),
            "cdvgan_bbh": lambda: generate_cdvgan_glitch("bbh", generator),
            "cdvgan_simplex": lambda: generate_cdvgan_glitch("simplex", generator),
            "cdvgan_uniform": lambda: generate_cdvgan_glitch("uniform", generator),
        })

    # Generate noise samples
    mean, std_dev = 0, 50
    gaussian_noise_samples = generate_gaussian_noise(mean, std_dev, args.num_samples, (LENGTH,))

    # Generate glitch data per signal type
    glitch_data = {}
    for signal_type in signal_function_map:
        logger.info(f"Generating data for: {signal_type}")
        glitch_data[signal_type] = generate_glitch_data(
            signal_type, gaussian_noise_samples, signal_function_map,
            args.snr_min, args.snr_max,
        )
    glitch_data["hybrid"] = generate_hybrid_glitch_data(
        gaussian_noise_samples, signal_function_map, args.snr_min, args.snr_max
    )

    # Evaluate models
    data_dict = {"data": glitch_data, "model_outputs": {}}
    for model_name in args.model:
        logger.info(f"Evaluating model: {model_name}")
        model_data = evaluate_model(
            model_name, MODEL_REGISTRY, scaler, glitch_data,
            args.output_dir, args.checkpoint_dir, device, args.batch_size,
            checkpoint_filename=args.checkpoint_filename,
        )
        data_dict["model_outputs"][model_name] = model_data

    out_file = os.path.join(args.output_dir, "simulation_results.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(data_dict, f)

    logger.info(f"Evaluation complete. Results saved to {out_file}")


if __name__ == "__main__":
    main()
