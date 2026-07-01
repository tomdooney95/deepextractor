"""
Generate synthetic time-domain training data.

Usage::

    deepextractor-generate --output-dir data/ --num-train 250000 --bilby-noise

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
SNR_SCALING_FACTOR_BILBY = 31.970149253731343

SIGNAL_TYPES = ["chirp", "sine", "sine_gaussian", "gaussian_pulse", "ringdown"]
SIGNAL_FUNCTION_MAP = {
    "chirp": generate_chirp,
    "sine": generate_sine,
    "sine_gaussian": generate_sine_gaussian,
    "gaussian_pulse": generate_gaussian_pulse,
    "ringdown": ringdown,
}


def generate_gaussian_noise(mean, std_dev, num_samples, sample_shape):
    """Generate Gaussian noise samples (pycbc or bilby)."""
    print("Generating pycbc noise...")
    return np.random.normal(loc=mean, scale=std_dev, size=(num_samples, *sample_shape))
    

def generate_gaussian_noise_for_bilby_network(num_samples, ifo_network, sample_rate=SAMPLE_RATE, 
                                              duration=T, minimum_frequency=MINIMUM_FREQUENCY):
    """
    Generate Gaussian noise samples for a network of detectors using bilby. 
    Returns a dictionary with detector names as keys and noise samples as values.
    """
    try:
        import bilby
    except ImportError as e:
        raise ImportError(
            "bilby is required for bilby noise generation. "
            "Install it with: pip install deepextractor[generative]"
        ) from e

    print(f"Found {ifo_network.number_of_interferometers} interferometers in network: {[ifo.name for ifo in ifo_network]}")
    
    gaussian_noise_samples = {}
    for ifo in ifo_network:
        gaussian_noise_samples[ifo.name] = []

    for i in tqdm(range(num_samples), desc="Generating bilby network noise..."):
        for ifo in ifo_network:
            ifo.minimum_frequency = minimum_frequency
        ifo_network.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sample_rate,
            duration=duration,
            start_time=0,
        )

        for ifo in ifo_network:
            white_time_domain_strain = list(ifo.whitened_time_domain_strain)
            gaussian_noise_samples[ifo.name].append(white_time_domain_strain)

    for ifo in ifo_network:
        gaussian_noise_samples[ifo.name] = np.asarray(gaussian_noise_samples[ifo.name])
    return gaussian_noise_samples


def generate_synthetic_data(gaussian_noise_samples, bilby_noise=False, phase="train",
                             t_min=0.125, t_max=2.0, snr_min=SNR_MIN, snr_max=SNR_MAX):
    """Generate synthetic noisy glitch and background data arrays."""
    noisy_glitch_ts = []
    pure_noise_ts = []

    for i in tqdm(range(len(gaussian_noise_samples)),
                  desc=f"Generating Synthetic {phase.capitalize()} Data"):
        background = gaussian_noise_samples[i]
        noisy_glitch = background.copy()
        n_injs = np.random.randint(1, 30)
        for _ in range(n_injs):
            snr_to_scale = np.random.uniform(snr_min, snr_max)
            if bilby_noise:
                snr_to_scale = snr_to_scale / SNR_SCALING_FACTOR_BILBY
            duration = np.random.uniform(t_min, t_max)
            s_type = random.choice(SIGNAL_TYPES)
            _, signal_injection = SIGNAL_FUNCTION_MAP[s_type](duration)
            len_glitch = len(signal_injection)
            id_start = int((T_INJ * SAMPLE_RATE / LENGTH) * len(background)) - len_glitch // 2
            glitch = signal_injection - np.mean(signal_injection)
            glitch = whitened_snr_scaling(glitch, snr=snr_to_scale)
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic time-domain training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default="data/", help="Root output directory.")
    parser.add_argument("--num-train", type=int, default=250000)
    parser.add_argument("--num-val", type=int, default=25000)
    parser.add_argument(
        "--bilby-noise", action="store_true", help="Use bilby noise instead of pycbc."
    )
    args = parser.parse_args()

    bilby_noise = args.bilby_noise
    noise_ext = "bilby_noise/" if bilby_noise else "pycbc_noise/"
    ext = "bilby" if bilby_noise else "pycbc"
    noise_type_path = os.path.join(args.output_dir, noise_ext)
    domain_path = os.path.join(noise_type_path, "time_domain")
    os.makedirs(domain_path, exist_ok=True)

    mean = 0
    std_dev = np.sqrt(SAMPLE_RATE / 2)  # PyCBC convention: variance = SAMPLE_RATE / 2

    if bilby_noise:
        try:
            import bilby
        except ImportError as e:
            raise ImportError(
                "bilby is required for bilby noise generation. "
                "Install it with: pip install deepextractor[generative]"
            ) from e

        ifo_network = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
        train_noise_dict = generate_gaussian_noise_for_bilby_network(args.num_train, ifo_network)
        val_noise_dict = generate_gaussian_noise_for_bilby_network(args.num_val, ifo_network)

        # For simplicity, we will use only L1 noise for training and validation
        train_noise = train_noise_dict["L1"]
        val_noise = val_noise_dict["L1"]
    else:
        train_noise = generate_gaussian_noise(mean, std_dev, args.num_train, (LENGTH,))
        val_noise = generate_gaussian_noise(mean, std_dev, args.num_val, (LENGTH,))

    glitch_train, bg_train = generate_synthetic_data(train_noise, bilby_noise, "train")
    glitch_val, bg_val = generate_synthetic_data(val_noise, bilby_noise, "val")

    scaler = StandardScaler()
    glitch_train_scaled = scaler.fit_transform(glitch_train.reshape(-1, 1)).reshape(glitch_train.shape)
    bg_train_scaled = scaler.transform(bg_train.reshape(-1, 1)).reshape(bg_train.shape)
    glitch_val_scaled = scaler.transform(glitch_val.reshape(-1, 1)).reshape(glitch_val.shape)
    bg_val_scaled = scaler.transform(bg_val.reshape(-1, 1)).reshape(bg_val.shape)

    with open(os.path.join(noise_type_path, f"scaler_{ext}.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    np.save(os.path.join(domain_path, "glitch_train_scaled"), glitch_train_scaled)
    np.save(os.path.join(domain_path, "background_train_scaled"), bg_train_scaled)
    np.save(os.path.join(domain_path, "glitch_val_scaled"), glitch_val_scaled)
    np.save(os.path.join(domain_path, "background_val_scaled"), bg_val_scaled)

    print("Done. Data saved to", domain_path)


if __name__ == "__main__":
    main()
