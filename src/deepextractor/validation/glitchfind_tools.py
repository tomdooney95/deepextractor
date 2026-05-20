# Tools from glitchfind to perform validation of DeepExtractor output

import numpy as np
from scipy.optimize import curve_fit
from gwpy.timeseries import TimeSeries

SAMPLE_RATE = 4096

def _monoLog(x, m, t):
    return np.log(m) - (t * x)


def num_exp(x, L, m, t):
    return L * m / t * np.exp(-t * x)


def p_val(x, L, params):
    rate = num_exp(x, L, *params)
    return 1 - np.exp(-rate)

def extract_p_values(
    noisy_glitch_ts,
    reconstructed_background_ts,
    q_fixed,
    f_low,
    f_high,
    sample_rate=SAMPLE_RATE,
    gps_time=1234567890.0,
    q_low=4,
    q_high=40,
    min_bin=2.0,
    max_bin=40.0,
    fit_bins=15,
    p0=(3, 0.5)
):

    bins = np.linspace(min_bin, max_bin, int((max_bin - min_bin) * 2 + 1))
    bin_mids = (bins[:-1] + bins[1:]) / 2

    ts_dict = {
        "Noise+Glitch": np.asarray(noisy_glitch_ts),
        "Reconstructed Background": np.asarray(reconstructed_background_ts),
    }

    p_values_q = {}
    duration = len(noisy_glitch_ts) / sample_rate

    for label, ts_data in ts_dict.items():

        try:
            ts = TimeSeries(
                ts_data,
                sample_rate=sample_rate,
                t0=gps_time - duration / 2,
            )

            if q_fixed:
                q_low = 20
                q_high = 20
                
            qgram = (
                ts.q_gram(
                    qrange=(q_low, q_high),
                    snrthresh=2, # Lower inclusive threshold on individual tile SNR to keep in the table
                    frange=(f_low, f_high),
                    mismatch=0.5,
                )
                .filter(
                    "time > %.2f" % (gps_time - duration / 2),
                    "time < %.2f" % (gps_time + duration / 2),
                )
            )

            # Extract the used Q value
            qspec = ts.q_transform(
                qrange = (q_low, q_high),
                frange = (f_low, f_high),
                whiten = False,
                mismatch = 0.5
            )

            q = qspec.q

            # Number of time-frequency tiles
            energies = qgram["energy"]

            # Not enough statistics (we have fewer data points than bins with which we want to fit data)
            if len(energies) < fit_bins:
                p_values[label] = np.nan
                continue

            # Build histogram of Q-energy density
            n, _ = np.histogram(energies, bins=bins, density=True)

            # Require strictly positive bins (we do not want empty bins to fit)
            if np.count_nonzero(n[:fit_bins]) < fit_bins // 2:
                p_values[label] = np.nan
                continue

            fit_yvals = np.log(np.clip(n[:fit_bins], 1e-4, None)) # to prevent log(0)
            xvals = bin_mids[:fit_bins]

            params, _ = curve_fit(
                _monoLog,
                xvals,
                fit_yvals,
                p0=p0,
                bounds=([1e-12, 0], [np.inf, np.inf]) # bounds for normalization and decay parameter respectively (to avoid again log(0))
            )

            m, t0_fit = params
            max_energy = np.max(energies)
            p = p_val(max_energy, len(energies), (m, t0_fit))

            p_values_q[label] = [p, q]

        except Exception:
            p_values_q[label] = np.nan

    return p_values_q