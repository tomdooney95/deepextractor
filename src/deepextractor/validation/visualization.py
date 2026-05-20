import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.optimize import curve_fit
from gwpy.timeseries import TimeSeries

from glitchfind_tools import _monoLog, p_val
from deepextractor.utils.visualization import plot_q_transform

SAMPLE_RATE = 4096
NYQUIST = SAMPLE_RATE / 2

# Energy distributions with p-values
def plot_hist(
    ax,
    noisy_ts,
    rec_bkg,
    q_low,
    q_high,
    f_low,
    f_high,
    title,
):
    bins = np.linspace(2.0, 40.0, int((40.0 - 2.0) * 2 + 1))
    bin_mids = (bins[:-1] + bins[1:]) / 2
    fit_bins = 15
    p0 = (3, 0.5)

    colors = ["C0", "C2"]

    q_noisy = None
    duration = len(noisy_ts) / SAMPLE_RATE
    ts_noisy = TimeSeries(
        noisy_ts,
        sample_rate=SAMPLE_RATE,
        t0=1234567890.0 - duration / 2,
    )
    try:
        qspec = ts_noisy.q_transform(
            qrange=(q_low, q_high),
            frange=(f_low, f_high),
            whiten=False,
            mismatch=0.5,
        )
        q_noisy = qspec.q
    except Exception as e:
        print(f"[plot_hist] q_transform failed: {e}")

    q_values = {}
    p_values = {}

    for (label, ts_data), color in zip(
        {
            "Noise+Glitch": noisy_ts,
            "Reconstructed Background": rec_bkg,
        }.items(),
        colors,
    ):
        duration = len(ts_data) / SAMPLE_RATE
        ts = TimeSeries(
            ts_data,
            sample_rate=SAMPLE_RATE,
            t0=1234567890.0 - duration / 2,
        )

        if label == "Reconstructed Background" and q_noisy is not None:
            effective_q_low  = q_noisy
            effective_q_high = q_noisy
        else:
            effective_q_low  = q_low
            effective_q_high = q_high

        try:
            qgram = ts.q_gram(
                qrange=(effective_q_low, effective_q_high),
                snrthresh=2,
                frange=(f_low, f_high),
                mismatch=0.5,
            ).filter(
                f"time > {1234567890.0 - duration / 2}",
                f"time < {1234567890.0 + duration / 2}",
            )
            energies = qgram["energy"]

            if len(energies) < fit_bins:
                print(f"[plot_hist] {label}: not enough tiles ({len(energies)} < {fit_bins})")
                p_values[label] = np.nan
                continue

            n, _ = np.histogram(energies, bins=bins, density=True)

            if np.count_nonzero(n[:fit_bins]) < fit_bins // 2:
                print(f"[plot_hist] {label}: too many empty bins in fit range")
                p_values[label] = np.nan
                continue

            fit_yvals = np.log(np.clip(n[:fit_bins], 1e-4, None))
            xvals = bin_mids[:fit_bins]

            params, _ = curve_fit(
                _monoLog,
                xvals,
                fit_yvals,
                p0=p0,
                bounds=([1e-12, 0], [np.inf, np.inf]),
            )

            p_values[label] = p_val(np.max(energies), len(energies), params)
            q_values[label] = q_noisy if q_noisy is not None else float("nan")

            ax.hist(
                energies,
                bins=bins,
                density=True,
                alpha=0.25,
                histtype="stepfilled",
                color=color,
            )
            ax.plot(
                bin_mids,
                np.exp(_monoLog(bin_mids, *params)),
                "--",
                color=color,
                lw=1.3,
            )

        except Exception as e:
            print(f"[plot_hist] {label}: exception: {e}")
            p_values[label] = np.nan
            q_values[label] = float("nan")

    ax.axvline(bin_mids[fit_bins - 1], color="orangered", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Energy", fontsize=18)
    ax.set_ylabel("Normalized tile energy", fontsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    q_display = q_values.get("Noise+Glitch", float("nan"))
    p_noisy   = p_values.get("Noise+Glitch", float("nan"))
    p_bkg     = p_values.get("Reconstructed Background", float("nan"))
    txt = (
        f"Noise+Glitch:  p={p_noisy:.2f}, Q={q_display:.1f}\n"
        f"Rec. Bkg:      p={p_bkg:.2f},   Q={q_display:.1f} (fixed)"
    )
    ax.text(0.275, 0.95, txt, transform=ax.transAxes, va="top", fontsize=18)

# Complete glitch reconstruction and validation
def plot_5x2(
    metadata,
    glitchy_ts,
    glitchy_ts_complete,
    rec_bkg_range,
    input_stft,
    out_stft,
    p_values_range,
    f_low,
    f_high,
    q_low=4,
    q_high=40,
):
    N_ROWS = 2
    N_COLS = 5
    height_ratios = [2.5, 2.5]

    meta_txt = ""
    for k, val in metadata.items():
        if isinstance(val, (bytes, np.bytes_)):
            val = val.decode()
        elif isinstance(val, (float, np.floating)):
            val = f"{val:.3f}"
        meta_txt += f"{k}: {val}\n"

    input_td  = glitchy_ts
    rec_td    = rec_bkg_range
    extracted = input_td - rec_td

    t = np.linspace(0, len(input_td) / SAMPLE_RATE, len(input_td))

    rec_bkg_complete = glitchy_ts_complete.copy()
    half = SAMPLE_RATE
    mid  = len(rec_bkg_complete) // 2
    rec_bkg_complete[mid - half : mid + half] -= extracted

    T    = len(rec_bkg_complete) / SAMPLE_RATE
    crop = (T / 2, 2)

    in_mag  = 20 * np.log10(input_stft[0] + 1e-8)
    in_ph   = input_stft[1]
    out_mag = 20 * np.log10(out_stft[0] + 1e-8)
    out_ph  = out_stft[1]

    vmin = min(in_mag.min(), out_mag.min())
    vmax = max(in_mag.max(), out_mag.max())

    time = np.linspace(0, 2, in_mag.shape[1])
    freq = np.linspace(0, NYQUIST, in_mag.shape[0])

    fig = plt.figure(figsize=(8 * N_COLS, 7 * N_ROWS))
    gs  = fig.add_gridspec(
        N_ROWS, N_COLS,
        hspace=0.3,
        wspace=0.3,
    )

    # ================= ROW 1 =================
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.text(0, 1, meta_txt, va="top", fontsize=22)

    ax = fig.add_subplot(gs[0, 1])
    ax.grid(alpha=0.4)
    ax.plot(t, rec_td, color="grey", alpha=0.8)
    ax.plot(t, extracted, color="C1")
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Amplitude", fontsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    ax = fig.add_subplot(gs[0, 2])
    plot_q_transform(glitchy_ts_complete, crop=crop, ax=ax)
    ax.set_title("Noise+Glitch Q-scan", fontsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(
        in_mag, aspect="auto", origin="lower",
        extent=[time[0], time[-1], freq[0], freq[-1]],
        vmin=vmin, vmax=vmax, cmap="magma",
    )
    ax.set_title("Noise+Glitch magnitude spectrogram", fontsize=20)
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Frequency (Hz)", fontsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    ax = fig.add_subplot(gs[0, 4])
    im = ax.imshow(
        in_ph, aspect="auto", origin="lower",
        extent=[time[0], time[-1], freq[0], freq[-1]],
        vmin=-np.pi, vmax=np.pi, cmap="twilight",
    )
    ax.set_title("Noise+Glitch phase spectrogram", fontsize=20)
    fig.colorbar(im, ax=ax, label="Phase (radians)")
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Frequency (Hz)", fontsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    # ================= ROW 2 =================
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    plot_hist(
        ax, input_td, rec_td,
        q_low, q_high, f_low, f_high,
        f"Q in [{q_low}, {q_high}]",
    )

    ax = fig.add_subplot(gs[1, 2])
    plot_q_transform(rec_bkg_complete, crop=crop, ax=ax)
    ax.set_title("Reconstructed Bkg Q-scan", fontsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(
        out_mag, aspect="auto", origin="lower",
        extent=[time[0], time[-1], freq[0], freq[-1]],
        vmin=vmin, vmax=vmax, cmap="magma",
    )
    ax.set_title("Reconstructed Bkg magnitude spectrogram", fontsize=20)
    fig.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Frequency (Hz)", fontsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    ax = fig.add_subplot(gs[1, 4])
    im = ax.imshow(
        out_ph, aspect="auto", origin="lower",
        extent=[time[0], time[-1], freq[0], freq[-1]],
        vmin=-np.pi, vmax=np.pi, cmap="twilight",
    )
    ax.set_title("Reconstructed Bkg phase spectrogram", fontsize=20)
    fig.colorbar(im, ax=ax, label="Phase (radians)")
    ax.set_xlabel("Time (s)", fontsize=18)
    ax.set_ylabel("Frequency (Hz)", fontsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=16)
    ax.tick_params(axis="both", which="major", labelsize=16)

    # ================= LEGEND =================
    handles = [
        Line2D([0], [0], color="grey", lw=2, label="Reconstructed Bkg"),
        Line2D([0], [0], color="C1",   lw=2, label="Extracted glitch"),
        Patch(facecolor="C0", alpha=0.25, label="Noise+Glitch"),
        Patch(facecolor="C2", alpha=0.25, label="Reconstructed Background"),
        Line2D([0], [0], color="C0", lw=1.3, linestyle="--", label="Noise+Glitch exp. fit"),
        Line2D([0], [0], color="C2", lw=1.3, linestyle="--", label="Reconstructed Bkg exp. fit"),
        Line2D([0], [0], color="orangered", lw=1, label="Last bin used in fit"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=18,
    )

    plt.show()