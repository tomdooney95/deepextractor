"""
Build whitened background datasets for H1 and L1.

Whitening strategy:
  - Each 36s context window is whitened independently using exactly 36s of data,
    consistent with the procedure used at test time.
  - GWpy's whiten() (fftlength=PSD_DURATION) corrupts fftlength/2 seconds at each
    edge; we manually crop MAX_FILTER_DUR = fftlength/2 seconds from each end,
    leaving a 32s usable window per context.
  - The context window advances by CONTEXT_STRIDE = 32s (the usable window size),
    so usable regions tile without gaps or overlaps — no GPS time is whitened twice.

Sample extraction:
  - A 8s window slides through the 32s usable region with step DELTA_T = 2s,
    yielding 13 overlapping samples per context window.
  - Adjacent samples share 6s of content but are distinct windows.

Train/test split note:
  - Always split by GPS time range, not sample index — overlapping samples within
    a context share the same underlying noise realization.

Example
-------
    python scripts/get_clean_backgrounds.py --runs O3a O3b --target 500 --out backgrounds_test.pkl
    python scripts/get_clean_backgrounds.py --runs O3a O3b O4a O4b --target 40000 --out backgrounds_full.pkl
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from gwdatafind import find_urls
from gwpy.segments import DataQualityFlag, Segment, SegmentList
from gwpy.timeseries import TimeSeries

from deepextractor.data.omicron import RUN_PERIODS

# ── Processing parameters ─────────────────────────────────────────────────────

IFOS             = ['H1', 'L1']
FRAME_TYPE       = {'H1': 'H1_HOFT_C00', 'L1': 'L1_HOFT_C00'}
OMICRON_DIR      = 'triggers/'
SAMPLE_RATE      = 4096   # Hz

CONTEXT_DURATION = 36     # s — data fetched and whitened per window
MAX_FILTER_DUR   = 2      # s — whitening filter edge corruption; removed from each end
PSD_DURATION     = 4      # s — Welch sub-segment length for PSD estimation
CONTEXT_STRIDE   = CONTEXT_DURATION - 2 * MAX_FILTER_DUR  # = 32s usable window

SAMPLE_DURATION  = 8      # s — output sample length
SAMPLE_LENGTH    = SAMPLE_DURATION * SAMPLE_RATE           # 32768 samples
DELTA_T          = 2      # s — sliding step within usable window
DELTA_T_SAMPLES  = DELTA_T * SAMPLE_RATE

SAMPLES_PER_CONTEXT = (CONTEXT_STRIDE - SAMPLE_DURATION) // DELTA_T + 1  # = 13

TRIGGER_BUFFER   = 0.5    # s — safety margin added to each side of a trigger
MIN_SEG_DUR      = CONTEXT_DURATION
MAX_AMP          = 30.0   # whitened noise should be ~N(0,1); reject windows exceeding this

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_glitch_segments(tstarts, tends, buffer):
    """Coalesced SegmentList covering each Omicron tile plus a safety buffer."""
    segs = SegmentList([
        Segment(ts - buffer, te + buffer)
        for ts, te in zip(tstarts, tends)
    ])
    segs.coalesce()
    return segs


def whiten_and_slice(ts):
    """
    Whiten a CONTEXT_DURATION-second GWpy TimeSeries and return overlapping
    SAMPLE_DURATION windows from the usable (edge-trimmed) region, plus the
    PSD used to whiten it.

    The PSD is computed separately via .psd() with the same
    fftlength/overlap/method/window .whiten() uses internally (median-averaged
    Welch, hann window -- gwpy's defaults for both, matched deliberately here
    rather than left implicit) -- .whiten() doesn't expose the PSD it computed,
    so this is a second, cheap call reusing the same parameters, not a
    reconstruction from the whitened output. Saved so injected signals can
    later be whitened against the *same* PSD as the noise they're added to,
    instead of a fixed analytic design curve.
    """
    psd = ts.psd(fftlength=PSD_DURATION, overlap=PSD_DURATION // 2, method='median', window='hann')

    whitened = ts.whiten(fftlength=PSD_DURATION, overlap=PSD_DURATION // 2, highpass=10.0)
    pad = MAX_FILTER_DUR * SAMPLE_RATE
    data = np.array(whitened, dtype=np.float32)[pad:-pad]

    windows = []
    for start in range(0, len(data) - SAMPLE_LENGTH + 1, DELTA_T_SAMPLES):
        w = data[start : start + SAMPLE_LENGTH]
        if np.isfinite(w).all() and np.abs(w).max() < MAX_AMP:
            windows.append(w)
    return windows, psd

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        '--runs', nargs='+', required=True,
        choices=sorted(RUN_PERIODS),
        metavar='RUN',
        help=f'Observing run(s) to process. Choices: {sorted(RUN_PERIODS)}',
    )
    p.add_argument(
        '--target', type=int, default=40_000,
        help='Target number of samples per IFO/run (default: 40000)',
    )
    p.add_argument(
        '--out', type=Path, default=Path('real_backgrounds_dict.pkl'),
        help='Output pickle file (default: real_backgrounds_dict.pkl)',
    )
    return p.parse_args()

# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    target = args.target

    backgrounds = {run: {ifo: [] for ifo in IFOS} for run in args.runs}

    for run in args.runs:
        run_start, run_end = RUN_PERIODS[run]

        for ifo in IFOS:
            print(f"\n{'─' * 60}")
            print(f"  {ifo}  {run}  (GPS {run_start} – {run_end})")
            print(f"  Context: {CONTEXT_DURATION}s  |  Stride: {CONTEXT_STRIDE}s  |  "
                  f"{SAMPLES_PER_CONTEXT} samples/context  |  target: {target}")
            print(f"{'─' * 60}")

            triggers = np.load(f"{OMICRON_DIR}{ifo.lower()}_{run.lower()}_triggers.npz")
            tstarts  = triggers["tstart"]
            tends    = triggers["tend"]

            flag = f'{ifo}:DMT-ANALYSIS_READY:1'
            print(f"  Querying {flag} ...")
            science_segs = DataQualityFlag.query(flag, run_start, run_end).active

            glitch_segs = build_glitch_segments(tstarts, tends, TRIGGER_BUFFER)
            clean_segs  = science_segs - glitch_segs

            n_qualifying = sum(1 for s in clean_segs if float(s[1] - s[0]) >= MIN_SEG_DUR)
            total_clean_h = sum(float(s[1] - s[0]) for s in clean_segs) / 3600
            print(f"  Clean time: {total_clean_h:.1f} h  |  "
                  f"segments >= {MIN_SEG_DUR}s: {n_qualifying}")

            samples   = []
            gps_times = []
            psd_gps_starts = []
            psds = []
            psd_freqs = None
            t0 = time.time()
            failed = 0

            for seg in clean_segs:
                if len(samples) >= target:
                    break
                if float(seg[1] - seg[0]) < MIN_SEG_DUR:
                    continue

                context_start = float(seg[0])
                while context_start + CONTEXT_DURATION <= float(seg[1]):
                    if len(samples) >= target:
                        break

                    try:
                        urls = find_urls(
                            ifo[0], FRAME_TYPE[ifo],
                            context_start, context_start + CONTEXT_DURATION,
                        )
                        ts = TimeSeries.read(
                            urls, channel=f'{ifo}:GDS-CALIB_STRAIN',
                            start=context_start, end=context_start + CONTEXT_DURATION,
                        )
                        if ts.sample_rate.value != SAMPLE_RATE:
                            ts = ts.resample(SAMPLE_RATE)

                        new, psd = whiten_and_slice(ts)
                        samples.extend(new)
                        gps_times.extend([context_start] * len(new))
                        # One PSD per context (shared by all SAMPLES_PER_CONTEXT
                        # windows sliced from it), not one per sample.
                        if new:
                            psd_gps_starts.append(context_start)
                            psds.append(np.asarray(psd.value, dtype=np.float64))
                            if psd_freqs is None:
                                psd_freqs = np.asarray(psd.frequencies.value, dtype=np.float64)

                    except Exception as e:
                        failed += 1
                        print(f"\n  ! GPS {context_start:.0f}: {e}")

                    context_start += CONTEXT_STRIDE

                    elapsed = time.time() - t0
                    rate = len(samples) / max(elapsed, 1e-6)
                    eta  = (target - len(samples)) / max(rate, 1e-6)
                    m, s = divmod(int(eta), 60)
                    print(
                        f"\r  {len(samples):>6}/{target}  |  "
                        f"{rate:.1f} samples/s  |  ETA {m}m {s:02d}s  |  "
                        f"failed fetches: {failed}",
                        end='',
                    )

            backgrounds[run][ifo] = {
                'samples':        np.array(samples[:target],   dtype=np.float32),
                'gps_starts':     np.array(gps_times[:target], dtype=np.float64),
                # One PSD per context (dedup'd, ~1/SAMPLES_PER_CONTEXT of the
                # sample count) -- match a sample to its PSD via gps_starts.
                'psd_gps_starts': np.array(psd_gps_starts, dtype=np.float64),
                'psd_freqs':      psd_freqs if psd_freqs is not None else np.array([]),
                # float64, not float32: real strain PSD values in the sensitive
                # band are ~1e-46 to 1e-48, below float32's smallest representable
                # magnitude (~1.4e-45) -- float32 silently flushes them to zero.
                'psds':           np.array(psds, dtype=np.float64),
            }
            print(f"\n  Done: {len(backgrounds[run][ifo]['samples'])} samples  |  "
                  f"failed fetches: {failed}")

    with open(args.out, 'wb') as f:
        pickle.dump(backgrounds, f)

    print(f"\nSaved {args.out}")
    for run in args.runs:
        for ifo in IFOS:
            n = len(backgrounds[run][ifo]['samples'])
            size_gb = n * SAMPLE_LENGTH * 4 / 1e9
            print(f"  {run} {ifo}: {n} samples  [{size_gb:.2f} GB]")


if __name__ == '__main__':
    main()
