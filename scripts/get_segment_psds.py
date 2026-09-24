"""
Backfill per-context PSDs for already-generated real background datasets.

get_clean_backgrounds.py now saves a PSD alongside every context it fetches
(see that script), but the O3a/O3b/O4a/O4b backgrounds already generated on
CIT predate that change -- their PSDs were computed internally by .whiten()
and never saved. This script recovers them: for each *unique* gps_starts
value already present in an existing backgrounds_<RUN>.pkl (13 samples share
one context, so this is ~1/13th the number of samples, not one fetch per
sample), it refetches the same CONTEXT_DURATION-second window and computes
the PSD with the exact same fftlength/overlap/method/window .whiten() uses
internally -- not an approximation of the original PSD, a reproduction of it.

Resumable: writes progress incrementally and skips gps_starts it's already
computed a PSD for on a rerun, so an interrupted session (or a deliberate
Ctrl-C) doesn't lose completed work. Given the volume involved (tens of
thousands of unique contexts across all runs/IFOs), this is meant to run as
a background/batch job, not interactively in a login shell.

Example
-------
    python scripts/get_segment_psds.py --runs O3a O3b O4a O4b \\
        --backgrounds-dir . --out-dir segment_psds/
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from gwdatafind import find_urls
from gwpy.timeseries import TimeSeries

# ── Processing parameters (must match get_clean_backgrounds.py exactly) ───────

IFOS             = ['H1', 'L1']
FRAME_TYPE       = {'H1': 'H1_HOFT_C00', 'L1': 'L1_HOFT_C00'}
SAMPLE_RATE      = 4096   # Hz
CONTEXT_DURATION = 36     # s -- same window .whiten() originally estimated the PSD over
PSD_DURATION     = 4      # s -- Welch sub-segment length

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_checkpoint(out_path: Path):
    """Load already-computed (gps_start -> psd) pairs from a prior run, if any."""
    if not out_path.exists():
        return {}, None
    with open(out_path, 'rb') as f:
        d = pickle.load(f)
    done = {gps: psd for gps, psd in zip(d['psd_gps_starts'], d['psds'])}
    return done, d['psd_freqs']


def save_checkpoint(out_path: Path, done: dict, freqs: np.ndarray):
    gps_sorted = sorted(done)
    payload = {
        'psd_gps_starts': np.array(gps_sorted, dtype=np.float64),
        'psd_freqs':      freqs,
        'psds':           np.array([done[g] for g in gps_sorted], dtype=np.float32),
    }
    tmp = out_path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f)
    tmp.replace(out_path)  # atomic on POSIX -- never leaves a half-written file


def fetch_psd(ifo: str, context_start: float):
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
    return ts.psd(fftlength=PSD_DURATION, overlap=PSD_DURATION // 2, method='median', window='hann')

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', nargs='+', required=True, metavar='RUN',
                   help='Observing run(s) to backfill, e.g. O3a O3b O4a O4b')
    p.add_argument('--backgrounds-dir', type=Path, default=Path('.'),
                   help='Directory containing backgrounds_<RUN>.pkl (default: cwd)')
    p.add_argument('--out-dir', type=Path, default=Path('segment_psds'),
                   help='Output directory for <RUN>_<IFO>_psds.pkl (default: segment_psds/)')
    p.add_argument('--checkpoint-every', type=int, default=50,
                   help='Save progress every N newly-computed PSDs (default: 50)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for run in args.runs:
        bg_path = args.backgrounds_dir / f'backgrounds_{run}.pkl'
        with open(bg_path, 'rb') as f:
            backgrounds = pickle.load(f)

        for ifo in IFOS:
            unique_gps = sorted(set(np.asarray(backgrounds[run][ifo]['gps_starts']).tolist()))
            out_path = args.out_dir / f'{run}_{ifo}_psds.pkl'
            done, freqs = load_checkpoint(out_path)
            todo = [g for g in unique_gps if g not in done]

            print(f"\n{'─' * 60}")
            print(f"  {run} {ifo}: {len(unique_gps)} unique contexts, "
                  f"{len(done)} already done, {len(todo)} remaining")
            print(f"{'─' * 60}")

            t0 = time.time()
            failed = 0
            since_checkpoint = 0

            for i, gps in enumerate(todo):
                try:
                    psd = fetch_psd(ifo, gps)
                    done[gps] = np.asarray(psd.value, dtype=np.float32)
                    if freqs is None:
                        freqs = np.asarray(psd.frequencies.value, dtype=np.float64)
                    since_checkpoint += 1
                except Exception as e:
                    failed += 1
                    print(f"\n  ! GPS {gps:.0f}: {e}")

                if since_checkpoint >= args.checkpoint_every:
                    save_checkpoint(out_path, done, freqs)
                    since_checkpoint = 0

                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(todo) - i - 1) / max(rate, 1e-6)
                m, s = divmod(int(eta), 60)
                print(f"\r  {i + 1:>6}/{len(todo)}  |  {rate:.2f} contexts/s  |  "
                      f"ETA {m}m {s:02d}s  |  failed: {failed}", end='')

            if since_checkpoint > 0 or (todo and not out_path.exists()):
                save_checkpoint(out_path, done, freqs)

            print(f"\n  Done: {len(done)}/{len(unique_gps)} contexts saved to {out_path}  "
                  f"|  failed: {failed}")


if __name__ == '__main__':
    main()
