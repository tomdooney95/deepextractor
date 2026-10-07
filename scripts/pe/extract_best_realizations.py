"""Build a pe_cases pickle with each event's best (lowest mean mismatch)
realisation, for a should-work-well PE sweep across all events at once.

For each event, "best" means lowest mean of (mismatch_signal_h1,
mismatch_signal_l1, mismatch_glitch) -- same ranking plot_separation_check.py
--all uses for a single event. The output pickle has the same {event:
[dict]} structure as the source, but each event's list holds only its own
best realisation at index 0 -- so run.py's existing pickle[label][0]
indexing, and generate_dag.py's existing --injection-file sweep, both work
unmodified across the whole event set.

Usage
-----
    python scripts/pe/extract_best_realizations.py \\
        --pickle PE_results/simulated_separation_v1/pe_cases_n10.pkl \\
        --out pe_cases_best.pkl

    # Subset of events, e.g. matching generate_dag.py's in-prior-range list:
    python scripts/pe/extract_best_realizations.py \\
        --pickle PE_results/simulated_separation_v1/pe_cases_n10.pkl \\
        --events GW150914,GW190412,GW190521 \\
        --out pe_cases_best.pkl
"""

import argparse
import pickle

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pickle", required=True, help="Source pe_cases_n*.pkl")
    p.add_argument("--events", type=str, default=None,
                   help="Comma-separated event subset (default: every event in the pickle)")
    p.add_argument("--out", required=True, help="Output pickle path")
    p.add_argument("--quiet", action="store_true",
                   help="Print only the winning realisation per event, not the full per-realisation table")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.pickle, "rb") as f:
        results = pickle.load(f)

    events = args.events.split(",") if args.events else list(results.keys())
    unknown = set(events) - set(results.keys())
    if unknown:
        raise SystemExit(f"Event(s) not in pickle: {unknown}. Available: {list(results.keys())}")

    best = {}
    summary_rows = []

    for event in events:
        realizations = results[event]
        rows = []
        for i, ex in enumerate(realizations):
            glitch_det = "H1" if ex["inject_h1"] else "L1"
            mean_mm = np.mean([ex["mismatch_signal_h1"], ex["mismatch_signal_l1"], ex["mismatch_glitch"]])
            rows.append((i, ex["mismatch_signal_h1"], ex["mismatch_signal_l1"], ex["mismatch_glitch"], glitch_det, mean_mm))
        rows.sort(key=lambda r: r[-1])  # best (lowest mean mismatch) first

        i_best = rows[0][0]
        best[event] = [realizations[i_best]]
        summary_rows.append((event, *rows[0]))

        if not args.quiet:
            print(f"\n{event} -- {len(realizations)} realisations, sorted best (lowest mean mismatch) to worst:")
            print(f"{'r':>3}  {'sig H1 %':>9}  {'sig L1 %':>9}  {'glitch %':>9}  {'glitch in':>9}  {'mean %':>7}")
            for i, mm_h1, mm_l1, mm_g, det, mean_mm in rows:
                print(f"{i:>3}  {mm_h1:>9.1f}  {mm_l1:>9.1f}  {mm_g:>9.1f}  {det:>9}  {mean_mm:>7.1f}")

    print(f"\n{'='*60}\nBest realisation per event:")
    print(f"{'event':>12}  {'best r':>6}  {'sig H1 %':>9}  {'sig L1 %':>9}  {'glitch %':>9}  {'glitch in':>9}  {'mean %':>7}")
    for event, i_best, mm_h1, mm_l1, mm_g, det, mean_mm in summary_rows:
        print(f"{event:>12}  {i_best:>6}  {mm_h1:>9.1f}  {mm_l1:>9.1f}  {mm_g:>9.1f}  {det:>9}  {mean_mm:>7.1f}")

    with open(args.out, "wb") as f:
        pickle.dump(best, f)
    print(f"\nSaved {args.out}: {len(best)} events, each realisation[0] = its own best draw")


if __name__ == "__main__":
    main()
