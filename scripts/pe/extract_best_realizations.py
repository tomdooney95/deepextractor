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
    print(f"{'event':>12}  {'best r':>6}  {'sig H1 %':>9}  {'sig L1 %':>9}  {'glitch %':>9}  {'mean %':>7}")
    for event in events:
        realizations = results[event]
        mean_mms = [
            np.mean([ex["mismatch_signal_h1"], ex["mismatch_signal_l1"], ex["mismatch_glitch"]])
            for ex in realizations
        ]
        i_best = int(np.argmin(mean_mms))
        ex = realizations[i_best]
        best[event] = [ex]
        print(f"{event:>12}  {i_best:>6}  {ex['mismatch_signal_h1']:>9.1f}  "
              f"{ex['mismatch_signal_l1']:>9.1f}  {ex['mismatch_glitch']:>9.1f}  {mean_mms[i_best]:>7.1f}")

    with open(args.out, "wb") as f:
        pickle.dump(best, f)
    print(f"\nSaved {args.out}: {len(best)} events, each realisation[0] = its own best draw")


if __name__ == "__main__":
    main()
