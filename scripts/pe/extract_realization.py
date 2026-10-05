"""Slice one event/realization out of a pe_cases_n*.pkl into its own pickle.

run.py always reads pickle[injection_label][0] -- it has no notion of
choosing a specific realisation. Rather than add that to Harsh's ported
code, this writes a tiny single-realisation pickle (same {event: [dict]}
structure, just one entry) that run.py's existing [0] indexing picks up
directly, so a specific (good- or bad-separation) realisation can be
targeted for a PE run without touching run.py.

Usage
-----
    python scripts/pe/extract_realization.py \\
        --pickle PE_results/simulated_separation_v1/pe_cases_n10.pkl \\
        --event GW150914 --realization 7 \\
        --out GW150914_r7.pkl
"""

import argparse
import pickle


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pickle", required=True, help="Source pe_cases_n*.pkl")
    p.add_argument("--event", required=True, help="Event name, e.g. GW150914")
    p.add_argument("--realization", type=int, required=True, help="Realisation index to extract")
    p.add_argument("--out", required=True, help="Output pickle path")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.pickle, "rb") as f:
        results = pickle.load(f)

    if args.event not in results:
        raise SystemExit(f"Event {args.event!r} not found. Available: {list(results.keys())}")

    realizations = results[args.event]
    if args.realization >= len(realizations):
        raise SystemExit(f"Only {len(realizations)} realisations for {args.event}, asked for index {args.realization}")

    sliced = {args.event: [realizations[args.realization]]}
    with open(args.out, "wb") as f:
        pickle.dump(sliced, f)
    print(f"Saved {args.out}: {args.event} realisation {args.realization} only")


if __name__ == "__main__":
    main()
