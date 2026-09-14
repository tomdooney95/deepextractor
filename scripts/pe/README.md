# Parameter estimation for de-glitched signals

Nested-sampling PE (bilby + dynesty) to validate DeepExtractor's glitch
subtraction accuracy, by running PE on the same event three ways:

- `control` — signal + noise only, no glitch (baseline).
- `dirty` — signal + noise + glitch left in.
- `prediction` — signal + noise after DeepExtractor's glitch removal.

Comparing posteriors across the three shows how much bias a glitch
introduces, and how much of that bias DeepExtractor's subtraction recovers.

Originally written by Harsh Narola (`parameter_estimation_glitch` repo);
duplicated here so PE code lives alongside the model producing what it's
validating. Two bugs from the original were fixed on import — see git log
for `scripts/pe/run.py`.

## Install

```bash
pip install -e ".[pe]"
```

(`bilby` itself is already a core `deepextractor` dependency; `[pe]` just
adds `dynesty`, the nested sampler used here.)

## Usage

Run from inside `scripts/pe/` — `run.py` loads `aLIGO_O4_high_asd.txt` and
imports `utils.py` via paths relative to its own location:

```bash
cd scripts/pe
python run.py --injection-file pe_cases_n1.pkl --injection-label GW150914 \
    --type control --outdir <dir> --label <name>
```

`--injection-label` selects which of the 13 bundled events (GW150914,
GW190412, GW190521, GW190828, GW191204, GW200129, GW200224, GW191109,
GW200225, GW231028, GW231226, GW240104, GW241110) to run. `--debug 1`
(default) runs a quick 1D chirp-mass likelihood sanity check before the
full nested-sampling run and saves a diagnostic plot.

`pe_cases_n1.pkl`'s `deglitched_*`/`g_hat_*` fields (the `prediction` case)
came from an earlier DeepExtractor iteration, not the current
`feature/signal-glitch-separation` model — regenerating them with the
current model's output is a follow-up, not done yet.

## Cluster submission (HTCondor)

`condor/job.sub` + `condor/job.dag` — a minimal DAGMan-wrapped single-job
example. Before using it:

- Edit `condor/job.sub`'s `Executable` line to point at your own conda
  env's python binary on whichever cluster you're submitting from.
- `condor/logs/` needs to exist before submission (Condor won't create it) —
  already present here, just don't delete it.
- The example only parameterizes `seed`/`label` — extend `job.dag`/`job.sub`
  if you want `--type`/`--injection-label` swept as part of a DAG.
