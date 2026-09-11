#!/bin/bash -l
# Short validation run BEFORE the real submit_separation.sh job: confirms the
# full-size model (173M params, 2048-wide, 16384-length sequences) actually
# fits in the H100's memory at --batch-size 32, and that the whole pipeline
# (env, data loading from the real transferred dataset, scaler fit/cache)
# launches cleanly. Already tested once directly on the real GPU (job
# 26571117: no OOM in ~7 min of training before hitting this script's own
# 30min cap, confirmed via nvidia-smi -l 5 polling) -- rerun mainly as a
# pipeline sanity check after switching back from the smaller 5-level
# config. Not self-resuming (deliberately, so reruns don't reuse a possibly-
# untested checkpoint).
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=40G
#SBATCH --output=logs/separation_test_%j.out
#SBATCH --job-name="DeepExtractor_signal_glitch_separation_TEST"

now=$(date)
echo "$now"

source /home/tdooney/miniconda3/etc/profile.d/conda.sh
conda activate deepextractor_v2

python -c "import torch; print('torch version:', torch.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /projects/0/prjs1498/deepextractor
mkdir -p logs

# Poll GPU memory in the background while training runs, since checking it
# in a separate process afterward would just see an empty, unrelated context.
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > logs/gpu_mem_${SLURM_JOB_ID}.log &
MEM_MONITOR_PID=$!

# No --amp -- matches submit_separation.sh (prior Snellius training found AMP
# unstable with whitened targets), and this test should validate the same
# config the real job will actually use.
python scripts/train_separation.py \
    --shard-dir /projects/0/prjs1498/data_separation_v3 \
    --detectors H1 L1 V1 --active-detectors H1 L1 \
    --scaler /projects/0/prjs1498/data_separation_v3/scaler.pkl \
    --features 64 128 256 512 1024 2048 \
    --dropout-p 0.1 --norm gn --num-groups 8 \
    --epochs 1 --batch-size 32 --workers 16 \
    --out /projects/0/prjs1498/checkpoints/separation_v1_test

kill $MEM_MONITOR_PID
echo "Peak GPU memory used (MiB):"
sort -n logs/gpu_mem_${SLURM_JOB_ID}.log | tail -1

echo "DONE"
