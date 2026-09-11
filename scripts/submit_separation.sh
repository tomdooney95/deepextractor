#!/bin/bash -l
# Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 5-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=40G
#SBATCH --output=logs/separation_%j.out
#SBATCH --job-name="DeepExtractor_signal_glitch_separation"

now=$(date)
echo "$now"

source /home/tdooney/miniconda3/etc/profile.d/conda.sh
conda activate deepextractor_v2

python -c "import torch; print('torch version:', torch.__version__)"
nvidia-smi --query-gpu=name --format=csv,noheader

cd /projects/0/prjs1498/deepextractor
mkdir -p logs

# Self-resuming: the 48h walltime cap almost certainly won't cover the full
# training run, so this same script gets resubmitted repeatedly. Only pass
# --resume if a checkpoint from a prior submission actually exists -- on a
# fresh --resume path, train_separation.py would try to torch.load() a file
# that doesn't exist yet and crash immediately.
OUT_DIR=/projects/0/prjs1498/checkpoints/separation_v1
CKPT=$OUT_DIR/checkpoint_best.pth.tar
RESUME_ARGS=()
if [ -f "$CKPT" ]; then
    echo "Found existing checkpoint at $CKPT -- resuming"
    RESUME_ARGS=(--resume "$CKPT")
else
    echo "No existing checkpoint -- starting fresh"
fi

# V1 permanently off (matches the real O3/O4 data-availability plan), presence-flag
# + masked-loss mechanism handles it; --workers matches --cpus-per-gpu above.
python scripts/train_separation.py \
    --shard-dir /projects/0/prjs1498/data_separation_v3 \
    --detectors H1 L1 V1 --active-detectors H1 L1 \
    --scaler /projects/0/prjs1498/data_separation_v3/scaler.pkl \
    --features 64 128 256 512 1024 2048 \
    --dropout-p 0.1 --norm gn --num-groups 8 \
    --epochs 300 --batch-size 32 --workers 16 --amp \
    --out "$OUT_DIR" \
    "${RESUME_ARGS[@]}"

echo "DONE"
