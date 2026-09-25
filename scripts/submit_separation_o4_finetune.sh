#!/bin/bash -l
# Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 5-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=40G
#SBATCH --output=logs/separation_o4_finetune_%j.out
#SBATCH --job-name="DeepExtractor_O4_finetune"

now=$(date)
echo "$now"

source /home/tdooney/miniconda3/etc/profile.d/conda.sh
conda activate deepextractor_v2

python -c "import torch; print('torch version:', torch.__version__)"
nvidia-smi --query-gpu=name --format=csv,noheader

cd /projects/0/prjs1498/deepextractor
mkdir -p logs

# Self-resuming across the 5-day walltime cap -- see submit_separation_o3_finetune.sh
# for why this checks for THIS run's own checkpoint (--resume) separately
# from the pretrained starting point (--init-from, first submission only).
OUT_DIR=/projects/0/prjs1498/checkpoints/o4_finetune
CKPT=$OUT_DIR/checkpoint_best.pth.tar
PRETRAINED_CKPT=/projects/0/prjs1498/checkpoints/separation_v1/checkpoint_best.pth.tar

if [ -f "$CKPT" ]; then
    echo "Found existing fine-tuning checkpoint at $CKPT -- resuming this fine-tuning run"
    INIT_OR_RESUME_ARGS=(--resume "$CKPT")
else
    echo "No fine-tuning checkpoint yet -- initializing from pretrained simulated-data checkpoint $PRETRAINED_CKPT"
    INIT_OR_RESUME_ARGS=(--init-from "$PRETRAINED_CKPT" --lr 0.0001)
fi

# V1 permanently off (matches real O4 data availability -- H1/L1 only),
# presence-flag + masked-loss mechanism handles it; --workers matches
# --cpus-per-gpu above. No --amp: prior Snellius training on this codebase
# found AMP unstable with whitened targets (see train_fn_td's docstring) --
# train_separation.py defaults --amp to off for the same reason.
python scripts/train_separation.py \
    --shard-dir /projects/0/prjs1498/deepextractor/real_separation_data/O4 \
    --detectors H1 L1 V1 --active-detectors H1 L1 \
    --scaler /projects/0/prjs1498/deepextractor/scalers/o4_finetune_scaler.pkl \
    --features 64 128 256 512 1024 2048 \
    --dropout-p 0.1 --norm gn --num-groups 8 \
    --epochs 300 --batch-size 32 --workers 16 \
    --lr-patience 3 --patience 7 \
    --out "$OUT_DIR" \
    "${INIT_OR_RESUME_ARGS[@]}"

echo "DONE"
