#!/bin/bash -l
# Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 1-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=40G
#SBATCH --output=logs/separation_o4_then_o3_%j.out
#SBATCH --job-name="DeepExtractor_O4_then_O3"

now=$(date)
echo "$now"

source /home/tdooney/miniconda3/etc/profile.d/conda.sh
conda activate deepextractor_v2

python -c "import torch; print('torch version:', torch.__version__)"
nvidia-smi --query-gpu=name --format=csv,noheader

cd /projects/0/prjs1498/deepextractor
mkdir -p logs

# Cross-training experiment: start from the O4-fine-tuned checkpoint, continue
# training on O3 data. See submit_separation_o3_then_o4.sh for why this uses
# a separate --out dir (keeps o4_finetune/checkpoint_best.pth.tar untouched)
# and reuses O3's already-fitted scaler rather than fitting a new one.
OUT_DIR=/projects/0/prjs1498/checkpoints/o4_then_o3
CKPT=$OUT_DIR/checkpoint_best.pth.tar
SOURCE_CKPT=/projects/0/prjs1498/checkpoints/o4_finetune/checkpoint_best.pth.tar

if [ -f "$CKPT" ]; then
    echo "Found existing checkpoint at $CKPT -- resuming this run"
    INIT_OR_RESUME_ARGS=(--resume "$CKPT")
else
    echo "No checkpoint yet -- initializing from O4-fine-tuned checkpoint $SOURCE_CKPT"
    INIT_OR_RESUME_ARGS=(--init-from "$SOURCE_CKPT" --lr 0.00005)
fi

python scripts/train_separation.py \
    --shard-dir /projects/0/prjs1498/deepextractor/real_separation_data/O3 \
    --detectors H1 L1 V1 --active-detectors H1 L1 \
    --scaler /projects/0/prjs1498/deepextractor/scalers/o3_finetune_scaler.pkl \
    --features 64 128 256 512 1024 2048 \
    --dropout-p 0.1 --norm gn --num-groups 8 \
    --epochs 300 --batch-size 32 --workers 16 \
    --lr-patience 3 --patience 7 \
    --out "$OUT_DIR" \
    "${INIT_OR_RESUME_ARGS[@]}"

echo "DONE"
