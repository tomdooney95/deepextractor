#!/bin/bash -l
# Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 5-00:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=40G
#SBATCH --output=logs/separation_o3_then_o4_%j.out
#SBATCH --job-name="DeepExtractor_O3_then_O4"

now=$(date)
echo "$now"

source /home/tdooney/miniconda3/etc/profile.d/conda.sh
conda activate deepextractor_v2

python -c "import torch; print('torch version:', torch.__version__)"
nvidia-smi --query-gpu=name --format=csv,noheader

cd /projects/0/prjs1498/deepextractor
mkdir -p logs

# Cross-training experiment: start from the O3-fine-tuned checkpoint, continue
# training on O4 data. Separate --out dir from o3_finetune/ -- --init-from only
# ever reads its source checkpoint, never writes to it, but pointing --out at
# o3_finetune/ itself would let this run's own checkpointing overwrite it, so
# a distinct output directory keeps the original O3 fine-tune fully intact
# regardless. Reuses O4's already-fitted scaler (matches this run's
# --shard-dir, so the provenance check added earlier passes) rather than
# fitting a new one -- the model is about to see O4 data, so O4's input
# statistics are what matter here.
OUT_DIR=/projects/0/prjs1498/checkpoints/o3_then_o4
CKPT=$OUT_DIR/checkpoint_best.pth.tar
SOURCE_CKPT=/projects/0/prjs1498/checkpoints/o3_finetune/checkpoint_best.pth.tar

if [ -f "$CKPT" ]; then
    echo "Found existing checkpoint at $CKPT -- resuming this run"
    INIT_OR_RESUME_ARGS=(--resume "$CKPT")
else
    echo "No checkpoint yet -- initializing from O3-fine-tuned checkpoint $SOURCE_CKPT"
    # Gentler LR than the original simulated->real fine-tune (1e-4): the O3/O4
    # domain gap is plausibly smaller than simulated/real, and the model has
    # already been through two convergence stages rather than one.
    INIT_OR_RESUME_ARGS=(--init-from "$SOURCE_CKPT" --lr 0.00005)
fi

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
