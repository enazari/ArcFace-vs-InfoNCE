#!/bin/bash
#SBATCH --account=your-account
#SBATCH --job-name=jepa-lora
#SBATCH --time=13:25:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:h100:2
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=ALL

CONFIG_NAME="jepa-lora"

# Load required modules (H100 needs StdEnv/2023 + CUDA 12.x + torch >= 2.5.1)
module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

# Navigate to project root
cd ..
echo "Project directory: $(pwd)"

# Parse num_identities from config to know which LMDBs to copy
N=$(grep "num_identities" configs/$CONFIG_NAME.yaml | awk '{print $2}')
echo "Config: $CONFIG_NAME, num_identities: $N"

# Copy LMDBs to SLURM_TMPDIR for fast local I/O
mkdir -p $SLURM_TMPDIR/data

echo "Copying training LMDB to local storage..."
cp -r data/ms1m_${N}_train.lmdb $SLURM_TMPDIR/data/
if [ $? -ne 0 ]; then echo "ERROR: Failed to copy train LMDB"; exit 1; fi

echo "Copying validation LMDB to local storage..."
cp -r data/ms1m_${N}_val.lmdb $SLURM_TMPDIR/data/
if [ $? -ne 0 ]; then echo "ERROR: Failed to copy val LMDB"; exit 1; fi

echo "Copying LFW LMDB to local storage..."
cp -r data/lfw_10fold_original_retinaface.lmdb $SLURM_TMPDIR/data/
if [ $? -ne 0 ]; then
    echo "WARNING: LFW LMDB not found, LFW evaluation will use network storage"
fi

echo "Copying CFP-FF LMDB to local storage..."
cp -r data/cfp_ff_10fold_retinaface.lmdb $SLURM_TMPDIR/data/
if [ $? -ne 0 ]; then
    echo "WARNING: CFP-FF LMDB not found, CFP-FF evaluation will be skipped"
fi

echo "Copying CFP-FP LMDB to local storage..."
cp -r data/cfp_fp_10fold_retinaface.lmdb $SLURM_TMPDIR/data/
if [ $? -ne 0 ]; then
    echo "WARNING: CFP-FP LMDB not found, CFP-FP evaluation will be skipped"
fi

echo "Local data size:"
du -sh $SLURM_TMPDIR/data/*

# Point training code at fast local storage via DATA_DIR env var.
# No symlink swap needed — scancel is safe, no cleanup required.
export DATA_DIR=$SLURM_TMPDIR/data

# Activate environment. Do NOT clear PYTHONPATH here — the opencv module adds cv2's
# path via PYTHONPATH, and clearing it would break 'import cv2' in eval_lfw.py.
source ../face-verification-env/bin/activate

# Prevent HuggingFace from attempting network calls on compute nodes
export HF_HUB_OFFLINE=1

# Run training
echo "Starting training with config: $CONFIG_NAME"
accelerate launch train.py --config $CONFIG_NAME

echo "Done."
