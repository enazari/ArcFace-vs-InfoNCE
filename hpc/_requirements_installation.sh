#!/bin/bash
#SBATCH --account=your-account
#SBATCH --job-name=fv_install
#SBATCH --time=00:45:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=user@example.com
#SBATCH --mail-type=ALL

echo "=== FACE-VERIFICATION ENVIRONMENT INSTALLATION ==="

# Step 1: Load required modules
# H100 GPUs require torch >= 2.5.1, which needs StdEnv/2023 + CUDA 12.x
echo "Loading required modules..."
module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

echo "Loaded modules:"
module list

# Step 2: Navigate to project parent (env lives alongside project dir)
# Layout: .../face-verification/face-verification/hpc/ (we are here)
#         .../face-verification/face-verification-env/  (env goes here)
cd ../..
PERSISTENT_DIR=$(pwd)/face-verification-env
echo "Current directory: $(pwd)"
echo "Persistent venv will be: $PERSISTENT_DIR"

# Remove existing environment if it exists
if [ -d "$PERSISTENT_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$PERSISTENT_DIR"
fi

# Step 3: Clear PYTHONPATH
echo "Clearing PYTHONPATH to avoid conflicts..."
unset PYTHONPATH
export PYTHONPATH=""

# Step 4: Build venv on SLURM_TMPDIR (fast local NVMe) to avoid Lustre I/O errors
# Large pip installs (torch ~2GB) cause [Errno 14] Bad address on Lustre.
BUILD_ENV=$SLURM_TMPDIR/face-verification-env
echo "Creating virtual environment in SLURM_TMPDIR..."
python -m venv --system-site-packages $BUILD_ENV

if [ ! -d "$BUILD_ENV" ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

source $BUILD_ENV/bin/activate
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: VIRTUAL_ENV not set - activation failed"
    exit 1
fi
echo "Build venv activated: $VIRTUAL_ENV"

# Step 5: Upgrade pip
echo "Upgrading pip..."
pip install --no-index --upgrade pip
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upgrade pip"
    exit 1
fi

# Step 6: Install packages (all on fast local storage)

echo "Installing numpy (<2 for opencv module compatibility)..."
pip install --no-index 'numpy<2'
if [ $? -ne 0 ]; then echo "ERROR: Failed to install numpy"; exit 1; fi

echo "Installing PyTorch ecosystem..."
pip install --no-index torch torchvision
if [ $? -ne 0 ]; then echo "ERROR: Failed to install torch/torchvision"; exit 1; fi

echo "Installing scikit-learn..."
pip install --no-index scikit-learn
if [ $? -ne 0 ]; then echo "ERROR: Failed to install scikit-learn"; exit 1; fi

echo "Installing HPC-cached packages..."
pip install --no-index lmdb pyyaml pillow accelerate tqdm scikit-image
if [ $? -ne 0 ]; then
    echo "WARNING: Some HPC-cached packages failed, continuing..."
fi
# opencv-python is NOT installed via pip on ComputeCanada.
# It is provided by the opencv module (loaded above). --system-site-packages makes it visible.

# Packages that may need internet (not typically in HPC wheel cache).
echo "Installing open-clip-torch (for CLIP backbone)..."
pip install open-clip-torch
if [ $? -ne 0 ]; then
    echo "WARNING: open-clip-torch failed. CLIP backbone will not be available."
fi

echo "Installing retina-face and tf-keras (may require internet)..."
pip install retina-face tf-keras
if [ $? -ne 0 ]; then
    echo "WARNING: retina-face/tf-keras failed. These need internet or pre-downloaded wheels."
    echo "They are only needed for LFW evaluation, training will still work without them."
fi

# Step 7: Verify packages before copying
echo "Verifying package compatibility..."
python -c "
import numpy as np
print(f'NumPy {np.__version__}')

import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')

import torchvision
print(f'Torchvision {torchvision.__version__}')

import lmdb
print(f'lmdb {lmdb.version()}')

import tqdm
print(f'tqdm {tqdm.__version__}')

# cv2 is NOT tested here. It is provided by 'module load opencv/4.8.1' and made
# visible at runtime via PYTHONPATH (set by the module). We cleared PYTHONPATH
# earlier to avoid pip conflicts, so cv2 is intentionally not importable here.
# Training scripts (nce.sh) load the module before activating the venv.

from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
test_array = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
result = ToTensor()(Image.fromarray(test_array))
print(f'ToTensor transform: {result.shape}')

print('All critical packages working.')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed"
    exit 1
fi

# Step 8: Copy venv from SLURM_TMPDIR to persistent Lustre storage
deactivate
echo "Copying venv to persistent storage: $PERSISTENT_DIR ..."
cp -a $BUILD_ENV $PERSISTENT_DIR

# Fix hardcoded paths (SLURM_TMPDIR path -> persistent path)
echo "Fixing venv paths..."
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate.csh
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate.fish
# Fix shebangs in pip and other scripts
find $PERSISTENT_DIR/bin -type f -exec grep -l "$BUILD_ENV" {} + 2>/dev/null | \
    xargs -r sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g"

# Verify the persistent copy works
source $PERSISTENT_DIR/bin/activate
echo "Persistent venv activated: $VIRTUAL_ENV"
python -c "import torch; print(f'torch {torch.__version__} OK')"
if [ $? -ne 0 ]; then
    echo "ERROR: Persistent venv verification failed"
    exit 1
fi

# Step 9: Summary
echo "=== INSTALLATION SUMMARY ==="
echo "Virtual environment: $VIRTUAL_ENV"
echo "Python: $(which python)"
echo ""
echo "Installed packages:"
pip list
echo ""
echo "=== INSTALLATION COMPLETED ==="
echo ""
echo "To use in training scripts:"
echo "  module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1"
echo "  unset PYTHONPATH && export PYTHONPATH=\"\""
echo "  source ../face-verification-env/bin/activate"
