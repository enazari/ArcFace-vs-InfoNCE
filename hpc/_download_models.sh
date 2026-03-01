#!/bin/bash
# Pre-download model weights for offline use on compute nodes.
# Run ONCE on a login node (has internet) before submitting training jobs.
#
# Usage:
#   cd hpc && bash download_models.sh

cd ..
source ../face-verification-env/bin/activate
mkdir -p data/checkpoints

echo "=== Downloading model weights to data/checkpoints/ ==="

# 1. CLIP ViT-B/32 (~340MB)
python -c "
import torch, os
dst = 'data/checkpoints/clip-vitb32-openai.bin'
if not os.path.exists(dst):
    print('Downloading CLIP ViT-B/32...')
    torch.hub.download_url_to_file(
        'https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/main/open_clip_pytorch_model.bin',
        dst)
    print(f'  saved: {dst} ({os.path.getsize(dst)/1e6:.0f} MB)')
else:
    print(f'CLIP already cached: {dst}')
"

# 2. DINOv2 ViT-B/14 (~330MB weights + repo clone for architecture)
python -c "
import torch, os
dst = 'data/checkpoints/dinov2-vitb14.pth'
if not os.path.exists(dst):
    print('Downloading DINOv2 ViT-B/14 weights...')
    torch.hub.download_url_to_file(
        'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        dst)
    print(f'  saved: {dst} ({os.path.getsize(dst)/1e6:.0f} MB)')
else:
    print(f'DINOv2 weights already cached: {dst}')

print('Caching DINOv2 repo (needed for model architecture)...')
torch.hub.set_dir('data/hub')
torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=False, verbose=False)
print('  saved: data/hub/facebookresearch_dinov2_main/')
"

# 3. I-JEPA ViT-H/14 (~10GB)
python -c "
import os
dst = 'data/checkpoints/ijepa-vith14-300e.pth.tar'
if not os.path.exists(dst):
    print('Downloading I-JEPA ViT-H/14 (~10GB, this will take a while)...')
    import torch
    torch.hub.download_url_to_file(
        'https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar',
        dst)
    print(f'  saved: {dst} ({os.path.getsize(dst)/1e9:.1f} GB)')
else:
    print(f'I-JEPA already cached: {dst}')
"

echo ""
echo "=== Cached files ==="
ls -lh data/checkpoints/
echo ""
echo "=== All models cached. Ready for offline compute nodes. ==="
