#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wanghanzhi/CVPR26_UAD

/mnt/cache/wanghanzhi/envs/llmpar/bin/python3 -c "import torch, torchvision, timm, yacs, PIL, yaml, numpy; print('torch=', torch.__version__, 'torchvision=', torchvision.__version__, 'timm=', timm.__version__); print('cuda=', torch.cuda.is_available(), 'gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); assert torch.cuda.is_available(); assert 'A800' in torch.cuda.get_device_name(0)"

test -d /mnt/cache/wanghanzhi/Datasets/WHU-MARS/train/RGB
test -d /mnt/cache/wanghanzhi/Datasets/WHU-MARS/train/IR
test -d /mnt/cache/wanghanzhi/Datasets/WHU-MARS/train/Thermal
test -f /mnt/cache/wanghanzhi/Datasets/jx_vit_base_p16_224-80ecf9dd.pth

CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 tools/smoke_uad.py \
  --dataset-root /mnt/cache/wanghanzhi/Datasets \
  --pretrain /mnt/cache/wanghanzhi/Datasets/jx_vit_base_p16_224-80ecf9dd.pth
