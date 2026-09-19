#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wanghanzhi/XK/CVPR26_UAD

/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3 -c "import sys, importlib.metadata as md, torch, torchvision, timm, PIL, yaml, numpy; actual={'python': '.'.join(map(str, sys.version_info[:2])), 'torch': torch.__version__.split('+')[0], 'torchvision': torchvision.__version__.split('+')[0], 'timm': timm.__version__, 'numpy': numpy.__version__, 'Pillow': PIL.__version__, 'PyYAML': yaml.__version__, 'tqdm': md.version('tqdm'), 'yacs': md.version('yacs')}; expected={'python':'3.10','torch':'2.2.2','torchvision':'0.17.2','timm':'1.0.27','numpy':'1.26.4','Pillow':'12.2.0','PyYAML':'6.0.3','tqdm':'4.67.3','yacs':'0.1.8'}; print('runtime=', actual); assert actual == expected, 'runtime version mismatch: expected {}'.format(expected); print('cuda=', torch.cuda.is_available(), 'gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); assert torch.cuda.is_available(); assert 'A800' in torch.cuda.get_device_name(0)"

test -d /mnt/cache/wanghanzhi/Datasets/WHU-MARS/train/RGB
test -d /mnt/cache/wanghanzhi/Datasets/WHU-MARS/train/IR
test -d /mnt/cache/wanghanzhi/Datasets/WHU-MARS/train/Thermal
test -f /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt

CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 /mnt/cache/wanghanzhi/envs/whu_mars/bin/python3 tools/smoke_uad.py \
  --dataset-root /mnt/cache/wanghanzhi/Datasets

CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 /mnt/cache/wanghanzhi/envs/whu_mars/bin/python3 tools/smoke_clip_vit.py \
  --pretrain /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
