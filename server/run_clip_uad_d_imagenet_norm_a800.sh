#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wanghanzhi/XK/CVPR26_UAD

echo "RUN=D_clip_uad_imagenet_norm CONFIG=configs/clip_uad_imagenet_norm.yml LOSS_TYPE=base+pca+gpd NORM=imagenet BACKBONE_LR=0.0008 HEAD_LR=0.008"
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 /mnt/cache/wanghanzhi/envs/whu_mars/bin/python3 train.py \
  --config_file configs/clip_uad_imagenet_norm.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_CHOICE clip \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt \
  MODEL.TRANSFORMER_TYPE clip_vit_b16 \
  MODEL.METRIC_LOSS_TYPE wrt \
  MODEL.IF_LABELSMOOTH off \
  MODEL.STRIDE_SIZE "[16, 16]" \
  MODEL.PCA_LOSS_WEIGHT 0.01 \
  MODEL.GPD_MOMENTUM 0.2 \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets \
  DATASETS.MODALITIES "['RGB', 'IR', 'Thermal']" \
  DATALOADER.SAMPLER PKM \
  DATALOADER.NUM_INSTANCE 4 \
  SOLVER.SEED 1234 \
  SOLVER.OPTIMIZER_NAME SGD \
  SOLVER.MAX_EPOCHS 120 \
  SOLVER.BASE_LR 0.008 \
  SOLVER.BACKBONE_LR 0.0008 \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.WARMUP_EPOCHS 5 \
  SOLVER.CHECKPOINT_PERIOD 120 \
  SOLVER.EVAL_PERIOD 10 \
  SOLVER.LOSS_TYPE base+pca+gpd \
  INPUT.SIZE_TRAIN "[256, 128]" \
  INPUT.SIZE_TEST "[256, 128]" \
  INPUT.PIXEL_MEAN "[0.485, 0.456, 0.406]" \
  INPUT.PIXEL_STD "[0.229, 0.224, 0.225]" \
  TEST.IMS_PER_BATCH 1024 \
  TEST.RE_RANKING False \
  TEST.NECK_FEAT before \
  TEST.FEAT_NORM yes \
  TEST.TOP_K_EVAL 0 \
  OUTPUT_DIR /mnt/cache/wanghanzhi/XK/CVPR26_UAD/outputs/clip_vit_b16/D_clip_uad_imagenet_norm_seed1234
