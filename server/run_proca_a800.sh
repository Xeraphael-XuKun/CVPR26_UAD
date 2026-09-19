#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wanghanzhi/CVPR26_UAD

CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 /mnt/cache/wanghanzhi/envs/llmpar/bin/python3 train.py \
  --config_file configs/ProCA.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_CHOICE imagenet \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/jx_vit_base_p16_224-80ecf9dd.pth \
  MODEL.METRIC_LOSS_TYPE wrt \
  MODEL.IF_LABELSMOOTH off \
  MODEL.STRIDE_SIZE "[16, 16]" \
  MODEL.PCA_LOSS_WEIGHT 0.01 \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets \
  DATASETS.MODALITIES "['RGB', 'IR', 'Thermal']" \
  DATALOADER.SAMPLER PKM \
  DATALOADER.NUM_INSTANCE 4 \
  SOLVER.SEED 1234 \
  SOLVER.OPTIMIZER_NAME SGD \
  SOLVER.MAX_EPOCHS 120 \
  SOLVER.BASE_LR 0.008 \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.WARMUP_EPOCHS 5 \
  SOLVER.CHECKPOINT_PERIOD 120 \
  SOLVER.EVAL_PERIOD 10 \
  SOLVER.LOSS_TYPE base+pca \
  INPUT.SIZE_TRAIN "[256, 128]" \
  INPUT.SIZE_TEST "[256, 128]" \
  TEST.IMS_PER_BATCH 1024 \
  TEST.RE_RANKING False \
  TEST.NECK_FEAT before \
  TEST.FEAT_NORM yes \
  TEST.TOP_K_EVAL 0 \
  OUTPUT_DIR /mnt/cache/wanghanzhi/CVPR26_UAD/outputs/whu_mars_1000/proca_seed1234
