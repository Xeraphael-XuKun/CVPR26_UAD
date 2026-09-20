#!/usr/bin/env bash
set -euo pipefail

cd /mnt/cache/wanghanzhi/XK/CVPR26_UAD

echo "RUN=V1_fair_vit_baseline CONFIG=configs/fair_vit_baseline.yml LOSS_TYPE=base NORM=0.5 BACKBONE_LR=0.008 DROP_PATH=0.1 SEED=1234"
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 /mnt/cache/wanghanzhi/envs/whu_mars/bin/python3 train.py \
  --config_file configs/fair_vit_baseline.yml \
  MODEL.DIST_TRAIN False \
  MODEL.PRETRAIN_CHOICE imagenet \
  MODEL.PRETRAIN_PATH /mnt/cache/wanghanzhi/Datasets/jx_vit_base_p16_224-80ecf9dd.pth \
  MODEL.TRANSFORMER_TYPE vit_base_in \
  MODEL.DROP_PATH 0.1 \
  DATASETS.ROOT_DIR /mnt/cache/wanghanzhi/Datasets \
  DATASETS.MODALITIES "['RGB', 'IR', 'Thermal']" \
  DATALOADER.SAMPLER PKM \
  DATALOADER.NUM_INSTANCE 4 \
  SOLVER.SEED 1234 \
  SOLVER.OPTIMIZER_NAME SGD \
  SOLVER.MAX_EPOCHS 120 \
  SOLVER.BASE_LR 0.008 \
  SOLVER.BACKBONE_LR 0.008 \
  SOLVER.BIAS_LR_FACTOR 2 \
  SOLVER.IMS_PER_BATCH 64 \
  SOLVER.WARMUP_EPOCHS 5 \
  SOLVER.CHECKPOINT_PERIOD 120 \
  SOLVER.EVAL_PERIOD 10 \
  SOLVER.LOSS_TYPE base \
  INPUT.PIXEL_MEAN "[0.5, 0.5, 0.5]" \
  INPUT.PIXEL_STD "[0.5, 0.5, 0.5]" \
  TEST.IMS_PER_BATCH 1024 \
  TEST.RE_RANKING False \
  TEST.NECK_FEAT before \
  TEST.FEAT_NORM yes \
  TEST.TOP_K_EVAL 0 \
  OUTPUT_DIR /mnt/cache/wanghanzhi/XK/CVPR26_UAD/outputs/fair_baseline_v2/vit/V1_baseline_seed1234
