import random
import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.make_model import build_transformer
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


sampler_path = Path(__file__).resolve().parents[1] / "datasets" / "sampler.py"
sampler_spec = importlib.util.spec_from_file_location("fair_sampler", sampler_path)
sampler_module = importlib.util.module_from_spec(sampler_spec)
sampler_spec.loader.exec_module(sampler_module)
PKMSampler = sampler_module.PKMSampler


class ShortBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8))

    def forward(self, x):
        return x


class LongBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2048))

    def forward(self, x):
        return x


def model_cfg(transformer_type):
    return SimpleNamespace(
        MODEL=SimpleNamespace(
            PRETRAIN_PATH="",
            PRETRAIN_CHOICE="none",
            COS_LAYER=False,
            NECK="bnneck",
            TRANSFORMER_TYPE=transformer_type,
            SIE_CAMERA=False,
            SIE_VIEW=False,
            SIE_COE=3.0,
            STRIDE_SIZE=[16, 16],
            DROP_PATH=0.1,
            DROP_OUT=0.0,
            ATT_DROP_RATE=0.0,
        ),
        INPUT=SimpleNamespace(SIZE_TRAIN=[256, 128]),
        TEST=SimpleNamespace(NECK_FEAT="before"),
        SOLVER=SimpleNamespace(SEED=1234),
    )


def check_shared_head_rng():
    outputs = []
    for name, backbone in (("short", ShortBackbone), ("long", LongBackbone)):
        torch.manual_seed(777)
        model = build_transformer(5, 0, 0, model_cfg(name), {name: backbone})
        outputs.append((model.classifier.weight.detach().clone(), torch.rand(4)))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], outputs[1][1])


def synthetic_data():
    modalities = ["RGB", "IR", "Thermal"]
    data = {
        modality: [
            ("{}_{}_{}.jpg".format(modality, pid, index), pid, index, modality_id)
            for pid in range(4)
            for index in range(8)
        ]
        for modality_id, modality in enumerate(modalities, 1)
    }
    return data, modalities


def check_sampler_rng():
    data, modalities = synthetic_data()
    first = list(PKMSampler(data, 8, 4, modalities, seed=1334))
    random.seed(9999)
    np.random.seed(9999)
    second = list(PKMSampler(data, 8, 4, modalities, seed=1334))
    third = list(PKMSampler(data, 8, 4, modalities, seed=1335))
    assert first == second
    assert first != third


class OptimizerSmokeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2, bias=False)


def optimizer_cfg(backbone_lr):
    return SimpleNamespace(SOLVER=SimpleNamespace(
        OPTIMIZER_NAME="SGD",
        BASE_LR=0.008,
        BACKBONE_LR=backbone_lr,
        BIAS_LR_FACTOR=2,
        WEIGHT_DECAY=1e-4,
        WEIGHT_DECAY_BIAS=1e-4,
        LARGE_FC_LR=False,
        MOMENTUM=0.9,
        CENTER_LR=0.5,
        MAX_EPOCHS=120,
        WARMUP_EPOCHS=5,
    ))


def check_groupwise_scheduler():
    center = nn.Linear(1, 1)
    for backbone_lr in (-1.0, 0.0008, 0.008):
        cfg = optimizer_cfg(backbone_lr)
        optimizer, _ = make_optimizer(cfg, OptimizerSmokeModel(), center)
        scheduler = create_scheduler(cfg, optimizer)
        initial_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        for epoch in (1, 60, 120):
            epoch_lrs = scheduler._get_lr(epoch)
            reference_scale = epoch_lrs[0] / initial_lrs[0]
            for initial_lr, epoch_lr in zip(initial_lrs, epoch_lrs):
                assert abs(epoch_lr / initial_lr - reference_scale) < 1e-10


def main():
    check_shared_head_rng()
    check_sampler_rng()
    check_groupwise_scheduler()
    print(
        "FAIR_BASELINE_SMOKE_OK shared_head=matched global_rng=matched "
        "sampler=isolated scheduler=per_group"
    )


if __name__ == "__main__":
    main()
