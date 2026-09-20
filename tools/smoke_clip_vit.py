import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.backbones.clip_vit import clip_vit_b16
from model.backbones.vit_pytorch import DropPath
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


def check_clip_backbone(path):
    model = clip_vit_b16(
        img_size=(256, 128),
        stride_size=(16, 16),
        drop_path_rate=0.1,
    )
    model.load_param(path)
    assert model.positional_embedding.shape == (129, 768)
    assert not hasattr(model, "proj"), "512D CLIP projection must not be part of the backbone"

    actual_rates = [
        block.drop_path.drop_prob if isinstance(block.drop_path, DropPath) else 0.0
        for block in model.transformer.resblocks
    ]
    expected_rates = torch.linspace(0, 0.1, 12).tolist()
    assert torch.allclose(torch.tensor(actual_rates), torch.tensor(expected_rates))

    torch.manual_seed(1234)
    dropped = model.transformer.resblocks[-1]._apply_drop_path(torch.ones(4, 128, 8))
    per_sample = dropped.permute(1, 0, 2).reshape(128, -1)
    assert torch.all(per_sample == per_sample[:, :1])
    assert (per_sample[:, 0] == 0).any(), "DropPath did not drop any sample in the smoke batch"

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 256, 128))
    assert output.shape == (1, 768)
    assert torch.isfinite(output).all()
    return tuple(output.shape), actual_rates


class OptimizerSmokeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2, bias=False)


def check_discriminative_lr():
    smoke_cfg = SimpleNamespace(SOLVER=SimpleNamespace(
        OPTIMIZER_NAME="SGD",
        BASE_LR=0.008,
        BACKBONE_LR=0.0008,
        BIAS_LR_FACTOR=2,
        WEIGHT_DECAY=1e-4,
        WEIGHT_DECAY_BIAS=1e-4,
        LARGE_FC_LR=False,
        MOMENTUM=0.9,
        CENTER_LR=0.5,
        MAX_EPOCHS=120,
        WARMUP_EPOCHS=5,
    ))

    model = OptimizerSmokeModel()
    center = nn.Linear(1, 1)
    optimizer, _ = make_optimizer(smoke_cfg, model, center)
    scheduler = create_scheduler(smoke_cfg, optimizer)

    initial_lrs = [group["initial_lr"] for group in optimizer.param_groups]
    assert initial_lrs[:2] == [0.0008, 0.0016]
    assert initial_lrs[2] == 0.008
    epoch_one_lrs = scheduler._get_lr(1)
    assert abs(epoch_one_lrs[2] / epoch_one_lrs[0] - 10.0) < 1e-8
    assert abs(epoch_one_lrs[1] / epoch_one_lrs[0] - 2.0) < 1e-8

    smoke_cfg.SOLVER.BACKBONE_LR = 0.008
    optimizer_all, _ = make_optimizer(smoke_cfg, model, center)
    scheduler_all = create_scheduler(smoke_cfg, optimizer_all)
    all_base_initial_lrs = [group["initial_lr"] for group in optimizer_all.param_groups]
    assert all_base_initial_lrs == [0.008, 0.016, 0.008]
    all_base_epoch_one_lrs = scheduler_all._get_lr(1)
    assert abs(all_base_epoch_one_lrs[1] / all_base_epoch_one_lrs[0] - 2.0) < 1e-8
    assert abs(all_base_epoch_one_lrs[2] / all_base_epoch_one_lrs[0] - 1.0) < 1e-8
    return initial_lrs, epoch_one_lrs, all_base_initial_lrs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", required=True)
    args = parser.parse_args()

    output_shape, drop_path_rates = check_clip_backbone(args.pretrain)
    initial_lrs, epoch_one_lrs, all_base_initial_lrs = check_discriminative_lr()
    print(
        "CLIP_VIT_SMOKE_OK output_shape={} drop_path_first_last=({:.6f}, {:.6f}) layered_initial_lrs={} "
        "layered_epoch1_lrs={} all_base_initial_lrs={}".format(
            output_shape,
            drop_path_rates[0],
            drop_path_rates[-1],
            sorted(set(initial_lrs)),
            sorted(set(epoch_one_lrs)),
            sorted(set(all_base_initial_lrs)),
        )
    )


if __name__ == "__main__":
    main()
