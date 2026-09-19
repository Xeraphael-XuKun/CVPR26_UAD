import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.sampler import PKMSampler
from datasets.whu_mars import WHU_MARS
from processor.processor import GPD, PCA, PrototypeBank


def check_sampler():
    modalities = ["RGB", "IR", "Thermal"]
    data = {
        modality: [(f"{modality}_{pid}_{idx}.jpg", pid, idx, mid)
                   for pid in range(2) for idx in range(4)]
        for mid, modality in enumerate(modalities, 1)
    }
    sampler = PKMSampler(data, batch_size=8, num_instances=4, modalities=modalities)
    indices = list(iter(sampler))
    assert len(indices) == 8
    sampled_pids = []
    for index_tuple in indices:
        pids = [data[modality][index_tuple[mid]][1] for mid, modality in enumerate(modalities)]
        assert len(set(pids)) == 1
        sampled_pids.append(pids[0])
    assert all(len(set(sampled_pids[i:i + 4])) == 1 for i in range(0, 8, 4))


def check_losses():
    identities, modalities, instances, dim, classes = 2, 3, 4, 16, 5
    target = torch.arange(identities).repeat_interleave(instances)
    feat = torch.randn(modalities * identities * instances, dim, requires_grad=True)

    loss_proca, ids, centers = PCA(
        feat,
        target,
        num_modalities=modalities,
        group_size=instances,
    )
    assert ids.tolist() == [0, 1]
    assert centers.shape == (identities, dim)
    assert not centers.requires_grad

    bank = PrototypeBank(classes, dim, momentum=0.2, device="cpu")
    loss_gpd = GPD(feat, target, bank, ids, centers, num_modalities=modalities, tau=0.03)
    total = loss_proca * 0.01 + loss_gpd
    total.backward()
    assert torch.isfinite(total)
    assert feat.grad is not None and torch.isfinite(feat.grad).all()

    bank.update(ids, centers, ddp=False)
    updated_norms = bank.prototypes[ids].norm(dim=1)
    assert torch.allclose(updated_norms, torch.ones_like(updated_norms), atol=1e-5)


def check_dataset(root):
    dataset = WHU_MARS(root=root, modalities=["RGB", "IR", "Thermal"], verbose=False)
    actual = {
        "train": (dataset.num_train_pids, dataset.num_train_imgs, dataset.num_train_cams),
        "query": (dataset.num_query_pids, dataset.num_query_imgs, dataset.num_query_cams),
        "gallery": (dataset.num_gallery_pids, dataset.num_gallery_imgs, dataset.num_gallery_cams),
    }
    expected = {
        "train": (500, 92133, 7),
        "query": (500, 6405, 7),
        "gallery": (500, 93609, 7),
    }
    assert actual == expected, f"dataset statistics mismatch: {actual}"
    return actual


def check_pretrain(path):
    state = torch.load(path, map_location="cpu")
    assert isinstance(state, dict) and state, "pretrained checkpoint is not a non-empty mapping"
    return len(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--pretrain", default="")
    args = parser.parse_args()

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    check_sampler()
    check_losses()
    details = []
    if args.dataset_root:
        details.append(f"dataset={check_dataset(args.dataset_root)}")
    if args.pretrain:
        details.append(f"pretrain_tensors={check_pretrain(args.pretrain)}")
    suffix = " " + " ".join(details) if details else ""
    print("UAD_COMPONENT_SMOKE_OK" + suffix)


if __name__ == "__main__":
    main()
