import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit_pytorch import DropPath


class LayerNorm(nn.LayerNorm):
    """LayerNorm that keeps OpenAI CLIP's fp16-safe behavior."""

    def forward(self, x):
        dtype = x.dtype
        return super().forward(x.float()).to(dtype)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads, drop_path=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))
        self.ln_2 = LayerNorm(width)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _apply_drop_path(self, x):
        # CLIP uses [sequence, batch, channel], while the shared DropPath
        # implementation expects batch-first tensors for per-sample masks.
        return self.drop_path(x.permute(1, 0, 2)).permute(1, 0, 2)

    def forward(self, x):
        normalized = self.ln_1(x)
        attention = self.attn(normalized, normalized, normalized, need_weights=False)[0]
        x = x + self._apply_drop_path(attention)
        return x + self._apply_drop_path(self.mlp(self.ln_2(x)))


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path_rate=0.0):
        super().__init__()
        drop_path_rates = torch.linspace(0, drop_path_rate, layers).tolist()
        print("CLIP DropPath rates: first={:.6f} last={:.6f} layers={}".format(
            drop_path_rates[0], drop_path_rates[-1], layers
        ))
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, drop_path_rates[layer])
            for layer in range(layers)
        ])

    def forward(self, x):
        return self.resblocks(x)


class CLIPVisionTransformer(nn.Module):
    """CLIP ViT-B/16 visual encoder ending at the 768D pre-projection CLS feature."""

    def __init__(self, img_size=(256, 128), patch_size=16, width=768, layers=12, heads=12,
                 drop_path_rate=0.0, drop_rate=0.0, attn_drop_rate=0.0,
                 camera=0, view=0, **kwargs):
        super().__init__()
        if tuple(kwargs.get("stride_size", (patch_size, patch_size))) != (patch_size, patch_size):
            raise ValueError("CLIP ViT-B/16 requires STRIDE_SIZE [16, 16]")
        if drop_rate != 0.0 or attn_drop_rate != 0.0:
            raise ValueError("CLIP ViT-B/16 currently requires DROP_OUT=0 and ATT_DROP_RATE=0")
        if camera > 0 or view > 0:
            raise ValueError("CLIP ViT-B/16 does not implement SIE camera/view embeddings")

        self.img_size = tuple(img_size)
        self.grid_size = (self.img_size[0] // patch_size, self.img_size[1] // patch_size)
        self.num_features = self.embed_dim = width

        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)
        self.ln_post = LayerNorm(width)

    def forward(self, x, cam_label=None, modal_label=None, view_label=None):
        if tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(
                "Input image size {} does not match CLIP backbone size {}".format(
                    tuple(x.shape[-2:]), self.img_size
                )
            )

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return self.ln_post(x[:, 0, :])

    def load_param(self, model_path):
        try:
            checkpoint = torch.jit.load(model_path, map_location="cpu").state_dict()
        except RuntimeError:
            checkpoint = torch.load(model_path, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

        visual = {
            key[len("visual."):]: value.float()
            for key, value in checkpoint.items()
            if key.startswith("visual.") and key != "visual.proj"
        }
        if not visual:
            raise ValueError("No visual.* CLIP weights found in {}".format(model_path))

        source_pos = visual["positional_embedding"]
        if source_pos.shape != self.positional_embedding.shape:
            visual["positional_embedding"] = resize_clip_positional_embedding(
                source_pos,
                self.grid_size,
            )

        missing, unexpected = self.load_state_dict(visual, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "CLIP visual weight mismatch: missing={} unexpected={}".format(missing, unexpected)
            )
        print(
            "Loaded CLIP ViT-B/16 visual weights; positional grid 14x14 -> {}x{} "
            "with bilinear interpolation; output is 768D pre-projection CLS.".format(*self.grid_size)
        )


def resize_clip_positional_embedding(positional_embedding, target_grid):
    cls_pos = positional_embedding[:1]
    grid_pos = positional_embedding[1:]
    source_size = int(math.sqrt(grid_pos.shape[0]))
    if source_size * source_size != grid_pos.shape[0]:
        raise ValueError("CLIP positional embedding grid is not square")

    grid_pos = grid_pos.reshape(1, source_size, source_size, -1).permute(0, 3, 1, 2)
    grid_pos = F.interpolate(
        grid_pos.float(),
        size=target_grid,
        mode="bilinear",
        align_corners=False,
    )
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(-1, positional_embedding.shape[-1])
    return torch.cat([cls_pos.float(), grid_pos], dim=0)


def clip_vit_b16(img_size=(256, 128), stride_size=(16, 16), drop_path_rate=0.1,
                 drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, **kwargs):
    return CLIPVisionTransformer(
        img_size=img_size,
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        stride_size=stride_size,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        camera=camera,
        view=view,
    )
