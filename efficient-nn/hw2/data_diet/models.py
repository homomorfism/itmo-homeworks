"""ResNet-V1 (low-resolution variant), a faithful PyTorch port of the Flax model in
https://github.com/mansheej/data_diet/blob/main/data_diet/models.py

Differences from the original Flax model, all deliberate and documented in the report:
  * 'SAME' padding with stride 2 is asymmetric in TF/Flax (pad 0 top / 1 bottom) and
    symmetric here (padding=1). Output shapes are identical, the sampling grid is
    shifted by one pixel.
  * Flax BatchNorm `momentum=0.9` == PyTorch `momentum=0.1` (same EMA, opposite convention).
"""
import math
from functools import partial

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------
#  Flax-compatible initialisation
# ---------------------------------------------------------------------------------------
# flax.linen.Conv / Dense default to lecun_normal() == variance_scaling(1.0, 'fan_in',
# 'truncated_normal'), which draws from a truncated normal on [-2, 2] rescaled so that the
# resulting variance is exactly 1 / fan_in.
_TRUNC_STD_CORRECTION = 0.8796256610342398  # std of a standard normal truncated to [-2, 2]


def lecun_normal_(tensor: torch.Tensor) -> torch.Tensor:
    fan_in = tensor[0].numel() if tensor.dim() > 1 else tensor.numel()
    std = math.sqrt(1.0 / fan_in) / _TRUNC_STD_CORRECTION
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    return tensor


def _init_module(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        lecun_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------------------
#  Blocks
# ---------------------------------------------------------------------------------------
class ResNetBlock(nn.Module):
    """Basic ResNet-V1 block."""

    expansion = 1

    def __init__(self, in_planes: int, filters: int, stride: int, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, filters, 3, stride, 1, bias=False)
        self.norm1 = norm(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1, bias=False)
        self.norm2 = norm(filters)
        self.act = nn.ReLU()  # not inplace: torch.func.vmap(grad(...)) in GraNd dislikes inplace ops
        out_planes = filters * self.expansion
        if stride != 1 or in_planes != out_planes:
            self.proj = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False), norm(out_planes)
            )
        else:
            self.proj = None

    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet-V1 block; the last norm is zero-initialised, as in Flax."""

    expansion = 4

    def __init__(self, in_planes: int, filters: int, stride: int, norm):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, filters, 1, 1, 0, bias=False)
        self.norm1 = norm(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride, 1, bias=False)
        self.norm2 = norm(filters)
        self.conv3 = nn.Conv2d(filters, filters * 4, 1, 1, 0, bias=False)
        self.norm3 = norm(filters * 4)
        nn.init.zeros_(self.norm3.weight)
        self.act = nn.ReLU()  # not inplace: torch.func.vmap(grad(...)) in GraNd dislikes inplace ops
        out_planes = filters * self.expansion
        if stride != 1 or in_planes != out_planes:
            self.proj = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, stride, 0, bias=False), norm(out_planes)
            )
        else:
            self.proj = None

    def forward(self, x):
        residual = x if self.proj is None else self.proj(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.act(self.norm2(self.conv2(y)))
        y = self.norm3(self.conv3(y))
        return self.act(residual + y)


# ---------------------------------------------------------------------------------------
#  Network
# ---------------------------------------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, stage_sizes, block_cls, num_classes, num_filters=64, lowres=True):
        super().__init__()
        norm = partial(nn.BatchNorm2d, eps=1e-5, momentum=0.1)
        self.lowres = lowres
        if lowres:
            self.conv_init = nn.Conv2d(3, num_filters, 3, 1, 1, bias=False)
            self.pool_init = nn.Identity()
        else:
            self.conv_init = nn.Conv2d(3, num_filters, 7, 2, 3, bias=False)
            self.pool_init = nn.MaxPool2d(3, 2, 1)
        self.bn_init = norm(num_filters)
        self.act = nn.ReLU()  # not inplace: torch.func.vmap(grad(...)) in GraNd dislikes inplace ops

        blocks, in_planes = [], num_filters
        for i, block_size in enumerate(stage_sizes):
            for j in range(block_size):
                stride = 2 if (i > 0 and j == 0) else 1
                filters = num_filters * 2**i
                blocks.append(block_cls(in_planes, filters, stride, norm))
                in_planes = filters * block_cls.expansion
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_planes, num_classes)

        self.apply(_init_module)
        for m in self.modules():  # re-apply the Flax zero-init of the bottleneck's last norm
            if isinstance(m, BottleneckResNetBlock):
                nn.init.zeros_(m.norm3.weight)

    def forward(self, x):
        x = self.pool_init(self.act(self.bn_init(self.conv_init(x))))
        x = self.blocks(x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)


def get_model(model: str, num_classes: int) -> nn.Module:
    if model == "resnet18_lowres":
        return ResNet18(num_classes=num_classes, lowres=True)
    if model == "resnet50_lowres":
        return ResNet50(num_classes=num_classes, lowres=True)
    raise NotImplementedError(model)


def get_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
