"""Model for HW1: small sequential CNN (conv / BN / ReLU / pool / linear only).

Layer table (input 3 x S x S, S multiple of 16, 100 classes):

    Conv7x7 s2 3->32, BN, ReLU, MaxPool 3x3 s2 p1     -> S/4
    Conv5x5    32->64,  BN, ReLU                       -> S/4
    Conv3x3 s2 64->128, BN, ReLU                       -> S/8
    Conv1x1    128->256, BN, ReLU                      -> S/8
    Conv3x3 s2 256->256, BN, ReLU                      -> S/16
    Conv1x1    256->512, BN, ReLU                      -> S/16
    GlobalAvgPool, Linear 512->256, ReLU, Linear 256->100

Every conv: padding = k // 2, bias = False.  Every ReLU: inplace = True.
"""
import torch
import torch.nn as nn

NUM_CLASSES = 100

# (name, cin, cout, k, stride) -- the six convolutions, in order
CONV_SPECS = [
    ("conv1", 3, 32, 7, 2),
    ("conv2", 32, 64, 5, 1),
    ("conv3", 64, 128, 3, 2),
    ("conv4", 128, 256, 1, 1),
    ("conv5", 256, 256, 3, 2),
    ("conv6", 256, 512, 1, 1),
]


def conv_bn_relu(cin, cout, k, stride):
    return [
        nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    ]


class SmallCNN(nn.Sequential):
    """Plain nn.Sequential so that each kernel-launching op is a separate,
    individually addressable module (handy for per-layer profiling)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        layers = []
        for i, (name, cin, cout, k, s) in enumerate(CONV_SPECS):
            layers += conv_bn_relu(cin, cout, k, s)
            if i == 0:  # MaxPool only after the stem
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        ]
        super().__init__(*layers)


def layer_names(model: nn.Sequential):
    """Human-readable name per module, e.g. 'conv1', 'bn1', 'relu1', 'pool1', 'fc1'."""
    names, counters = [], {}
    for m in model:
        base = {
            nn.Conv2d: "conv", nn.BatchNorm2d: "bn", nn.ReLU: "relu",
            nn.MaxPool2d: "pool", nn.AdaptiveAvgPool2d: "gap",
            nn.Flatten: "flatten", nn.Linear: "fc",
        }[type(m)]
        counters[base] = counters.get(base, 0) + 1
        names.append(f"{base}{counters[base]}")
    return names


def build_model(device="cuda"):
    model = SmallCNN().to(device).eval()
    return model


if __name__ == "__main__":
    m = SmallCNN()
    print(m)
    print("params:", sum(p.numel() for p in m.parameters()))
    print("buffers:", sum(b.numel() for b in m.buffers()))
    x = torch.randn(2, 3, 64, 64)
    with torch.inference_mode():
        print(m(x).shape)
    print(layer_names(m))
