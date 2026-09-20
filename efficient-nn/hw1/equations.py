"""Analytical cost model of SmallCNN (see models.py) as closed-form functions of
image size S (H = W = S) and batch size B.

All four public functions accept scalars or NumPy arrays (broadcasting):

    flops(S, B)                    -> FLOPs of one forward pass        (1 MAC = 2 FLOPs)
    memory(S, B)                   -> peak torch.cuda.max_memory_allocated(), bytes
    latency(S, B, theta)           -> seconds     (theta calibrated on the GPU)
    energy(S, B, theta_energy)     -> joules      (theta_energy calibrated on the GPU)

Everything is derived from one per-layer table (LAYERS) so that FLOPs, bytes
moved and live activations are consistent with each other.

Notation: activation sizes are written in units of B*S^2 elements, e.g. the
output of conv1 is 32 channels at (S/2)^2 = 32/4 = 8 B*S^2 elements.
"""
from __future__ import annotations

import numpy as np

FP32 = 4  # bytes per element

# ----------------------------------------------------------------------------
# Per-layer table.
#   act_in, act_out : activation elements read / written, in units of B*S^2
#                     (for the head they are in units of B, see `head`)
#   flops           : FLOPs per B*S^2 (or per B for the head)
#   weights         : parameter + buffer elements read by the layer (constant)
#   alloc           : does the op allocate a new output tensor?  (in-place ReLU
#                     does not; Flatten is a view)
#
#  activation sizes:  x = 3,  conv1 out = 32/4 = 8,  pool1 out = 32/16 = 2,
#                     conv2 out = 64/16 = 4,  conv3 out = 128/64 = 2,
#                     conv4 out = 256/64 = 4, conv5 out = 256/256 = 1,
#                     conv6 out = 512/256 = 2         (all in units of B*S^2)
#  conv MACs / (B*S^2) = Cout * Cin * k^2 / stride_total^2   (stride_total = cumulative)
#     conv1: 32*3*49 / 4     = 1176     conv2: 64*32*25 / 16   = 3200
#     conv3: 128*64*9 / 64   = 1152     conv4: 256*128 / 64    = 512
#     conv5: 256*256*9 / 256 = 2304     conv6: 512*256 / 256   = 512
#  BN (eval)  : y = x*scale + shift -> 2 FLOPs / element
#  ReLU       : 1 FLOP / element (max with 0), in place: read + write
#  MaxPool3x3 : 8 comparisons / output element
# ----------------------------------------------------------------------------
LAYERS = [
    # name     act_in act_out  flops     weights                 alloc  head
    ("conv1",   3,     8,      2 * 1176, 3 * 32 * 49,            True,  False),
    ("bn1",     8,     8,      2 * 8,    4 * 32,                 True,  False),
    ("relu1",   8,     8,      1 * 8,    0,                      False, False),
    ("pool1",   8,     2,      8 * 2,    0,                      True,  False),
    ("conv2",   2,     4,      2 * 3200, 32 * 64 * 25,           True,  False),
    ("bn2",     4,     4,      2 * 4,    4 * 64,                 True,  False),
    ("relu2",   4,     4,      1 * 4,    0,                      False, False),
    ("conv3",   4,     2,      2 * 1152, 64 * 128 * 9,           True,  False),
    ("bn3",     2,     2,      2 * 2,    4 * 128,                True,  False),
    ("relu3",   2,     2,      1 * 2,    0,                      False, False),
    ("conv4",   2,     4,      2 * 512,  128 * 256,              True,  False),
    ("bn4",     4,     4,      2 * 4,    4 * 256,                True,  False),
    ("relu4",   4,     4,      1 * 4,    0,                      False, False),
    ("conv5",   4,     1,      2 * 2304, 256 * 256 * 9,          True,  False),
    ("bn5",     1,     1,      2 * 1,    4 * 256,                True,  False),
    ("relu5",   1,     1,      1 * 1,    0,                      False, False),
    ("conv6",   1,     2,      2 * 512,  256 * 512,              True,  False),
    ("bn6",     2,     2,      2 * 2,    4 * 512,                True,  False),
    ("relu6",   2,     2,      1 * 2,    0,                      False, False),
    # global average pool: reads 2 B*S^2, writes 512 B, ~1 FLOP per input element
    ("gap1",    2,     0,      1 * 2,    0,                      True,  False),
    # ---- head: sizes in units of B, flops per B ----
    ("fc1",     512,   256,    2 * 512 * 256, 512 * 256 + 256,   True,  True),
    ("relu7",   256,   256,    256,      0,                      False, True),
    ("fc2",     256,   100,    2 * 256 * 100, 256 * 100 + 100,   True,  True),
]
LAYER_NAMES = [l[0] for l in LAYERS]

# gap1 writes B*512 elements; fold it into the head-sized terms
GAP_OUT_PER_B = 512

PARAM_ELEMS = 1_042_820         # sum(p.numel() for p in model.parameters())
BUFFER_ELEMS = 2_502            # BN running_mean/var (+ 6 x num_batches_tracked, int64)
WEIGHT_BYTES = FP32 * (PARAM_ELEMS + 2_496) + 8 * 6   # ~4.18 MB, resident all the time

# PyTorch allocates its cuBLAS workspaces through the caching allocator on the
# first GEMM (the Linear layers) and keeps them for the life of the process, so
# they are part of max_memory_allocated():
#   cuBLAS   : CUBLAS_WORKSPACE_CONFIG default ":4096:2:16:8" = 4096*2 + 16*8 KiB = 8320 KiB
#   cuBLASLt : CUBLASLT_WORKSPACE_SIZE default 1024 KiB
CUBLAS_WORKSPACE_BYTES = (8320 + 1024) * 1024          # = 9.125 MiB
CONST_BYTES = WEIGHT_BYTES + CUBLAS_WORKSPACE_BYTES    # ~ 13.1 MiB

# Peak live activation set, in units of B*S^2 elements, for the sequential
# forward under inference_mode.  The input x (3 B S^2) is held by the caller
# for the whole forward.  At bn1 the input conv1-output (8) and the freshly
# allocated bn1-output (8) are both alive:   3 + 8 + 8 = 19.
# (conv1: 3+8 = 11,  pool1: 3+8+2 = 13,  conv2: 3+2+4 = 9, ... all smaller.)
PEAK_LIVE_ELEMS = 3 + 8 + 8   # = 19  ->  76 B S^2 bytes


def _S2(image_size):
    S = np.asarray(image_size, dtype=np.float64)
    return S * S


def _B(batch):
    return np.asarray(batch, dtype=np.float64)


# ----------------------------------------------------------------------------
# Per-layer primitives (used by latency/energy; each returns an array
# broadcast over (S, B)).
# ----------------------------------------------------------------------------
def layer_flops(image_size, batch):
    """List of per-layer FLOPs arrays, in LAYERS order."""
    S2, B = _S2(image_size), _B(batch)
    out = []
    for name, a_in, a_out, fl, w, alloc, head in LAYERS:
        out.append(B * fl if head else B * S2 * fl)
    # gap1: + one division per output element
    out[LAYER_NAMES.index("gap1")] = out[LAYER_NAMES.index("gap1")] + B * GAP_OUT_PER_B
    return out


def layer_bytes(image_size, batch):
    """List of per-layer bytes moved (activations read+written + weights read)."""
    S2, B = _S2(image_size), _B(batch)
    out = []
    for name, a_in, a_out, fl, w, alloc, head in LAYERS:
        scale = B if head else B * S2
        out.append(FP32 * (scale * (a_in + a_out) + w))
    out[LAYER_NAMES.index("gap1")] = out[LAYER_NAMES.index("gap1")] + FP32 * B * GAP_OUT_PER_B
    return out


# ----------------------------------------------------------------------------
# 1. FLOPs
# ----------------------------------------------------------------------------
FLOPS_PER_BS2 = sum(fl for (_, _, _, fl, _, _, head) in LAYERS if not head)   # = 17795
FLOPS_PER_B = sum(fl for (_, _, _, fl, _, _, head) in LAYERS if head) + GAP_OUT_PER_B  # = 314112
CONV_FLOPS_PER_BS2 = sum(fl for (n, _, _, fl, _, _, _) in LAYERS if n.startswith("conv"))  # = 17712


def flops(image_size, batch):
    """FLOPs of one forward pass.

        FLOPs(S, B) = 17793 * B * S^2 + 314112 * B

    of which the six convolutions contribute 17712 * B * S^2 (99.5 %).
    """
    S2, B = _S2(image_size), _B(batch)
    return FLOPS_PER_BS2 * B * S2 + FLOPS_PER_B * B


def bytes_moved(image_size, batch):
    """Total DRAM traffic of one forward pass, bytes (ideal: every activation is
    read once by its consumer and written once by its producer, weights read once).

        Bytes(S, B) = 4 * (133 * B * S^2 + 1636 * B + 1045316)
    """
    return sum(layer_bytes(image_size, batch))


# ----------------------------------------------------------------------------
# 2. Memory
# ----------------------------------------------------------------------------
def memory(image_size, batch):
    """Peak torch.cuda.max_memory_allocated() of one forward pass, bytes.

        Memory(S, B) = 76 * B * S^2 + 13.75e6      (bytes)

    76 B S^2 = 4 bytes * (x: 3 + conv1 out: 8 + bn1 out: 8) B S^2, the largest
    simultaneously-live set of activations (reached inside bn1);  the constant
    is parameters + BN buffers (4.18 MB) + cuBLAS/cuBLASLt workspaces (9.57 MB),
    all resident for the whole forward pass.
    Not modelled: cuDNN convolution workspace (heuristic-dependent, matters only
    when B*S^2 is small, see README) and the 512-byte allocator rounding.
    """
    S2, B = _S2(image_size), _B(batch)
    return FP32 * PEAK_LIVE_ELEMS * B * S2 + CONST_BYTES


# ----------------------------------------------------------------------------
# 3. Latency  --  per-layer roofline with a launch floor
# ----------------------------------------------------------------------------
THETA_KEYS = ("t_launch", "bandwidth", "peak_flops")


def latency(image_size, batch, theta):
    """Predicted wall-clock latency of one forward pass, seconds.

        T(S, B) = sum_over_layers  max( t_launch,
                                        bytes_l(S,B) / bandwidth,
                                        flops_l(S,B) / peak_flops )

    theta = {"t_launch": s per kernel-launching op,
             "bandwidth": effective bytes/s,
             "peak_flops": effective FLOP/s}
    Three regimes: small (S, B) -> every layer sits on the t_launch floor
    (launch-bound); elementwise layers (BN, ReLU, pool) are always memory-bound;
    the convolutions become compute-bound as B*S^2 grows.
    """
    t0, bw, pf = theta["t_launch"], theta["bandwidth"], theta["peak_flops"]
    fl, by = layer_flops(image_size, batch), layer_bytes(image_size, batch)
    total = 0.0
    for f, b in zip(fl, by):
        total = total + np.maximum(t0, np.maximum(b / bw, f / pf))
    return total


def latency_additive(image_size, batch, theta):
    """Baseline for comparison: no overlap between the three costs.

        T(S, B) = N_layers * t_launch + Bytes(S,B)/bandwidth + FLOPs(S,B)/peak_flops
    """
    t0, bw, pf = theta["t_launch"], theta["bandwidth"], theta["peak_flops"]
    return len(LAYERS) * t0 + bytes_moved(image_size, batch) / bw + flops(image_size, batch) / pf


# ----------------------------------------------------------------------------
# 4. Energy
# ----------------------------------------------------------------------------
THETA_ENERGY_KEYS = ("p_static", "e_flop", "e_byte")


def energy(image_size, batch, theta_energy):
    """Predicted energy of one forward pass, joules, whole GPU.

        E(S, B) = p_static * T(S, B, theta) + e_flop * FLOPs(S, B) + e_byte * Bytes(S, B)

    p_static [W]  : power that is drawn as long as the GPU is 'busy' with the
                    pass, whatever the kernels do (idle/static + clocks);
    e_flop [J]    : dynamic energy per FLOP;
    e_byte [J]    : dynamic energy per byte moved through DRAM.
    theta_energy must also contain the latency theta under key "latency".
    NB: for a fixed network FLOPs/Bytes is constant (33.4 FLOP/byte here), so
    e_flop and e_byte are collinear; calibrate.py fits e_flop with e_byte = 0.
    """
    T = latency(image_size, batch, theta_energy["latency"])
    return (theta_energy["p_static"] * T
            + theta_energy["e_flop"] * flops(image_size, batch)
            + theta_energy["e_byte"] * bytes_moved(image_size, batch))


if __name__ == "__main__":
    S = np.array([32, 64, 128, 224, 256, 384, 512])
    B = np.array([1, 8, 64, 256])[:, None]
    print("FLOPs per B*S^2:", FLOPS_PER_BS2, " per B:", FLOPS_PER_B, " conv per B*S^2:", CONV_FLOPS_PER_BS2)
    print("bytes per B*S^2:", sum(a + b for (_, a, b, _, _, _, h) in LAYERS if not h) * FP32,
          " weight bytes:", WEIGHT_BYTES)
    print("GFLOPs:\n", flops(S, B) / 1e9)
    print("Memory MB:\n", memory(S, B) / 2**20)
    th = {"t_launch": 5e-6, "bandwidth": 800e9, "peak_flops": 20e12}
    print("latency ms:\n", latency(S, B, th) * 1e3)
    the = {"p_static": 100.0, "e_flop": 1e-11, "e_byte": 5e-11, "latency": th}
    print("energy mJ:\n", energy(S, B, the) * 1e3)
