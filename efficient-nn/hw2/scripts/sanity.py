#!/usr/bin/env python
"""Checks that the port matches the original implementation where it can be checked
cheaply, plus a throughput benchmark used to budget the sweep."""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_diet.data import GPUData, randomize_labels, subset_indices  # noqa: E402
from data_diet.models import get_model, get_num_params  # noqa: E402
from data_diet.scores import el2n_scores, grand_scores  # noqa: E402
from data_diet.train import Args, lr_at  # noqa: E402

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data")
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ---- model -------------------------------------------------------------------------
m = get_model("resnet18_lowres", 10)
n = get_num_params(m)
# flax ResNet18(lowres, num_classes=10) on 32x32x3 has 11 173 962 parameters
check("resnet18_lowres parameter count", n == 11_173_962, f"got {n}")
m50 = get_model("resnet50_lowres", 100)
print(f"      resnet50_lowres/cifar100 params: {get_num_params(m50):,}")
x = torch.randn(4, 3, 32, 32)
check("forward shape", tuple(m(x).shape) == (4, 10), str(tuple(m(x).shape)))
w = m.conv_init.weight
check("lecun_normal init std ~ 1/sqrt(fan_in)",
      abs(w.std().item() - (1 / 27) ** 0.5) < 0.02, f"std={w.std().item():.4f} vs {(1/27)**0.5:.4f}")

# ---- lr schedule -------------------------------------------------------------------
a = Args()
sched = [(1, 0.1), (23399, 0.1), (23400, 0.02), (46799, 0.02), (46800, 0.004),
         (62399, 0.004), (62400, 0.0008), (78000, 0.0008)]
check("stepped lr schedule", all(abs(lr_at(a, s) - v) < 1e-9 for s, v in sched))
check("num_steps == 200 * 390", a.num_steps == 78000, str(a.num_steps))

# ---- subset selection ---------------------------------------------------------------
sc = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
check("keep_max_scores", list(subset_indices("keep_max_scores", 5, 2, sc, None, None)) == [0, 2])
check("keep_min_scores", list(subset_indices("keep_min_scores", 5, 2, sc, None, None)) == [1, 3])
check("offset window", list(subset_indices("offset", 5, 2, sc, 1, None)) == [2, 4],
      "skip top-1 then take the next 2 -> {4.0, 3.0}")
i1 = subset_indices("random", 100, 10, None, None, 0)
check("random subset is reproducible and sorted",
      (i1 == subset_indices("random", 100, 10, None, None, 0)).all() and (np.diff(i1) > 0).all())

# ---- label randomisation ------------------------------------------------------------
y = np.repeat(np.arange(10), 5000)
yr = randomize_labels(y, 0.1, 7777)
frac = (yr != y).mean()
check("10% label permutation corrupts ~9%", 0.085 < frac < 0.095, f"{frac:.4f}")
check("label marginal preserved", (np.bincount(yr) == np.bincount(y)).all())

if not torch.cuda.is_available():
    print("\nno CUDA: skipping data/score/throughput checks")
    sys.exit(0 if ok else 1)

dev = torch.device("cuda")
data = GPUData("cifar10", DATA_DIR, dev)
check("train/test sizes", (data.n_train, data.n_test) == (50000, 10000),
      f"{data.n_train}/{data.n_test}")
xb = data.eval_batch(torch.arange(2048, device=dev))
check("normalised images ~ N(0,1)", abs(xb.mean().item()) < 0.15 and abs(xb.std().item() - 1) < 0.2,
      f"mean={xb.mean().item():.3f} std={xb.std().item():.3f}")

g = torch.Generator(device=dev).manual_seed(0)
xa = data.augment(data.X_train_u8[:512], g)
check("augment keeps shape", tuple(xa.shape) == (512, 3, 32, 32))
xa2 = data.augment(data.X_train_u8[:512], torch.Generator(device=dev).manual_seed(0))
check("augmentation is seed-reproducible", torch.equal(xa, xa2))
check("augmentation actually changes pixels", not torch.allclose(xa, xb[:512]))

# batch iterator: same example must not repeat inside an epoch
it = data.train_batches(np.arange(50000), 128, 390, 0, 0)
seen = torch.cat([b[1] for b in it])
check("epoch covers distinct examples", seen.unique().numel() == seen.numel() == 390 * 128)

# ---- scores ------------------------------------------------------------------------
mm = get_model("resnet18_lowres", 10).to(dev).to(memory_format=torch.channels_last)
mm.eval()
e = el2n_scores(mm, data, 1000)
check("EL2N range", e.shape == (50000,) and e.min() >= 0 and e.max() <= np.sqrt(2) + 1e-5,
      f"[{e.min():.3f}, {e.max():.3f}]  mean={e.mean():.3f}")
# a uniform predictor gives ||p - onehot||_2 = sqrt(0.9^2 + 9*0.01) = 0.9487; an untrained
# net with untrained BatchNorm buffers is usually more confident than that, so only report
print(f"      EL2N at init: mean {e.mean():.4f} (uniform predictor would give 0.9487)")

# GraNd must agree with an explicit per-sample backward pass
idx = torch.arange(8, device=dev)
xs, ys = data.eval_batch(idx), data.y_train[idx]
ref = []
for i in range(8):
    mm.zero_grad(set_to_none=True)
    F.cross_entropy(mm(xs[i:i + 1]), ys[i:i + 1]).backward()
    ref.append(torch.cat([p.grad.flatten() for p in mm.parameters()]).norm().item())
gn = grand_scores(mm, data, 8)[:8]
check("GraNd == per-sample backward norm", np.allclose(gn, ref, rtol=2e-3, atol=1e-5),
      f"max rel err {np.max(np.abs(gn - np.array(ref)) / np.array(ref)):.2e}")

# ---- throughput --------------------------------------------------------------------
print("\nthroughput (ResNet-18, batch 128, channels_last):")
for amp in ("fp32", "bf16"):
    mm = get_model("resnet18_lowres", 10).to(dev).to(memory_format=torch.channels_last).train()
    opt = torch.optim.SGD(mm.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    dt = torch.bfloat16 if amp == "bf16" else None
    torch.backends.cudnn.benchmark = True
    it = data.train_batches(np.arange(50000), 128, 10**9, 0, 0)
    for i, (_, _, x, yb) in enumerate(it):
        if i == 60:
            torch.cuda.synchronize(); t0 = time.time()
        if i == 360:
            break
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            loss = F.cross_entropy(mm(x).float(), yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    sps = 300 / (time.time() - t0)
    print(f"      {amp}: {sps:7.1f} steps/s -> {390/sps:5.2f} s/epoch -> "
          f"{78000/sps/60:5.1f} min per 200-epoch run")

t0 = time.time(); el2n_scores(mm, data, 1000); torch.cuda.synchronize()
print(f"      EL2N over 50k: {time.time()-t0:.1f}s")
t0 = time.time(); grand_scores(mm, data, 32); torch.cuda.synchronize()
print(f"      GraNd over 50k (batch 32): {time.time()-t0:.1f}s")
print(f"      peak GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
