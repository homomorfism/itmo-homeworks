"""Step-based training loop, a PyTorch port of data_diet/train.py.

The schedule is defined in *steps*, not epochs, and is identical for every run regardless
of how much data the run uses -- this is what the original scripts do (`num_steps` and
`decay_steps` are hardcoded multiples of EP_STEPS = 390 = 50000 // 128 in every script,
including the subset ones). A run on 20% of CIFAR-10 therefore still takes 78 000
optimisation steps; it just sees each example five times as often.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import GPUData, subset_indices
from .models import get_model, get_num_params
from .scores import ForgettingTracker, el2n_scores, grand_scores

EP_STEPS = 390  # 50000 // 128, hardcoded in the original scripts

META_MODEL_SEED, META_TRAIN_SEED, META_SUBSET_SEED, SEED_INCR = 42, 4242, 4426, 424242


@dataclass
class Args:
    # bookkeeping
    name: str = "run"
    out_dir: str = "results/runs"
    data_dir: str = "data"
    run: int = 0
    # data
    dataset: str = "cifar10"
    subset: str | None = None            # None | random | keep_max_scores | keep_min_scores | offset
    subset_size: int | None = None
    subset_frac: float | None = None     # convenience: overrides subset_size
    scores_path: str | None = None
    subset_offset: int | None = None
    random_label_fraction: float = 0.0
    random_label_seed: int = 7777
    augment: bool = True
    # model
    model: str = "resnet18_lowres"
    # optimiser (original: SGD + Nesterov momentum, decoupled-free L2 on every parameter)
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    decay_factor: float = 0.2
    decay_epochs: tuple = (60, 120, 160)
    # training
    num_epochs: int = 200
    train_batch_size: int = 128
    test_batch_size: int = 1000
    amp: str = "bf16"                    # bf16 | fp32
    compile: bool = False
    # instrumentation
    log_every_epochs: int = 1
    track_forgetting: bool = False
    el2n_epochs: tuple = ()
    grand_epochs: tuple = ()
    grand_batch_size: int = 32
    save_final_ckpt: bool = False
    seed_offset: int = 0                 # extra decorrelation for retraining seeds

    @property
    def num_steps(self) -> int:
        return self.num_epochs * EP_STEPS

    @property
    def decay_steps(self):
        return [e * EP_STEPS for e in self.decay_epochs]

    @property
    def model_seed(self) -> int:
        return META_MODEL_SEED + (self.run + self.seed_offset) * SEED_INCR

    @property
    def train_seed(self) -> int:
        return META_TRAIN_SEED + (self.run + self.seed_offset) * SEED_INCR

    @property
    def subset_seed(self) -> int:
        return META_SUBSET_SEED + (self.run + self.seed_offset) * SEED_INCR


def lr_at(args: Args, step: int) -> float:
    """flax.training.lr_schedule.create_stepped_learning_rate_schedule, steps_per_epoch=1."""
    lr = args.lr
    for i, boundary in enumerate(args.decay_steps):
        if step >= boundary:
            lr = args.lr * args.decay_factor ** (i + 1)
    return lr


@torch.no_grad()
def evaluate(model, data, batch_size, amp_dtype):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in data.test_batches(batch_size):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        n += y.numel()
    model.train()
    return loss_sum / n, correct / n


def train(args: Args, data: GPUData | None = None) -> dict:
    """`data` may be a GPUData built earlier; reusing it saves ~20 s of startup per run."""
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else None

    run_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "scores"), exist_ok=True)

    if data is None:
        data = GPUData(args.dataset, args.data_dir, device,
                       args.random_label_fraction, args.random_label_seed)

    # ---- subset ------------------------------------------------------------------
    scores = np.load(args.scores_path) if args.scores_path else None
    size = args.subset_size
    if args.subset_frac is not None:
        size = int(round(args.subset_frac * data.n_train))
    idx = subset_indices(args.subset, data.n_train, size, scores,
                         args.subset_offset, args.subset_seed)

    # ---- model / optimiser -------------------------------------------------------
    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    model = get_model(args.model, data.num_classes).to(device).to(memory_format=torch.channels_last)
    model.train()
    n_params = get_num_params(model)
    fwd = torch.compile(model) if args.compile else model
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=args.nesterov)

    tracker = ForgettingTracker(data.n_train, device) if args.track_forgetting else None
    el2n_eps, grand_eps = set(args.el2n_epochs), set(args.grand_epochs)

    cfg = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(args).items()}
    cfg.update(n_train=int(idx.size), num_params=n_params, num_steps=args.num_steps,
               model_seed=args.model_seed, train_seed=args.train_seed,
               subset_seed=args.subset_seed)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    np.save(os.path.join(run_dir, "subset_idx.npy"), idx)

    rows = []
    t0 = time.time()
    log_every = args.log_every_epochs * EP_STEPS
    running_loss, running_correct, running_n = 0.0, 0, 0

    for step, bidx, x, y in data.train_batches(idx, args.train_batch_size, args.num_steps,
                                               0, args.train_seed, args.augment):
        lr = lr_at(args, step)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = fwd(x)
            loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            correct = logits.detach().float().argmax(-1) == y
            if tracker is not None:
                tracker.update(bidx, correct)
            running_loss += loss.detach() * y.numel()
            running_correct += correct.sum()
            running_n += y.numel()

        epoch = step // EP_STEPS
        at_epoch_end = step % EP_STEPS == 0

        if step % log_every == 0 or step == args.num_steps:
            tr_loss = (running_loss / running_n).item()
            tr_acc = (running_correct / running_n).item()
            running_loss, running_correct, running_n = 0.0, 0, 0
            te_loss, te_acc = evaluate(model, data, args.test_batch_size, amp_dtype)
            rows.append(dict(step=step, epoch=epoch, lr=lr, train_loss=tr_loss,
                             train_acc=tr_acc, test_loss=te_loss, test_acc=te_acc,
                             wall=time.time() - t0))
            print(f"[{args.name}] {100*step/args.num_steps:5.1f}% ep {epoch:3d} "
                  f"lr {lr:.4f} train {tr_acc:.3f} test {te_acc:.4f} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)

        if at_epoch_end and epoch in el2n_eps:
            np.save(os.path.join(run_dir, "scores", f"el2n_ep{epoch}.npy"),
                    el2n_scores(model, data, args.test_batch_size))
            if tracker is not None:
                np.save(os.path.join(run_dir, "scores", f"forget_ep{epoch}.npy"), tracker.scores())
        if at_epoch_end and epoch in grand_eps:
            np.save(os.path.join(run_dir, "scores", f"grand_ep{epoch}.npy"),
                    grand_scores(model, data, args.grand_batch_size))

    # ---- wrap up -----------------------------------------------------------------
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    if tracker is not None:
        np.save(os.path.join(run_dir, "scores", f"forget_ep{args.num_epochs}.npy"), tracker.scores())
    if args.save_final_ckpt:
        torch.save(model.state_dict(), os.path.join(run_dir, "final.pt"))

    summary = dict(name=args.name, n_train=int(idx.size), num_params=n_params,
                   final_test_acc=float(df.test_acc.iloc[-1]),
                   best_test_acc=float(df.test_acc.max()),
                   final_train_acc=float(df.train_acc.iloc[-1]),
                   wall_minutes=(time.time() - t0) / 60.0)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.name}] DONE final {summary['final_test_acc']:.4f} "
          f"best {summary['best_test_acc']:.4f} in {summary['wall_minutes']:.1f}m", flush=True)
    return summary
