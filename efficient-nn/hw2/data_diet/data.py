"""Data loading, GPU-side augmentation and subset selection.

The whole dataset lives on the GPU as uint8 (CIFAR-10 train = 146 MiB), and augmentation
(reflect-pad 4 -> random 32x32 crop -> random horizontal flip) is done on the GPU. This
removes the host dataloader from the critical path entirely, which matters because a
single reproduction run is 78k steps of a very small network.

Index convention: scores, forgetting counts and subsets are all indexed by position in
the *original* torchvision train split order. The original JAX code sorts the dataset by
class first; that only permutes the index space and is irrelevant here, except that we do
not reproduce its (buggy) in-place label-randomisation, see `randomize_labels`.
"""
from __future__ import annotations

import numpy as np
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
# the original code deliberately reuses the CIFAR-10 statistics for CIFAR-100
CIFAR100_MEAN, CIFAR100_STD = CIFAR10_MEAN, CIFAR10_STD

DATASETS = {
    "cifar10": dict(num_classes=10, mean=CIFAR10_MEAN, std=CIFAR10_STD),
    "cifar100": dict(num_classes=100, mean=CIFAR100_MEAN, std=CIFAR100_STD),
}


def load_raw(dataset: str, data_dir: str):
    """-> (X_train uint8 [N,32,32,3], y_train int64 [N], X_test, y_test). Cached as .npy."""
    import os

    cache = os.path.join(data_dir, f"{dataset}_raw.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["Xtr"], z["ytr"], z["Xte"], z["yte"]

    from torchvision import datasets as tvd

    cls = {"cifar10": tvd.CIFAR10, "cifar100": tvd.CIFAR100}[dataset]
    tr = cls(root=data_dir, train=True, download=True)
    te = cls(root=data_dir, train=False, download=True)
    Xtr, ytr = np.asarray(tr.data, dtype=np.uint8), np.asarray(tr.targets, dtype=np.int64)
    Xte, yte = np.asarray(te.data, dtype=np.uint8), np.asarray(te.targets, dtype=np.int64)
    os.makedirs(data_dir, exist_ok=True)
    np.savez(cache, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


def randomize_labels(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Permute the labels of a random `fraction` of the examples among themselves.

    This mirrors the intent of `randomize_labels` in the original repository (which
    shuffles labels inside the selected subset of a class-sorted array). A permutation
    leaves the label marginal untouched and corrupts ~fraction * (1 - 1/K) of the labels.
    """
    if fraction <= 0:
        return y.copy()
    rng = np.random.RandomState(seed)
    n = y.shape[0]
    idx = rng.choice(n, int(round(n * fraction)), replace=False)
    y_rand = y.copy()
    y_rand[np.sort(idx)] = y[idx]
    return y_rand


# ---------------------------------------------------------------------------------------
#  Subset selection
# ---------------------------------------------------------------------------------------
def subset_indices(subset: str, n: int, size: int, scores: np.ndarray | None,
                   offset: int | None, seed: int | None) -> np.ndarray:
    """All selections return sorted int64 indices into the full train split."""
    if subset in (None, "none", "full"):
        return np.arange(n, dtype=np.int64)
    if subset == "random":
        rng = np.random.RandomState(seed)
        idx = rng.choice(n, size, replace=False)
    elif subset == "keep_max_scores":
        idx = np.argsort(scores, kind="stable")[-size:]
    elif subset == "keep_min_scores":
        idx = np.argsort(scores, kind="stable")[:size]
    elif subset == "offset":
        # window of `size` examples starting `offset` positions below the top score
        order = np.argsort(scores, kind="stable")[::-1]  # descending
        idx = order[offset: offset + size]
    else:
        raise NotImplementedError(subset)
    return np.sort(idx).astype(np.int64)


# ---------------------------------------------------------------------------------------
#  GPU tensors + augmentation
# ---------------------------------------------------------------------------------------
class GPUData:
    """Holds the dataset on the GPU and yields augmented training batches."""

    def __init__(self, dataset: str, data_dir: str, device: torch.device,
                 random_label_fraction: float = 0.0, random_label_seed: int = 0):
        Xtr, ytr, Xte, yte = load_raw(dataset, data_dir)
        self.num_classes = DATASETS[dataset]["num_classes"]
        mean = torch.tensor(DATASETS[dataset]["mean"], device=device).view(1, 3, 1, 1) * 255.0
        std = torch.tensor(DATASETS[dataset]["std"], device=device).view(1, 3, 1, 1) * 255.0
        self.mean, self.std = mean, std
        self.device = device

        self.y_train_clean = torch.from_numpy(ytr).to(device)
        ytr_used = randomize_labels(ytr, random_label_fraction, random_label_seed)
        self.noisy_mask = torch.from_numpy(ytr_used != ytr).to(device)

        # NHWC uint8 -> NCHW uint8 on device
        self.X_train_u8 = torch.from_numpy(Xtr).to(device).permute(0, 3, 1, 2).contiguous()
        self.y_train = torch.from_numpy(ytr_used).to(device)
        Xte_t = torch.from_numpy(Xte).to(device).permute(0, 3, 1, 2).float()
        self.X_test = ((Xte_t - mean) / std).contiguous(memory_format=torch.channels_last)
        self.y_test = torch.from_numpy(yte).to(device)
        self.n_train, self.n_test = Xtr.shape[0], Xte.shape[0]

    # -- batch construction ---------------------------------------------------------
    def _normalize(self, x_u8: torch.Tensor) -> torch.Tensor:
        return (x_u8.float() - self.mean) / self.std

    def augment(self, x_u8: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """reflect-pad 4 -> random 32x32 crop -> random horizontal flip (p=0.5)."""
        B, C, H, W = x_u8.shape
        x = self._normalize(x_u8)
        x = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
        oy = torch.randint(0, 9, (B,), device=x.device, generator=gen)
        ox = torch.randint(0, 9, (B,), device=x.device, generator=gen)
        ar = torch.arange(H, device=x.device)
        rows = (oy[:, None] + ar)[:, None, :, None].expand(B, C, H, W)
        cols = (ox[:, None] + ar)[:, None, None, :].expand(B, C, H, W)
        bi = torch.arange(B, device=x.device)[:, None, None, None].expand(B, C, H, W)
        ci = torch.arange(C, device=x.device)[None, :, None, None].expand(B, C, H, W)
        x = x[bi, ci, rows, cols]
        flip = torch.rand(B, device=x.device, generator=gen) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
        return x.contiguous(memory_format=torch.channels_last)

    def eval_batch(self, idx: torch.Tensor) -> torch.Tensor:
        """Un-augmented, normalised training images (used for scoring)."""
        return self._normalize(self.X_train_u8[idx]).contiguous(memory_format=torch.channels_last)

    def train_batches(self, subset_idx: np.ndarray, batch_size: int, num_steps: int,
                      start_step: int, seed: int, augment: bool = True):
        """Yield (step, batch_idx, x, y); reshuffles every epoch and drops the remainder,
        exactly like `train_batches` in the original repository."""
        idx = torch.from_numpy(subset_idx).to(self.device)
        n = idx.numel()
        shuf = torch.Generator(device=self.device).manual_seed(seed)
        aug = torch.Generator(device=self.device).manual_seed(seed + 1)
        order = idx[torch.randperm(n, device=self.device, generator=shuf)]
        step, start = start_step + 1, 0
        while step <= num_steps:
            end = start + batch_size
            if end > n:
                order = idx[torch.randperm(n, device=self.device, generator=shuf)]
                start = 0
                continue
            bidx = order[start:end]
            x_u8 = self.X_train_u8[bidx]
            x = self.augment(x_u8, aug) if augment else \
                self._normalize(x_u8).contiguous(memory_format=torch.channels_last)
            yield step, bidx, x, self.y_train[bidx]
            step += 1
            start = end

    def test_batches(self, batch_size: int):
        for s in range(0, self.n_test, batch_size):
            e = min(s + batch_size, self.n_test)
            yield self.X_test[s:e], self.y_test[s:e]

    def full_train_batches(self, batch_size: int):
        for s in range(0, self.n_train, batch_size):
            e = min(s + batch_size, self.n_train)
            idx = torch.arange(s, e, device=self.device)
            yield idx, self.eval_batch(idx), self.y_train[idx]
