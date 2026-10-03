"""EL2N, GraNd and forgetting scores.

Definitions follow the paper (Paul, Ganguli & Dziugaite, NeurIPS 2021):

  GraNd (Def. 2.1)   chi_t(x, y) = E_w || grad_w  CE(p(w, x), y) ||_2
  EL2N  (Def. 2.3)   E_w || softmax(f(w, x)) - onehot(y) ||_2

Both are expectations over weight initialisations, estimated in the paper (and here) by
the mean over 10 independent training runs. As in the original implementation, the network
is put in eval mode (BatchNorm uses its running statistics) and images are *not* augmented.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def el2n_scores(model, data, batch_size: int = 1000) -> np.ndarray:
    """|| softmax(logits) - onehot(y) ||_2 for every training example."""
    was_training = model.training
    model.eval()
    out = torch.empty(data.n_train, device=data.device)
    for idx, x, y in data.full_train_batches(batch_size):
        p = F.softmax(model(x).float(), dim=-1)
        err = p - F.one_hot(y, data.num_classes).float()
        out[idx] = err.norm(dim=-1)
    model.train(was_training)
    return out.cpu().numpy()


@torch.no_grad()
def margin_scores(model, data, batch_size: int = 1000) -> np.ndarray:
    """max_{k != y} p_k - p_y (the paper's 'max margin' baseline)."""
    was_training = model.training
    model.eval()
    out = torch.empty(data.n_train, device=data.device)
    for idx, x, y in data.full_train_batches(batch_size):
        p = F.softmax(model(x).float(), dim=-1)
        p_true = p.gather(1, y[:, None])
        p_other = p.scatter(1, y[:, None], -float("inf"))
        out[idx] = (p_other.max(dim=-1).values - p_true.squeeze(1))
    model.train(was_training)
    return out.cpu().numpy()


def grand_scores(model, data, batch_size: int = 32) -> np.ndarray:
    """Per-example gradient norm over *all* parameters, via torch.func.vmap(grad(.)).

    The original code computes the full per-sample Jacobian with jax.jacrev and takes its
    row norms; vmap(grad(...)) is the same quantity computed without materialising the
    Jacobian for the whole batch at once.
    """
    from torch.func import functional_call, grad, vmap

    was_training = model.training
    model.eval()
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_fn(p, b, x, y):
        logits = functional_call(model, (p, b), (x.unsqueeze(0),))
        return F.cross_entropy(logits, y.unsqueeze(0))

    per_sample_grad = vmap(grad(loss_fn), in_dims=(None, None, 0, 0))
    out = torch.empty(data.n_train, device=data.device)
    for idx, x, y in data.full_train_batches(batch_size):
        g = per_sample_grad(params, buffers, x, y)
        sq = None
        for v in g.values():
            s = v.reshape(v.shape[0], -1).pow(2).sum(1)
            sq = s if sq is None else sq + s
        out[idx] = sq.sqrt()
    model.train(was_training)
    return out.cpu().numpy()


# ---------------------------------------------------------------------------------------
#  Forgetting score (Toneva et al., 2019), as tracked by the original repository
# ---------------------------------------------------------------------------------------
class ForgettingTracker:
    """Counts, per example, how often a correct prediction turns into a wrong one.

    Updated from the predictions of the *training* forward pass (augmented inputs,
    BatchNorm in train mode), which is what the original implementation does. Examples
    that are never classified correctly get score +inf, so they rank as the hardest.
    """

    def __init__(self, n: int, device):
        self.prev_acc = torch.zeros(n, dtype=torch.int8, device=device)
        self.num_forgets = torch.zeros(n, dtype=torch.float32, device=device)
        self.ever_correct = torch.zeros(n, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, idx: torch.Tensor, correct: torch.Tensor):
        c = correct.to(torch.int8)
        forgot = self.prev_acc[idx] > c
        self.num_forgets[idx] += forgot.float()
        self.prev_acc[idx] = c
        self.ever_correct[idx] |= correct

    def scores(self) -> np.ndarray:
        s = self.num_forgets.clone()
        s[~self.ever_correct] = float("inf")
        return s.cpu().numpy()
