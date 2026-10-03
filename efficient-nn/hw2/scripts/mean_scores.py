#!/usr/bin/env python
"""Average a score across independent runs: E_w[score] of the paper's definitions.

  mean_scores.py --runs results/runs/full --score el2n_ep20 --n 10 --out results/scores
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def mean_score(run_root: str, score: str, n_runs: int) -> np.ndarray:
    xs = []
    for r in range(n_runs):
        p = os.path.join(run_root, f"run{r}", "scores", f"{score}.npy")
        xs.append(np.load(p))
    x = np.stack(xs)
    if not np.isfinite(x).all():  # forgetting scores use +inf for never-learned examples
        big = np.nanmax(np.where(np.isfinite(x), x, -np.inf)) + 1.0
        x = np.where(np.isfinite(x), x, big)
    return x.mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="results/runs/full")
    p.add_argument("--score", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--out", default="results/scores")
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    s = mean_score(a.runs, a.score, a.n)
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or f"{a.score}_mean{a.n}"
    np.save(os.path.join(a.out, f"{tag}.npy"), s)
    print(f"{tag}: n={s.size} mean={s.mean():.4f} std={s.std():.4f} "
          f"min={s.min():.4f} max={s.max():.4f}")


if __name__ == "__main__":
    main()
