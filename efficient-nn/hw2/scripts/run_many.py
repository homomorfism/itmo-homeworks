#!/usr/bin/env python
"""Run a list of training jobs inside one process.

CUDA init plus loading CIFAR-10 onto the GPU costs ~20-25 s, which over ~130 runs is an
hour of pure startup. This runner executes a whole stage in one process and caches the
GPUData per (dataset, label-noise) combination. Each job is still independent: a failure
is reported and the rest continue, and finished runs are skipped on a restart.
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from data_diet.data import GPUData  # noqa: E402
from data_diet.train import Args, train  # noqa: E402

_CACHE: dict = {}


def get_data(a: Args) -> GPUData:
    key = (a.dataset, a.data_dir, a.random_label_fraction, a.random_label_seed)
    if key not in _CACHE:
        _CACHE.clear()  # at most one dataset copy on the GPU at a time
        torch.cuda.empty_cache()
        _CACHE[key] = GPUData(a.dataset, a.data_dir, torch.device("cuda"),
                              a.random_label_fraction, a.random_label_seed)
    return _CACHE[key]


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    deadline = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    ok = 0
    for i, kw in enumerate(jobs, 1):
        for k in ("el2n_epochs", "grand_epochs", "decay_epochs"):
            if k in kw and kw[k] is not None:
                kw[k] = tuple(kw[k])
        a = Args(**kw)
        if (Path(a.out_dir) / a.name / "summary.json").exists():
            print(f"[{i}/{len(jobs)}] skip   {a.name}", flush=True)
            ok += 1
            continue
        if deadline and time.time() > deadline:
            print(f"[{i}/{len(jobs)}] deadline reached, stopping before {a.name}", flush=True)
            break
        t0 = time.time()
        try:
            train(a, get_data(a))
            ok += 1
            print(f"[{i}/{len(jobs)}] ok     {(time.time()-t0)/60:5.1f}m  {a.name}", flush=True)
        except Exception:
            print(f"[{i}/{len(jobs)}] FAIL   {a.name}", flush=True)
            traceback.print_exc()
        torch.cuda.empty_cache()
    print(f"run_many: {ok}/{len(jobs)} ok", flush=True)
    sys.exit(0 if ok == len(jobs) else 1)


if __name__ == "__main__":
    main()
