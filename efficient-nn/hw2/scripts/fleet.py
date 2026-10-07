#!/usr/bin/env python
"""Run a list of training jobs across several GPUs, `per_gpu` workers on each.

The single-GPU `sweep.py` runs a stage in one `run_many.py` process because one training
process already saturates an RTX 3090. On a datacentre GPU (A800) a single tiny ResNet-18
at batch 128 is launch-latency-bound and leaves the card mostly idle, so the throughput
lever is the opposite one: pack several processes per GPU and spread them over the fleet.

This runner shards a job list round-robin into `len(gpus) * per_gpu` shards and runs each
shard as its own `run_many.py` process pinned to one GPU via `CUDA_VISIBLE_DEVICES`. Each
process caches `GPUData`, skips runs that already have a `summary.json`, and reports its own
failures -- exactly the single-GPU behaviour, just fanned out. Jobs inside one stage are
homogeneous (same epoch count), so round-robin balances the shards.

Measured aggregate throughput, bf16, ResNet-18, CIFAR batch 128:
  1 proc/GPU  34 steps/s      4 proc/GPU  73 steps/s
  8 proc/GPU  86 steps/s     12 proc/GPU  98 steps/s
so ~8 workers/GPU is the knee; four A800s then do ~340 steps/s, ~4.3x one RTX 3090.
"""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNMANY = str(ROOT / "scripts" / "run_many.py")


def run_jobs(jobs, gpus=(0, 1, 2, 3), per_gpu=8, log_dir=None, tag="fleet", deadline=0.0):
    """Run `jobs` (a list of run_many.py job-spec dicts) across the fleet. Blocks.

    Returns the number of shards that exited 0. A shard is a `run_many.py` process; its
    own stdout (one line per job: ok/skip/FAIL) is tee'd to `{log_dir}/{tag}.shard{i}.log`.
    """
    jobs = list(jobs)
    if not jobs:
        print(f"[fleet {tag}] nothing to do", flush=True)
        return 0
    log_dir = Path(log_dir or (ROOT / "results" / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    nshard = min(len(gpus) * per_gpu, len(jobs))
    shards = [[] for _ in range(nshard)]
    for i, j in enumerate(jobs):
        shards[i % nshard].append(j)

    print(f"[fleet {tag}] {len(jobs)} jobs over {nshard} shards "
          f"({len(gpus)} gpus x up to {per_gpu}) ...", flush=True)
    procs = []
    for s, shard in enumerate(shards):
        gpu = gpus[s % len(gpus)]
        spec = log_dir / f"{tag}.shard{s}.json"
        spec.write_text(json.dumps(shard))
        logf = open(log_dir / f"{tag}.shard{s}.log", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT))
        p = subprocess.Popen([PY, RUNMANY, str(spec), str(deadline)],
                             stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        procs.append((p, logf, gpu, s))

    t0 = time.time()
    ok = 0
    for p, logf, gpu, s in procs:
        rc = p.wait()
        logf.close()
        ok += rc == 0
    dt = (time.time() - t0) / 60.0

    # surface the per-job lines from every shard, in run order
    done = fail = skip = 0
    for _, _, gpu, s in procs:
        for line in (log_dir / f"{tag}.shard{s}.log").read_text().splitlines():
            if " ok " in line:
                done += 1
            elif "FAIL" in line:
                fail += 1
                print(f"  [gpu{gpu} shard{s}] {line}", flush=True)
            elif "skip" in line:
                skip += 1
    print(f"[fleet {tag}] done in {dt:.1f}m  ok={done} skip={skip} fail={fail} "
          f"(shards 0-exit {ok}/{len(procs)})", flush=True)
    return done, skip, fail
