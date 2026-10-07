#!/usr/bin/env python
"""CIFAR-100 pruning campaign -- the dataset the single-RTX-3090 budget skipped.

Mirrors the CIFAR-10 protocol of `sweep.py` exactly (same optimiser, schedule, 100-epoch
budget, step-count independent of subset size, scores averaged over 10 inits), only the
dataset changes. Writes everything under `results/cifar100/` so the CIFAR-10 campaign is
untouched.

Stages (each resumable -- a run with a summary.json is skipped):
  scores : 10 runs x 20 epochs  -> E_w[EL2N] (ep 1..20), E_w[GraNd] (ep 1,10,20)
  full   : 6  runs x 100 epochs -> baseline + seed spread + forgetting score
  (mean-score averaging over the runs above)
  prune  : 9 keep-fractions x {random, EL2N@20, GraNd@1, GraNd@20, forgetting}

    uv run scripts/campaign_cifar100.py                 # whole campaign
    uv run scripts/campaign_cifar100.py --stages prune  # just the grid (needs scores done)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".venv" / "bin" / "python")

from scripts import fleet  # noqa: E402

OUT = ROOT / "results" / "cifar100"
RUNS = OUT / "runs"
SCORES = OUT / "scores"
LOGS = OUT / "logs"

DATASET = "cifar100"
NUM_EPOCHS = 100
DECAY_EPOCHS = [30, 60, 80]
SCORING_EPOCHS = 20
N_SCORE_RUNS = 10
N_FULL_RUNS = 6
EL2N_EPOCHS_SHORT = [1, 2, 3, 4, 5, 10, 15, 20]
GRAND_EPOCHS = [1, 10, 20]
KEEP_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def base(**kw):
    d = dict(dataset=DATASET, amp="bf16", out_dir=str(RUNS), data_dir="data",
             num_epochs=NUM_EPOCHS, decay_epochs=DECAY_EPOCHS, log_every_epochs=5)
    d.update(kw)
    return d


def jobs_scores():
    return [base(name=f"scores/run{r}", run=r, num_epochs=SCORING_EPOCHS,
                 el2n_epochs=EL2N_EPOCHS_SHORT, grand_epochs=GRAND_EPOCHS)
            for r in range(N_SCORE_RUNS)]


def jobs_full():
    return [base(name=f"full/run{r}", run=r, track_forgetting=True, log_every_epochs=1,
                 el2n_epochs=[20, NUM_EPOCHS // 2, NUM_EPOCHS])
            for r in range(N_FULL_RUNS)]


def mean_scores(run_root, score, n, tag):
    out = SCORES / f"{tag}.npy"
    if not out.exists():
        subprocess.run([PY, str(ROOT / "scripts" / "mean_scores.py"), "--runs", str(run_root),
                        "--score", score, "--n", str(n), "--out", str(SCORES), "--tag", tag],
                       check=True, cwd=str(ROOT))
    return str(out)


def build_scores():
    SCORES.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["el2n_ep20"] = mean_scores(RUNS / "scores", "el2n_ep20", N_SCORE_RUNS,
                                     f"el2n_ep20_mean{N_SCORE_RUNS}")
    paths["el2n_ep10"] = mean_scores(RUNS / "scores", "el2n_ep10", N_SCORE_RUNS,
                                     f"el2n_ep10_mean{N_SCORE_RUNS}")
    paths["grand_ep1"] = mean_scores(RUNS / "scores", "grand_ep1", N_SCORE_RUNS,
                                     f"grand_ep1_mean{N_SCORE_RUNS}")
    paths["grand_ep20"] = mean_scores(RUNS / "scores", "grand_ep20", N_SCORE_RUNS,
                                      f"grand_ep20_mean{N_SCORE_RUNS}")
    paths["forget_full"] = mean_scores(RUNS / "full", f"forget_ep{NUM_EPOCHS}", N_FULL_RUNS,
                                       f"forget_ep{NUM_EPOCHS}_mean{N_FULL_RUNS}")
    return paths


# score -> (subset mode, path key). "random" is handled separately.
METHODS = {
    "el2n_ep20": "el2n_ep20",
    "grand_ep1": "grand_ep1",
    "grand_ep20": "grand_ep20",
    "forget_full": "forget_full",
}


def jobs_prune(paths):
    jobs = []
    for frac in KEEP_FRACS:
        tag = f"keep{int(frac*100):03d}"
        jobs.append(base(name=f"prune/random/{tag}/run0", run=0, subset="random",
                         subset_frac=frac, seed_offset=100))
        for meth, key in METHODS.items():
            jobs.append(base(name=f"prune/{meth}/{tag}/run0", run=0, subset="keep_max_scores",
                             subset_frac=frac, scores_path=paths[key], seed_offset=100))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="scores,full,prune")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--per_gpu", type=int, default=8)
    a = ap.parse_args()
    gpus = tuple(int(x) for x in a.gpus.split(","))
    stages = a.stages.split(",")
    LOGS.mkdir(parents=True, exist_ok=True)

    # scores and full are independent; run them together to fill the fleet
    stage1 = []
    if "scores" in stages:
        stage1 += jobs_scores()
    if "full" in stages:
        stage1 += jobs_full()
    if stage1:
        fleet.run_jobs(stage1, gpus=gpus, per_gpu=a.per_gpu, log_dir=str(LOGS), tag="c100_s1")

    if "prune" in stages:
        paths = build_scores()
        fleet.run_jobs(jobs_prune(paths), gpus=gpus, per_gpu=a.per_gpu, log_dir=str(LOGS),
                       tag="c100_prune")
    print("CAMPAIGN DONE", flush=True)


if __name__ == "__main__":
    main()
