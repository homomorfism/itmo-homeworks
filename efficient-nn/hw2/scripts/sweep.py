#!/usr/bin/env python
"""Run the reproduction plan on a single GPU.

One training process already saturates an RTX 3090 at this model size (measured:
79 steps/s alone, 72-73 steps/s aggregate with 2-6 concurrent processes), so the plan is
executed sequentially and the only real lever is the number and length of the runs.

Stages run in priority order and each one is resumable: a run that already has a
`summary.json` is skipped, so the sweep can be stopped and restarted, and later stages
can be dropped if the clock runs out.

Measured cost on an RTX 3090 (bf16, batch 128): 4.9 s per nominal epoch of 390 steps,
i.e. 8.2 min for a 100-epoch run, 16.4 min for a 200-epoch run, 1.7 min for a 20-epoch
scoring run, plus 40 s per GraNd snapshot.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
RUNS = ROOT / "results" / "runs"
SCORES = ROOT / "results" / "scores"

# EL2N is one eval pass (2.8 s), so snapshot it often; GraNd needs per-example gradients
# over all 11.2M parameters (39 s), so only where a figure needs it.
EL2N_EPOCHS_SHORT = (1, 2, 3, 4, 5, 10, 15, 20)
GRAND_EPOCHS = (1, 10, 20)
METHODS = ("random", "el2n_ep20", "grand_ep1", "forget_full")
KEEP_FRACS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
EXTRA_SEED_FRACS = (0.5, 0.8)          # a second seed only here, to show the seed spread
SCORE_EPOCH_SWEEP = (1, 2, 5, 10, 20)  # "how early is early enough", at 50% kept
OFFSETS = (0, 250, 500, 1000, 2500, 5000, 10000)
NOISE_FRACS = (0.2, 0.4, 0.6, 0.8)
SCORING_EPOCHS = 20                    # a 20-epoch run is a prefix of the full run: lr is
                                       # still 0.1 everywhere before the first decay
MIN_PER_EPOCH = 0.082                  # measured: 4.9 s per 390-step epoch, bf16, RTX 3090


class Job:
    def __init__(self, name, minutes=0.0, **kw):
        self.name, self.minutes, self.kw = name, minutes, kw

    def spec(self):
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.kw.items()}
        d["name"] = self.name
        return d

    def done(self):
        return (RUNS / self.name / "summary.json").exists()


def run_stage(name, jobs, log_dir, deadline=None):
    """Execute a stage's jobs in ONE python process (see scripts/run_many.py)."""
    todo = [j for j in jobs if not j.done()]
    est = sum(j.minutes for j in todo)
    print(f"\n=== stage {name}: {len(jobs)} jobs, {len(todo)} to run, ~{est:.0f} min est ===",
          flush=True)
    if not todo:
        return True
    os.makedirs(log_dir, exist_ok=True)
    if deadline and time.time() > deadline:
        print("  deadline reached, stage skipped", flush=True)
        return False
    spec = Path(log_dir) / f"{name}.jobs.json"
    spec.write_text(json.dumps([j.spec() for j in todo], indent=1))
    log = Path(log_dir) / f"{name}.log"
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run([PY, str(ROOT / "scripts" / "run_many.py"), str(spec),
                            str(deadline or 0.0)], stdout=f, stderr=subprocess.STDOUT,
                           cwd=ROOT)
    for line in log.read_text().splitlines():
        if line.startswith("[") and any(w in line for w in (" ok ", "FAIL", "skip",
                                                            "deadline")):
            print("  " + line, flush=True)
    print(f"=== stage {name} done in {(time.time()-t0)/60:.1f}m "
          f"(exit {p.returncode}, log {log}) ===", flush=True)
    return True


def mean_scores(run_root, score, n, tag):
    out = SCORES / f"{tag}.npy"
    if not out.exists():
        subprocess.run([PY, str(ROOT / "scripts" / "mean_scores.py"), "--runs", run_root,
                        "--score", score, "--n", str(n), "--out", str(SCORES), "--tag", tag],
                       check=True, cwd=ROOT)
    return str(out)


# ---------------------------------------------------------------------------------------
#  stages
# ---------------------------------------------------------------------------------------
def base(c, epochs=None, log_every=None):
    return dict(num_epochs=epochs or c.num_epochs, decay_epochs=c.decay_epochs,
                dataset=c.dataset, amp=c.amp, log_every_epochs=log_every or c.log_every,
                out_dir=str(RUNS), data_dir=c.data_dir)


def st_scores(c):
    """10 short runs whose only job is to produce E_w[EL2N] and E_w[GraNd]."""
    m = c.scoring_epochs * MIN_PER_EPOCH + len(GRAND_EPOCHS) * 0.7
    return [Job(f"scores/run{r}", m, run=r, el2n_epochs=EL2N_EPOCHS_SHORT,
                grand_epochs=GRAND_EPOCHS, **base(c, epochs=c.scoring_epochs, log_every=5))
            for r in range(c.n_score_runs)]


def st_full(c):
    """Full-data runs: the baseline, its seed spread, and the forgetting score."""
    return [Job(f"full/run{r}", c.num_epochs * MIN_PER_EPOCH + 0.5, run=r, track_forgetting=True,
                el2n_epochs=tuple(sorted({20, c.num_epochs // 2, c.num_epochs})),
                **base(c, log_every=1))
            for r in range(c.n_full_runs)]


def st_prune(c, paths, fracs, runs):
    jobs = []
    m = c.num_epochs * MIN_PER_EPOCH
    for r in runs:
        for frac in fracs:
            tag = f"keep{int(frac*100):03d}"
            jobs.append(Job(f"prune/random/{tag}/run{r}", m, run=r, subset="random",
                            subset_frac=frac, seed_offset=100, **base(c)))
            for meth in METHODS[1:]:
                jobs.append(Job(f"prune/{meth}/{tag}/run{r}", m, run=r,
                                subset="keep_max_scores", subset_frac=frac,
                                scores_path=paths[meth], seed_offset=100, **base(c)))
    return jobs


def st_score_epoch(c, keep=0.5):
    jobs, m = [], c.num_epochs * MIN_PER_EPOCH
    for ep in SCORE_EPOCH_SWEEP:
        p = mean_scores(str(RUNS / "scores"), f"el2n_ep{ep}", c.n_score_runs,
                        f"el2n_ep{ep}_mean{c.n_score_runs}")
        jobs.append(Job(f"score_epoch/el2n_ep{ep}/run0", m, run=0, subset="keep_max_scores",
                        subset_frac=keep, scores_path=p, seed_offset=200, **base(c)))
    return jobs


def st_window(c, path, size_frac=0.4, noise=0.0, prefix="offset"):
    jobs, m = [], c.num_epochs * MIN_PER_EPOCH
    size = int(round(size_frac * 50000))
    for off in OFFSETS:
        if off + size > 50000:
            continue
        jobs.append(Job(f"{prefix}/size{size}_off{off}/run0", m, run=0, subset="offset",
                        subset_size=size, subset_offset=off, scores_path=path,
                        random_label_fraction=noise, seed_offset=400, **base(c)))
    return jobs


def st_noise_scores(c):
    m = c.scoring_epochs * MIN_PER_EPOCH
    return [Job(f"noise10/scores/run{r}", m, run=r, random_label_fraction=0.1,
                el2n_epochs=EL2N_EPOCHS_SHORT,
                **base(c, epochs=c.scoring_epochs, log_every=5))
            for r in range(c.n_noise_runs)]


def st_noise_full(c):
    return [Job(f"noise10/full/run{r}", c.num_epochs * MIN_PER_EPOCH, run=r,
                random_label_fraction=0.1, track_forgetting=True,
                el2n_epochs=(20, c.num_epochs), **base(c, log_every=5))
            for r in range(c.n_noise_full)]


def st_noise_prune(c, path):
    jobs, m = [], c.num_epochs * MIN_PER_EPOCH
    for frac in NOISE_FRACS:
        tag = f"keep{int(frac*100):03d}"
        jobs.append(Job(f"noise10/prune/random/{tag}/run0", m, run=0, subset="random",
                        subset_frac=frac, random_label_fraction=0.1, seed_offset=500,
                        **base(c)))
        jobs.append(Job(f"noise10/prune/el2n_ep10/{tag}/run0", m, run=0,
                        subset="keep_max_scores", subset_frac=frac, scores_path=path,
                        random_label_fraction=0.1, seed_offset=500, **base(c)))
    return jobs


def st_verify(c, paths):
    """Does the shortened schedule change the conclusion? Same three points at 200 epochs,
    exactly the schedule of the paper."""
    e = c.verify_epochs
    kw = dict(num_epochs=e, decay_epochs=tuple(round(x * e / 200) for x in (60, 120, 160)),
              dataset=c.dataset, amp=c.amp, log_every_epochs=5, out_dir=str(RUNS),
              data_dir=c.data_dir)
    m = e * MIN_PER_EPOCH
    return [Job("verify200/full/run0", m, run=0, **kw),
            Job("verify200/random/keep050/run0", m, run=0, subset="random",
                subset_frac=0.5, seed_offset=100, **kw),
            Job("verify200/el2n_ep20/keep050/run0", m, run=0, subset="keep_max_scores",
                subset_frac=0.5, scores_path=paths["el2n_ep20"], seed_offset=100, **kw)]


def st_fp32(c, paths):
    """Does bf16 autocast change the conclusion? Two points in fp32."""
    kw = base(c)
    kw["amp"] = "fp32"
    m = c.num_epochs * MIN_PER_EPOCH * 2
    return [Job("fp32/full/run0", m, run=0, **kw),
            Job("fp32/el2n_ep20/keep050/run0", m, run=0, subset="keep_max_scores",
                subset_frac=0.5, scores_path=paths["el2n_ep20"], seed_offset=100, **kw)]


# verify200 runs right after the main grid: it is the control for the one deviation that
# could change the headline conclusion (100 epochs instead of the paper's 200).
ORDER = ["scores", "full", "prune", "verify", "score_epoch", "window", "noise", "fp32",
         "prune2"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default=",".join(ORDER))
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--decay_epochs", default="30,60,80")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--amp", default="bf16")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--n_score_runs", type=int, default=10)
    p.add_argument("--n_full_runs", type=int, default=6)
    p.add_argument("--n_noise_runs", type=int, default=5)
    p.add_argument("--n_noise_full", type=int, default=3)
    p.add_argument("--scoring_epochs", type=int, default=SCORING_EPOCHS)
    p.add_argument("--verify_epochs", type=int, default=200)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_root", default=str(ROOT / "results"))
    p.add_argument("--hours", type=float, default=0.0, help="stop starting new runs after")
    p.add_argument("--dry_run", action="store_true")
    c = p.parse_args()
    c.decay_epochs = tuple(int(x) for x in c.decay_epochs.split(","))
    global RUNS, SCORES
    RUNS = Path(c.out_root) / "runs"
    SCORES = Path(c.out_root) / "scores"
    log_dir = str(Path(c.out_root) / "logs")
    stages = c.stages.split(",")
    SCORES.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + c.hours * 3600 if c.hours else None

    if c.dry_run:
        fake = {m: "X" for m in METHODS[1:]}
        plan = [("scores", st_scores(c)), ("full", st_full(c)),
                ("prune", st_prune(c, fake, KEEP_FRACS, [0])),
                ("score_epoch", [Job(f"score_epoch/el2n_ep{e}/run0", c.num_epochs * MIN_PER_EPOCH)
                                 for e in SCORE_EPOCH_SWEEP]),
                ("window", st_window(c, "X")),
                ("verify", st_verify(c, fake)),
                ("noise", st_noise_scores(c) + st_noise_full(c) + st_noise_prune(c, "X")
                 + st_window(c, "X", noise=0.1, prefix="noise10/offset")),
                ("fp32", st_fp32(c, fake)),
                ("prune2", st_prune(c, fake, EXTRA_SEED_FRACS, [1]))]
        tot = 0.0
        for k, v in plan:
            m = sum(j.minutes for j in v)
            tot += m
            print(f"{k:12s} {len(v):4d} runs  ~{m:6.0f} min  ({m/60:4.1f} h)   cum {tot/60:4.1f} h")
        return

    if "scores" in stages:
        run_stage("scores", st_scores(c), log_dir, deadline)
    if "full" in stages:
        run_stage("full", st_full(c), log_dir, deadline)

    def all_means():
        """Mean scores for every snapshot that exists, for the distribution/correlation
        figures. EL2N at epoch 20 comes from the 10 scoring runs (more models); the later
        epochs only exist in the full-data runs."""
        for e in EL2N_EPOCHS_SHORT:
            if (RUNS / "scores" / "run0" / "scores" / f"el2n_ep{e}.npy").exists():
                mean_scores(str(RUNS / "scores"), f"el2n_ep{e}", c.n_score_runs,
                            f"el2n_ep{e}_mean{c.n_score_runs}")
        for e in GRAND_EPOCHS:
            if (RUNS / "scores" / "run0" / "scores" / f"grand_ep{e}.npy").exists():
                mean_scores(str(RUNS / "scores"), f"grand_ep{e}", c.n_score_runs,
                            f"grand_ep{e}_mean{c.n_score_runs}")
        for e in sorted({c.num_epochs // 2, c.num_epochs}):
            for kind in ("el2n", "forget"):
                if (RUNS / "full" / "run0" / "scores" / f"{kind}_ep{e}.npy").exists():
                    mean_scores(str(RUNS / "full"), f"{kind}_ep{e}", c.n_full_runs,
                                f"{kind}_ep{e}_mean{c.n_full_runs}")

    paths = {}
    if any(s in stages for s in ORDER[2:]):
        all_means()
        n, nf = c.n_score_runs, c.n_full_runs
        paths["el2n_ep20"] = mean_scores(str(RUNS / "scores"), "el2n_ep20", n,
                                         f"el2n_ep20_mean{n}")
        paths["el2n_ep10"] = mean_scores(str(RUNS / "scores"), "el2n_ep10", n,
                                         f"el2n_ep10_mean{n}")
        paths["grand_ep1"] = mean_scores(str(RUNS / "scores"), "grand_ep1", n,
                                         f"grand_ep1_mean{n}")
        paths["grand_ep20"] = mean_scores(str(RUNS / "scores"), "grand_ep20", n,
                                          f"grand_ep20_mean{n}")
        paths["forget_full"] = mean_scores(str(RUNS / "full"), f"forget_ep{c.num_epochs}", nf,
                                           f"forget_ep{c.num_epochs}_mean{nf}")
        with open(SCORES / "index.json", "w") as f:
            json.dump(paths, f, indent=2)

    def guarded(label, fn):
        """A stage must never take the rest of the night down with it."""
        try:
            fn()
        except Exception:
            import traceback
            print(f"!! stage {label} failed, continuing with the next one", flush=True)
            traceback.print_exc()

    if "prune" in stages:
        guarded("prune", lambda: run_stage("prune", st_prune(c, paths, KEEP_FRACS, [0]),
                                           log_dir, deadline))
    if "verify" in stages:
        guarded("verify", lambda: run_stage("verify200", st_verify(c, paths), log_dir,
                                            deadline))
    if "score_epoch" in stages:
        guarded("score_epoch", lambda: run_stage("score_epoch", st_score_epoch(c), log_dir,
                                                 deadline))
    if "window" in stages:
        guarded("window", lambda: run_stage("window", st_window(c, paths["el2n_ep10"]),
                                            log_dir, deadline))
    if "noise" in stages:
        def _noise():
            run_stage("noise_scores", st_noise_scores(c), log_dir, deadline)
            np10 = mean_scores(str(RUNS / "noise10" / "scores"), "el2n_ep10",
                               c.n_noise_runs, f"noise10_el2n_ep10_mean{c.n_noise_runs}")
            run_stage("noise_full", st_noise_full(c), log_dir, deadline)
            run_stage("noise_prune", st_noise_prune(c, np10), log_dir, deadline)
            run_stage("noise_window", st_window(c, np10, noise=0.1,
                                                prefix="noise10/offset"), log_dir, deadline)
        guarded("noise", _noise)
    if "fp32" in stages:
        guarded("fp32", lambda: run_stage("fp32", st_fp32(c, paths), log_dir, deadline))
    if "prune2" in stages:
        guarded("prune2", lambda: run_stage("prune2", st_prune(c, paths, EXTRA_SEED_FRACS,
                                                               [1]), log_dir, deadline))
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
