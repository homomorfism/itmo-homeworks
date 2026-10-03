#!/usr/bin/env python
"""Walk results/runs and build results/summary.csv (one row per training run).

Run names, and the axes each one carries:

    scores/run{r}                              a short scoring run
    full/run{r}                                full-data baseline
    prune/{method}/keep{NNN}/run{r}            the main pruning grid
    score_epoch/el2n_ep{E}/run{r}              score computed at epoch E, 50% kept
    offset/size{S}_off{O}/run{r}               sliding window over the ranking
    noise10/...                                the same, with 10% of the labels permuted
    verify200/{full|method/keepNNN}/run{r}     control: the paper's 200-epoch schedule
    fp32/{full|method/keepNNN}/run{r}          control: fp32 instead of bf16 autocast
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
PREFIXES = ("noise10", "verify200", "fp32")


def parse(name: str, args: dict) -> dict:
    parts = name.split("/")
    prefix = parts.pop(0) if parts[0] in PREFIXES else None
    kind = parts[0]
    rec = dict(name=name, prefix=prefix or "main", method=None, keep_frac=None, offset=None,
               subset_size=None, score_epoch=None,
               noise=args.get("random_label_fraction", 0.0),
               run=args["run"], n_train=args["n_train"])

    if kind == "scores":
        kind, rec["method"] = "scores", "scoring_run"
    elif kind == "full":
        rec["method"] = "full"
    elif kind == "prune":
        rec["method"] = parts[1]
    elif kind == "score_epoch":
        rec["method"] = parts[1]
        rec["score_epoch"] = int(re.search(r"ep(\d+)", parts[1]).group(1))
    elif kind == "offset":
        rec["method"] = "el2n_window"
        rec["offset"] = args["subset_offset"]
        rec["subset_size"] = args["n_train"]
    else:
        # verify200/fp32 write the method straight under the prefix
        kind, rec["method"] = "prune", kind
    rec["kind"] = kind
    rec["group"] = kind if prefix is None else f"{prefix}_{kind}"
    rec["keep_frac"] = round(args["n_train"] / 50000, 4)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--out", default=str(ROOT / "results" / "summary.csv"))
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in sorted(runs.rglob("summary.json")):
        d = s.parent
        args = json.loads((d / "args.json").read_text())
        summ = json.loads(s.read_text())
        rec = parse(str(d.relative_to(runs)), args)
        rec.update(final_test_acc=summ["final_test_acc"], best_test_acc=summ["best_test_acc"],
                   final_train_acc=summ["final_train_acc"], wall_minutes=summ["wall_minutes"],
                   num_epochs=args["num_epochs"], amp=args["amp"])
        rows.append(rec)
    if not rows:
        print("no finished runs found", file=sys.stderr)
        return
    df = pd.DataFrame(rows).sort_values(["prefix", "kind", "method", "keep_frac", "offset",
                                         "run"])
    df.to_csv(out, index=False)
    print(f"{len(df)} runs -> {out}  ({df.wall_minutes.sum()/60:.1f} GPU-hours)")
    print(df.groupby(["group", "method"]).agg(
        n=("final_test_acc", "size"), acc=("final_test_acc", "mean")).to_string())


if __name__ == "__main__":
    main()
