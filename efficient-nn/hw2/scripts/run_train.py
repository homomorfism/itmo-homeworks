#!/usr/bin/env python
"""Train one run. Every field of data_diet.train.Args is exposed as --flag."""
import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_diet.train import Args, train  # noqa: E402


def intlist(s):
    return tuple(int(x) for x in s.split(",") if x != "")


def main():
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(Args):
        default = f.default
        if f.name in ("el2n_epochs", "grand_epochs"):
            p.add_argument(f"--{f.name}", type=intlist, default=default)
        elif f.name == "decay_epochs":
            p.add_argument(f"--{f.name}", type=intlist, default=default)
        elif isinstance(default, bool):
            p.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=default)
        elif f.name in ("subset", "scores_path"):
            p.add_argument(f"--{f.name}", type=str, default=default)
        elif f.name in ("subset_size", "subset_offset"):
            p.add_argument(f"--{f.name}", type=int, default=default)
        elif f.name == "subset_frac":
            p.add_argument(f"--{f.name}", type=float, default=default)
        else:
            p.add_argument(f"--{f.name}", type=type(default), default=default)
    a = p.parse_args()
    args = Args(**vars(a))
    train(args)


if __name__ == "__main__":
    main()
