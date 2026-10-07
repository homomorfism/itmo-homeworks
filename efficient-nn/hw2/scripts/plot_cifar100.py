#!/usr/bin/env python
"""CIFAR-100 pruning figure, same style as results/figures/01_prune_cifar10.png."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plotstyle as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "results" / "cifar100"
FIGS = OUT / "figures"
METHODS = ["random", "el2n_ep20", "grand_ep20", "grand_ep1", "forget_full"]


def main():
    ps.setup()
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT / "summary.csv")
    base = df[df.method == "full"].final_test_acc
    b, sd = base.mean() * 100, base.std() * 100

    pr = df[df.kind == "prune"].copy()
    pr["keep"] = (pr.keep_frac * 100).round().astype(int)
    tab = pr.pivot_table(index="keep", columns="method", values="final_test_acc") * 100

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = tab.index.values
    ax.axhspan(b - sd, b + sd, color=ps.INK2, alpha=0.10, lw=0)
    ax.axhline(b, color=ps.INK2, lw=1.2, ls="--")
    ps.direct_label(ax, x.max(), b, f"все данные {b:.1f}%", ps.INK2, dx=6)

    label_items = []
    for m in METHODS:
        if m not in tab:
            continue
        c, mk, lab = ps.METHOD_STYLE[m]
        y = tab[m].values
        ax.plot(x, y, color=c, marker=mk, label=lab)
        label_items.append((y[-1], lab.split(" (")[0], c))

    ax.set_xlabel("оставлено данных, %")
    ax.set_ylabel("test accuracy, %")
    ax.set_xticks(x)
    ps.title(ax, "Прунинг CIFAR-100 / ResNet-18",
             sub=f"база {b:.2f}% (6 запусков, sd {sd:.2f} п.п.); скоры усреднены по 10 моделям, "
                 f"forgetting — по 6; 100 эпох, bf16")
    ax.legend(loc="lower right", ncol=2)
    ps.finish(fig, FIGS / "prune_cifar100.png")


if __name__ == "__main__":
    main()
