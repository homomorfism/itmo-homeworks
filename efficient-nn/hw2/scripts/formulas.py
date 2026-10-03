#!/usr/bin/env python
"""Render the paper's definitions as PNGs for the slides (matplotlib mathtext)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plotstyle as ps  # noqa: E402

OUT = ROOT / "results" / "figures" / "formulas"

FORMULAS = {
    "grand": (r"$\chi_t(x,y)\ =\ \mathbb{E}_{w_t}\ "
              r"\| \nabla_{w_t}\, \ell ( p(w_t,x),\, y ) \|_2$", 10.2),
    "el2n": (r"$\mathrm{EL2N}_t(x,y)\ =\ \mathbb{E}_{w_t}\ "
             r"\| p(w_t,x) - y \|_2$", 8.6),
    "lemma": (r"$\nabla_{w_t} \ell\ =\ \sum_{k=1}^{K}\ "
              r"\nabla_{f^{(k)}} \ell ( f_t(x), y )^{T}\ \psi_t^{(k)}(x)$", 10.6),
    "logit_grad": (r"$\nabla_{f}\, \ell ( f_t(x), y )\ =\ p(w_t,x) - y$", 8.2),
}


def main():
    ps.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    for name, (tex, w) in FORMULAS.items():
        fig = plt.figure(figsize=(w, 1.5))
        fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=30, color=ps.INK)
        fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)
        print(f"  -> {OUT/name}.png")


if __name__ == "__main__":
    main()
