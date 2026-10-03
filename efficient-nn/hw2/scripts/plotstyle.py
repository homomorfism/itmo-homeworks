"""Shared plotting style.

Palette: the validated reference categorical palette (slots blue / orange / aqua /
violet). Validated with the data-viz validator under `--pairs all`, light surface
#fcfcfb: worst all-pairs CVD dE 9.2 (deutan), worst normal-vision dE 16.3 -> all checks
pass. Aqua sits below 3:1 contrast on the light surface, so every series also carries a
distinct marker and a direct label at the end of its line (secondary encoding), and the
underlying numbers are written out to results/tables/*.md.
Figures are always rendered on the light surface and shown on a light card in the slides,
so no dark variant is needed.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#84837c"
GRID = "#e6e5e1"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
YELLOW, MAGENTA, RED = "#eda100", "#e87ba4", "#e34948"

# series slot -> (colour, marker)
METHOD_STYLE = {
    "random":       (BLUE,   "o", "Random"),
    "el2n_ep20":    (ORANGE, "s", "EL2N (эпоха 20)"),
    "el2n_ep10":    (ORANGE, "s", "EL2N (эпоха 10)"),
    "grand_ep1":    (AQUA,   "^", "GraNd (эпоха 1)"),
    "grand_ep20":   (YELLOW, "v", "GraNd (эпоха 20)"),
    "forget_full":  (VIOLET, "D", "Forgetting (всё обучение)"),
    "full":         (INK2,   "",  "Все данные"),
}


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 9.5,
        "font.family": "DejaVu Sans",
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK3,
        "ytick.color": INK3,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "text.color": INK,
    })


def title(ax, text, sub=None):
    ax.set_title(text, loc="left", color=INK, pad=20 if sub else 6)
    if sub:
        ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=8.5, color=INK3,
                va="bottom", ha="left")


def band(ax, x, lo, hi, color, alpha=0.14):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, lw=0)


def direct_label(ax, x, y, text, color, dx=4, dy=0, **kw):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                color=color, fontsize=8.5, va="center", fontweight="bold", **kw)


def spread_labels(ax, items, min_gap_frac=0.058, dx=6):
    """Place end-of-line labels so they never overlap.

    `items` is [(y, text, colour)]; labels are nudged apart by at least
    `min_gap_frac` of the y-range, keeping their original order.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    overflow = ys[-1] - (hi - gap * 0.4)
    if overflow > 0:
        ys = [y - overflow for y in ys]
    for (y0, text, c), y in zip(items, ys):
        x = ax.get_xlim()[1]
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, 0), color=c,
                    fontsize=8.5, va="center", fontweight="bold",
                    annotation_clip=False)


def finish(fig, path, note=None):
    if note:
        fig.text(0.0, -0.02, note, fontsize=7.5, color=INK3, ha="left", va="top")
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")
