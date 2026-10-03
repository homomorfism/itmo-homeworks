#!/usr/bin/env python
"""Build every figure and table of the report from results/summary.csv + results/scores."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import plotstyle as ps  # noqa: E402

FIGS = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
SCORES = ROOT / "results" / "scores"
RUNS = ROOT / "results" / "runs"


# ---------------------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------------------
def score_path(stem: str) -> Path | None:
    """results/scores/<stem>_mean<N>.npy, whatever N the sweep used."""
    hits = sorted(SCORES.glob(f"{stem}_mean*.npy"))
    return hits[-1] if hits else None


def load_score(stem: str) -> np.ndarray | None:
    p = score_path(stem)
    return np.load(p) if p else None


def n_models_of(stem: str) -> int | None:
    p = score_path(stem)
    return int(p.stem.rsplit("mean", 1)[1]) if p else None


def forget_stem() -> str | None:
    """The forgetting score from the LONGEST run (it is only meaningful at the end)."""
    import re
    hits = [(int(re.search(r"forget_ep(\d+)_", h.name).group(1)), h)
            for h in SCORES.glob("forget_ep*_mean*.npy")]
    return max(hits)[1].stem.rsplit("_mean", 1)[0] if hits else None


def agg(df, by):
    """mean and 16th/84th percentile across retraining seeds (as in the paper)."""
    g = df.groupby(by)["final_test_acc"]
    out = g.agg(mean="mean", n="size").reset_index()
    out["lo"], out["hi"] = g.quantile(0.16).values, g.quantile(0.84).values
    return out


def baseline(df, group="full"):
    f = df[df.group == group]["final_test_acc"]
    if not len(f):
        return None
    return f.mean(), f.quantile(0.16), f.quantile(0.84), len(f), f.std()


def spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[m], b[m]).statistic)


def to_md(df: pd.DataFrame, fmt: str = "{:.4f}") -> str:
    """Markdown table without pulling in `tabulate`."""
    def cell(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else fmt.format(v)
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    cols = [str(c) for c in df.columns]
    rows = [[cell(v) for v in r] for r in df.itertuples(index=False)]
    w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
         for i, c in enumerate(cols)]
    out = ["| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
           "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
    out += ["| " + " | ".join(r[i].rjust(w[i]) for i in range(len(cols))) + " |"
            for r in rows]
    return "\n".join(out) + "\n"


def save_table(name, df):
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.md").write_text(to_md(df))
    df.to_csv(TABLES / f"{name}.csv", index=False)
    print(f"  -> tables/{name}.md")


# ---------------------------------------------------------------------------------------
#  1 — the main pruning curve
# ---------------------------------------------------------------------------------------
METHODS = ["random", "el2n_ep20", "grand_ep1", "forget_full"]


def fig_prune(df):
    d = df[df.group == "prune"]
    if d.empty:
        return
    bm, blo, bhi, bn, bsd = baseline(df)
    methods = [m for m in METHODS if m in set(d.method)]
    nseed = int(d.groupby(["method", "keep_frac"]).size().min())

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    for ax, zoom in zip(axes, (False, True)):
        ax.axhspan(blo, bhi, color=ps.INK3, alpha=0.12, lw=0)
        ax.axhline(bm, color=ps.INK2, lw=1.2, ls=(0, (4, 3)))
        for m in methods:
            c, mk, lab = ps.METHOD_STYLE[m]
            a = agg(d[d.method == m], "keep_frac").sort_values("keep_frac")
            x = a.keep_frac * 100
            if a.n.max() > 1:
                ps.band(ax, x, a.lo, a.hi, c)
            ax.plot(x, a["mean"], color=c, marker=mk, label=lab, mec=ps.SURFACE, mew=0.8)
        if not zoom:
            ps.direct_label(ax, 10, bm, f"все данные {bm:.4f}", ps.INK2, dx=0, dy=9)
            ax.set_ylabel("итоговая test accuracy")
            ps.title(ax, "Прунинг обучающих данных CIFAR-10",
                     f"ResNet-18; скор усреднён по {n_models_of('el2n_ep20')} моделям")
            ax.legend(loc="lower right")
        else:
            sub = d[d.keep_frac >= 0.4]
            ax.set_xlim(37, 93)
            lo = min(bm - 0.015, sub.groupby(["method", "keep_frac"]).final_test_acc.mean().min())
            ax.set_ylim(lo - 0.003, max(bhi, sub.final_test_acc.max()) + 0.003)
            labels = []
            for m in methods:
                c, _, lab = ps.METHOD_STYLE[m]
                a = agg(d[d.method == m], "keep_frac").sort_values("keep_frac")
                a = a[a.keep_frac >= 0.4]
                labels.append((a["mean"].iloc[-1], lab.split(" (")[0], c))
            ps.spread_labels(ax, labels)
            ps.title(ax, "Увеличенный фрагмент",
                     f"пунктир — все данные, {bn} запусков, sd {bsd*100:.2f} п.п.")
        ax.set_xlabel("оставлено данных, %")
    fig.subplots_adjust(wspace=0.3)
    ps.finish(fig, FIGS / "01_prune_cifar10.png")

    t = agg(d, ["method", "keep_frac"]).pivot(index="keep_frac", columns="method",
                                              values="mean").reset_index()
    t.insert(0, "оставлено, %", (t.pop("keep_frac") * 100).round().astype(int))
    t["все данные"] = bm
    save_table("prune_cifar10", t)
    return nseed


# ---------------------------------------------------------------------------------------
#  2 — how early can the score be computed
# ---------------------------------------------------------------------------------------
def fig_score_epoch(df):
    d = df[df.group == "score_epoch"]
    if d.empty:
        return
    bm, blo, bhi, _, _ = baseline(df)
    keep = d.keep_frac.iloc[0]
    rnd = df[(df.group == "prune") & (df.method == "random") & (df.keep_frac == keep)]
    a = agg(d, "score_epoch").sort_values("score_epoch")
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    ax.axhspan(blo, bhi, color=ps.INK3, alpha=0.12, lw=0)
    ax.axhline(bm, color=ps.INK2, lw=1.2, ls=(0, (4, 3)))
    ps.direct_label(ax, a.score_epoch.iloc[0], bm, "все данные", ps.INK2, dx=0, dy=8)
    if not rnd.empty:
        rm = rnd.final_test_acc.mean()
        ax.axhline(rm, color=ps.BLUE, lw=1.2, ls=(0, (1, 2)))
        ps.direct_label(ax, a.score_epoch.iloc[0], rm, f"случайные {int(keep*100)}%",
                        ps.BLUE, dx=0, dy=-11)
    ax.plot(a.score_epoch, a["mean"], color=ps.ORANGE, marker="s", mec=ps.SURFACE, mew=0.8,
            label=f"EL2N, {int(keep*100)}% с наибольшим скором")
    ax.set_xscale("log")
    ax.set_xticks(list(a.score_epoch))
    ax.set_xticklabels([str(int(e)) for e in a.score_epoch])
    ax.minorticks_off()
    ax.set_xlabel("эпоха, на которой посчитан EL2N")
    ax.set_ylabel("итоговая test accuracy")
    ps.title(ax, "Важные примеры видны уже в начале обучения",
             f"подмножество {int(keep*100)}% CIFAR-10, скор усреднён по "
             f"{n_models_of('el2n_ep20')} моделям")
    ax.legend(loc="lower right")
    ps.finish(fig, FIGS / "03_score_epoch.png")
    save_table("score_epoch", a.rename(columns={"score_epoch": "эпоха EL2N",
                                                "mean": "test acc"}))


# ---------------------------------------------------------------------------------------
#  3 — averaging over initialisations
# ---------------------------------------------------------------------------------------
def fig_n_models():
    ref = load_score("el2n_ep20")
    if ref is None:
        return
    n_ref = n_models_of("el2n_ep20")
    singles = []
    for r in range(n_ref):
        p = RUNS / "scores" / f"run{r}" / "scores" / "el2n_ep20.npy"
        if p.exists():
            singles.append(np.load(p))
    if len(singles) < 2:
        return
    S = np.stack(singles)
    # correlation of a k-model average with the k-model average of the other models
    rows = []
    rng = np.random.RandomState(0)
    for k in range(1, len(singles) + 1):
        cs = []
        for _ in range(20):
            idx = rng.permutation(len(singles))
            a = S[idx[:k]].mean(0)
            cs.append(spearman(a, ref))
        rows.append((k, float(np.mean(cs)), float(np.std(cs))))
    pair = [spearman(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S))]

    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    k, m, sd = map(np.array, zip(*rows))
    ps.band(ax, k, m - sd, m + sd, ps.AQUA)
    ax.plot(k, m, color=ps.AQUA, marker="^", mec=ps.SURFACE, mew=0.8,
            label=f"среднее по k моделям vs эталон ({n_ref} моделей)")
    ax.axhline(float(np.mean(pair)), color=ps.VIOLET, lw=1.2, ls=(0, (1, 2)))
    ps.direct_label(ax, 1, float(np.mean(pair)), "две отдельные модели между собой",
                    ps.VIOLET, dx=0, dy=-11)
    ax.set_xticks(list(k))
    ax.set_xlabel("число моделей k, по которым усреднён EL2N")
    ax.set_ylabel("корреляция Спирмена")
    ps.title(ax, "Один скор — это в основном шум инициализации",
             "EL2N на эпохе 20, CIFAR-10")
    ax.legend(loc="lower right")
    ps.finish(fig, FIGS / "04_n_models.png")
    save_table("score_averaging", pd.DataFrame(
        {"k моделей": k, "Спирмен с эталоном": m, "sd": sd}))


# ---------------------------------------------------------------------------------------
#  4 — sliding window over the ranking
# ---------------------------------------------------------------------------------------
def fig_window(df):
    d = df[df.group.isin(["offset", "noise10_offset"])]
    if d.empty:
        return
    panels = [("offset", "чистые метки", ps.ORANGE, "s", "full"),
              ("noise10_offset", "10% перемешанных меток", ps.VIOLET, "D", "noise10_full")]
    panels = [p for p in panels if not d[d.group == p[0]].empty]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 4.1), squeeze=False)
    for ax, (g, lab, c, mk, bgroup) in zip(axes[0], panels):
        dd = d[d.group == g]
        a = agg(dd, "offset").sort_values("offset")
        size = int(dd.subset_size.iloc[0])
        b = baseline(df, bgroup)
        if b:
            ax.axhline(b[0], color=ps.INK2, lw=1.2, ls=(0, (4, 3)))
            ps.direct_label(ax, a.offset.iloc[0], b[0], "все данные", ps.INK2, dx=0, dy=8)
        if a.n.max() > 1:
            ps.band(ax, a.offset, a.lo, a.hi, c)
        ax.plot(a.offset, a["mean"], color=c, marker=mk, mec=ps.SURFACE, mew=0.8)
        best = a.loc[a["mean"].idxmax()]
        ax.scatter([best.offset], [best["mean"]], s=90, facecolor="none", edgecolor=c, lw=1.6,
                   zorder=4)
        ps.direct_label(ax, best.offset, best["mean"],
                        f"оптимум: выбросить {int(best.offset)}", c, dx=8, dy=9)
        ax.set_xscale("symlog", linthresh=250)
        ax.set_xticks(list(a.offset))
        ax.set_xticklabels([str(int(o)) for o in a.offset], rotation=45)
        ax.minorticks_off()
        ax.set_xlabel("выброшено примеров с наибольшим EL2N")
        ps.title(ax, lab, f"окно из {size} примеров ({size/500:.0f}% CIFAR-10), EL2N эпоха 10")
        save_table(f"window_{g}", a)
    axes[0][0].set_ylabel("итоговая test accuracy")
    ps.finish(fig, FIGS / "05_window.png")


# ---------------------------------------------------------------------------------------
#  5 — pruning under label noise
# ---------------------------------------------------------------------------------------
def fig_noise_prune(df):
    d = df[df.group == "noise10_prune"]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    b = baseline(df, "noise10_full")
    if b:
        ax.axhspan(b[1], b[2], color=ps.INK3, alpha=0.12, lw=0)
        ax.axhline(b[0], color=ps.INK2, lw=1.2, ls=(0, (4, 3)))
        ps.direct_label(ax, 20, b[0], "все (шумные) данные", ps.INK2, dx=0, dy=8)
    bc = baseline(df, "full")
    if bc:
        ax.axhline(bc[0], color=ps.AQUA, lw=1.2, ls=(0, (1, 2)))
        ps.direct_label(ax, 20, bc[0], "для сравнения: чистые данные", ps.AQUA, dx=0, dy=8)
    for m in ("random", "el2n_ep10", "el2n_ep20"):
        if m not in set(d.method):
            continue
        c, mk, lab = ps.METHOD_STYLE[m]
        a = agg(d[d.method == m], "keep_frac").sort_values("keep_frac")
        if a.n.max() > 1:
            ps.band(ax, a.keep_frac * 100, a.lo, a.hi, c)
        ax.plot(a.keep_frac * 100, a["mean"], color=c, marker=mk, label=lab,
                mec=ps.SURFACE, mew=0.8)
    ax.set_xlabel("оставлено данных, %")
    ax.set_ylabel("итоговая test accuracy")
    ps.title(ax, "Прунинг при 10% шума в метках",
             "«оставить самые сложные» ломается: наверху рейтинга — битые метки")
    ax.legend(loc="lower right")
    ps.finish(fig, FIGS / "06_noise_prune.png")
    save_table("noise_prune", agg(d, ["method", "keep_frac"]))


# ---------------------------------------------------------------------------------------
#  6 — score distributions
# ---------------------------------------------------------------------------------------
def fig_score_dists():
    eps = [e for e in (1, 2, 5, 10, 20, 50, 100, 200) if score_path(f"el2n_ep{e}")]
    if len(eps) < 2:
        return
    fig, axes = plt.subplots(1, len(eps), figsize=(1.95 * len(eps), 2.7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, eps):
        s = load_score(f"el2n_ep{e}")
        ax.hist(s, bins=60, color=ps.ORANGE, alpha=0.85, lw=0)
        ax.set_title(f"эпоха {e}", loc="left", fontsize=9, color=ps.INK)
        ax.set_xlabel("EL2N")
        ax.set_xlim(0, 1.45)
    axes[0].set_ylabel("число примеров")
    fig.suptitle(f"Распределение EL2N (усреднён по {n_models_of('el2n_ep20')} моделям) "
                 f"по ходу обучения", x=0.0, ha="left", fontsize=10.5,
                 fontweight="bold", color=ps.INK)
    fig.tight_layout()
    ps.finish(fig, FIGS / "07_score_distributions.png")


# ---------------------------------------------------------------------------------------
#  7 — do the scores agree with each other
# ---------------------------------------------------------------------------------------
def fig_score_corr():
    fstem = forget_stem()
    eps = [e for e in (1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100, 200)
           if score_path(f"el2n_ep{e}")]
    if not fstem or len(eps) < 2:
        return
    ref = load_score(fstem)
    corr = [spearman(load_score(f"el2n_ep{e}"), ref) for e in eps]

    names, mats = [], []
    for stem, lab in [("el2n_ep1", "EL2N@1"), ("el2n_ep5", "EL2N@5"), ("el2n_ep10", "EL2N@10"),
                      ("el2n_ep20", "EL2N@20"), ("grand_ep1", "GraNd@1"),
                      ("grand_ep10", "GraNd@10"), ("grand_ep20", "GraNd@20"),
                      (fstem, "forget")]:
        s = load_score(stem)
        if s is not None:
            names.append(lab)
            mats.append(s)
    M = np.array([[spearman(a, b) for b in mats] for a in mats])

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    ax = axes[0]
    ax.plot(eps, corr, color=ps.ORANGE, marker="s", mec=ps.SURFACE, mew=0.8)
    ax.set_xscale("log")
    ax.set_xticks(eps)
    ax.set_xticklabels([str(e) for e in eps], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlabel("эпоха, на которой посчитан EL2N")
    ax.set_ylabel("корреляция Спирмена с forgetting")
    ps.title(ax, "EL2N быстро догоняет forgetting",
             "forgetting считается за всё обучение целиком")

    ax = axes[1]
    e20 = load_score("el2n_ep20")
    m = np.isfinite(ref)
    hb = ax.hexbin(e20[m], ref[m], gridsize=45, cmap="Blues", bins="log", mincnt=1, lw=0)
    fig.colorbar(hb, ax=ax, label="примеров (log)")
    ax.grid(False)
    ax.set_xlabel("EL2N, эпоха 20")
    ax.set_ylabel("число забываний за обучение")
    ps.title(ax, f"Спирмен = {spearman(e20, ref):.2f}", "точка — пример CIFAR-10")

    ax = axes[2]
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color=ps.SURFACE if M[i, j] > 0.6 else ps.INK)
    fig.colorbar(im, ax=ax, label="Спирмен")
    ps.title(ax, "Все скоры ранжируют почти одинаково")
    ps.finish(fig, FIGS / "08_score_correlation.png")
    save_table("el2n_vs_forget", pd.DataFrame({"эпоха EL2N": eps,
                                               "Спирмен с forgetting": corr}))
    save_table("score_agreement", pd.DataFrame(M, index=names, columns=names).reset_index(
        names="скор"))


# ---------------------------------------------------------------------------------------
#  8 — noisy-label detection
# ---------------------------------------------------------------------------------------
def fig_noise_detection():
    hits = (sorted(SCORES.glob("noise10_el2n_ep10_mean*.npy"))
            or sorted(SCORES.glob("noise10_el2n_ep*_mean*.npy")))
    if not hits:
        return
    s = np.load(hits[-1])
    from data_diet.data import load_raw, randomize_labels
    _, y, _, _ = load_raw("cifar10", str(ROOT / "data"))
    noisy = randomize_labels(y, 0.1, 7777) != y
    order = np.argsort(-s)
    k = np.arange(1, len(s) + 1)
    cum = np.cumsum(noisy[order])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    ax.plot(k, cum / k, color=ps.VIOLET, lw=1.6)
    ax.axhline(noisy.mean(), color=ps.INK2, lw=1.2, ls=(0, (4, 3)))
    ps.direct_label(ax, 1500, noisy.mean(), f"базовая частота {noisy.mean():.3f}", ps.INK2,
                    dx=0, dy=9)
    ax.set_xscale("log")
    ax.set_xlabel("k — взято примеров с наибольшим EL2N")
    ax.set_ylabel("доля битых меток среди них")
    ps.title(ax, "EL2N как детектор испорченных меток", "CIFAR-10, 10% меток перемешано")
    ax = axes[1]
    ax.hist(s[~noisy], bins=70, color=ps.AQUA, alpha=0.75, lw=0, label="метка верная",
            density=True)
    ax.hist(s[noisy], bins=70, color=ps.VIOLET, alpha=0.65, lw=0, label="метка испорчена",
            density=True)
    ax.set_xlabel("EL2N, эпоха 10")
    ax.set_ylabel("плотность")
    ps.title(ax, "Испорченные примеры живут на правом хвосте")
    ax.legend(loc="upper left")
    ps.finish(fig, FIGS / "09_noise_detection.png")
    save_table("noise_detection", pd.DataFrame(
        [dict(k=kk, precision=float(cum[kk - 1] / kk),
              recall=float(cum[kk - 1] / noisy.sum())) for kk in
         (1000, 2500, 4500, 5000, 7500, 10000)]))


# ---------------------------------------------------------------------------------------
#  9 — the images themselves
# ---------------------------------------------------------------------------------------
def fig_examples():
    s = load_score("el2n_ep20")
    if s is None:
        return
    from data_diet.data import load_raw
    X, y, _, _ = load_raw("cifar10", str(ROOT / "data"))
    names = ["самолёт", "авто", "птица", "кошка", "олень", "собака", "жаба", "конь",
             "корабль", "грузовик"]
    ncol = 8
    fig, axes = plt.subplots(10, 2 * ncol + 1, figsize=(10.6, 6.0))
    for c in range(10):
        idx = np.where(y == c)[0]
        o = idx[np.argsort(s[idx])]
        for j in range(ncol):
            for col, im in ((j, X[o[j]]), (ncol + 1 + j, X[o[-(j + 1)]])):
                ax = axes[c, col]
                ax.imshow(im)
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
        axes[c, ncol].axis("off")
        axes[c, 0].set_ylabel(names[c], rotation=0, ha="right", va="center", fontsize=8,
                              color=ps.INK2, labelpad=8)
    axes[0, ncol // 2].set_title("наименьший EL2N (эпоха 20)", fontsize=10, color=ps.INK)
    axes[0, ncol + 1 + ncol // 2].set_title("наибольший EL2N (эпоха 20)", fontsize=10,
                                            color=ps.INK)
    fig.subplots_adjust(wspace=0.06, hspace=0.06)
    ps.finish(fig, FIGS / "10_examples.png")


# ---------------------------------------------------------------------------------------
#  10 — training curves
# ---------------------------------------------------------------------------------------
def fig_curves():
    picks = [("full/run0", "все данные", ps.INK2, (0, (4, 3))),
             ("prune/random/keep050/run0", "случайные 50%", ps.BLUE, "-"),
             ("prune/el2n_ep20/keep050/run0", "EL2N 50%", ps.ORANGE, "-"),
             ("prune/random/keep020/run0", "случайные 20%", ps.AQUA, "-"),
             ("prune/el2n_ep20/keep020/run0", "EL2N 20%", ps.VIOLET, "-")]
    have = [(p, l, c, s) for p, l, c, s in picks if (RUNS / p / "metrics.csv").exists()]
    if len(have) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for p, lab, c, st in have:
        m = pd.read_csv(RUNS / p / "metrics.csv")
        axes[0].plot(m.epoch, m.test_acc, color=c, lw=1.4, ls=st, label=lab)
        axes[1].plot(m.epoch, m.train_acc, color=c, lw=1.4, ls=st, label=lab)
    for ax, t, yl in ((axes[0], "Test accuracy", "test accuracy"),
                      (axes[1], "Train accuracy (на своём подмножестве)", "train accuracy")):
        ax.set_xlabel("номинальная эпоха (шаг / 390)")
        ax.set_ylabel(yl)
        ps.title(ax, t, "у всех запусков одинаковое число шагов и одно lr-расписание")
    axes[0].set_ylim(0.4, 0.97)
    axes[0].legend(loc="lower right")
    ps.finish(fig, FIGS / "11_training_curves.png")


# ---------------------------------------------------------------------------------------
#  11 — the two protocol controls
# ---------------------------------------------------------------------------------------
def fig_controls(df):
    rows = []
    for prefix, lab in (("main", "наш протокол (100 эпох, bf16)"),
                        ("verify200", "200 эпох, как в статье"),
                        ("fp32", "100 эпох, fp32")):
        sub = df[df.prefix == prefix]
        if sub.empty:
            continue
        full = sub[sub.kind == "full"]["final_test_acc"]
        pr = sub[(sub.kind == "prune") & (sub.keep_frac == 0.5)]
        for meth in ("full", "random", "el2n_ep20"):
            v = full if meth == "full" else pr[pr.method == meth]["final_test_acc"]
            if len(v):
                rows.append(dict(protocol=lab, method=meth, acc=v.mean(), n=len(v)))
    if len({r["protocol"] for r in rows}) < 2:
        return
    t = pd.DataFrame(rows)
    piv = t.pivot(index="protocol", columns="method", values="acc")
    order = [p for p in ("наш протокол (100 эпох, bf16)", "200 эпох, как в статье",
                         "100 эпох, fp32") if p in piv.index]
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    w, xs = 0.26, np.arange(len(piv))
    for i, (meth, lab) in enumerate((("full", "все данные"), ("random", "случайные 50%"),
                                     ("el2n_ep20", "EL2N 50%"))):
        if meth not in piv:
            continue
        c = ps.METHOD_STYLE[meth if meth != "full" else "full"][0]
        v = piv[meth].values
        ax.bar(xs + (i - 1) * w, v, w * 0.92, color=c, label=lab)
        for x, y in zip(xs + (i - 1) * w, v):
            if np.isfinite(y):
                ax.text(x, y + 0.0015, f"{y:.4f}", ha="center", fontsize=7.5, color=ps.INK2)
    ax.set_xticks(xs, piv.index, fontsize=8.5)
    ax.set_ylim(min(0.9, float(np.nanmin(piv.values)) - 0.01), float(np.nanmax(piv.values)) + 0.008)
    ax.set_ylabel("итоговая test accuracy")
    ps.title(ax, "Контроль: вывод не зависит от сокращённого расписания и bf16",
             "одно и то же сравнение при разных протоколах обучения")
    ax.legend(loc="lower right", ncols=3)
    ps.finish(fig, FIGS / "12_controls.png")
    save_table("controls", piv.reset_index())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default=str(ROOT / "results"))
    a = ap.parse_args()
    global FIGS, TABLES, SCORES, RUNS
    root = Path(a.results_root)
    FIGS, TABLES, SCORES, RUNS = (root / "figures", root / "tables", root / "scores",
                                  root / "runs")
    ps.setup()
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    csv = root / "summary.csv"
    df = pd.read_csv(csv) if csv.exists() else pd.DataFrame(
        columns=["group", "method", "prefix", "kind", "keep_frac"])
    jobs = [("prune", lambda: fig_prune(df)), ("score_epoch", lambda: fig_score_epoch(df)),
            ("n_models", fig_n_models), ("window", lambda: fig_window(df)),
            ("noise_prune", lambda: fig_noise_prune(df)), ("score_dists", fig_score_dists),
            ("score_corr", fig_score_corr), ("noise_detect", fig_noise_detection),
            ("examples", fig_examples), ("curves", fig_curves),
            ("controls", lambda: fig_controls(df))]
    for name, f in jobs:
        try:
            f()
        except Exception as e:
            print(f"  !! {name}: {type(e).__name__}: {e}")
    print("figures done")


if __name__ == "__main__":
    main()
