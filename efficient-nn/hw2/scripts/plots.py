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


def fig_grand_epoch(df):
    """GraNd at epoch 1 vs epoch 20, on the fractions where both were run."""
    d = df[df.group == "prune"]
    if "grand_ep20" not in set(d.method):
        return
    fracs = sorted(set(d[d.method == "grand_ep20"].keep_frac))
    bm, blo, bhi, bn, bsd = baseline(df)
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.axhspan(blo, bhi, color=ps.INK3, alpha=0.12, lw=0)
    ax.axhline(bm, color=ps.INK2, lw=1.2, ls=(0, (4, 3)))
    ps.direct_label(ax, fracs[-1] * 100, bm, "все данные", ps.INK2, dx=-2, dy=8, ha="right")
    for m in ("random", "grand_ep1", "grand_ep20", "el2n_ep20"):
        c, mk, lab = ps.METHOD_STYLE[m]
        a = (d[(d.method == m) & (d.keep_frac.isin(fracs))]
             .groupby("keep_frac").final_test_acc.mean().reset_index())
        ax.plot(a.keep_frac * 100, a.final_test_acc, color=c, marker=mk, label=lab,
                mec=ps.SURFACE, mew=0.8)
    ax.set_xticks([int(f * 100) for f in fracs])
    ax.set_xlabel("оставлено данных, %")
    ax.set_ylabel("итоговая test accuracy")
    ps.title(ax, "Дело в эпохе, а не в скоре",
             "тот же GraNd, снятый на 1-й и на 20-й эпохе")
    ax.legend(loc="lower right")
    ps.finish(fig, FIGS / "02_grand_epoch.png")
    save_table("grand_epoch", d[d.keep_frac.isin(fracs)]
               .groupby(["keep_frac", "method"]).final_test_acc.mean().unstack().reset_index())


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
    pm = float(np.mean(pair))
    ax.axhline(pm, color=ps.VIOLET, lw=1.2, ls=(0, (1, 2)),
               label="две отдельные модели между собой")
    ps.direct_label(ax, 10, pm, f"{pm:.2f}", ps.VIOLET, dx=-4, dy=10, ha="right")
    ax.set_xticks(list(k))
    ax.set_ylim(pm - 0.07, 1.012)
    ax.set_xlabel("число моделей k, по которым усреднён EL2N")
    ax.set_ylabel("корреляция Спирмена")
    ps.title(ax, "Один скор — это в основном шум инициализации",
             "EL2N на эпохе 20; эталон — среднее по всем 10, поэтому при k=10 ровно 1")
    ax.legend(loc="lower right", framealpha=0)
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
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2), squeeze=False)
    for ax, (g, lab, c, mk, bgroup) in zip(axes[0], panels):
        dd = d[d.group == g]
        a = agg(dd, "offset").sort_values("offset").reset_index(drop=True)
        size = int(dd.subset_size.iloc[0])
        x = np.arange(len(a))                      # categorical: the offsets are a sweep,
        b = baseline(df, bgroup)                   # not a quantitative axis
        if b:
            ax.axhline(b[0], color=ps.INK2, lw=1.2, ls=(0, (4, 3)))
            ps.direct_label(ax, x[0], b[0], "все данные", ps.INK2, dx=2, dy=9, ha="left")
        if a.n.max() > 1:
            ps.band(ax, x, a.lo, a.hi, c)
        ax.plot(x, a["mean"], color=c, marker=mk, mec=ps.SURFACE, mew=0.8)
        best = int(a["mean"].idxmax())
        ax.scatter([x[best]], [a["mean"].iloc[best]], s=95, facecolor="none", edgecolor=c,
                   lw=1.7, zorder=4)
        ps.direct_label(ax, x[best], a["mean"].iloc[best],
                        f"лучшее окно:\nвыбросить {int(a.offset.iloc[best])}", c,
                        dx=0 if best < len(a) - 2 else -10, dy=-34,
                        ha="center" if best < len(a) - 2 else "right")
        ax.set_xticks(x, [str(int(o)) for o in a.offset], fontsize=8.5)
        ax.set_xlabel("выброшено примеров с наибольшим EL2N")
        ax.set_xlim(-0.4, len(a) - 0.6)
        ps.title(ax, lab, f"окно из {size} примеров ({size/500:.0f}% CIFAR-10), EL2N эпоха 10, "
                          f"{int(a.n.max())} сид(ов)")
        save_table(f"window_{g}", a.drop(columns=["n"]) if a.n.max() == 1 else a)
    axes[0][0].set_ylabel("итоговая test accuracy")
    fig.subplots_adjust(wspace=0.22)
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
    eps = [e for e in (1, 5, 10, 20, 50, 100, 200) if score_path(f"el2n_ep{e}")][:5]
    if len(eps) < 2:
        return
    fig, axes = plt.subplots(1, len(eps), figsize=(2.5 * len(eps), 3.1), sharey=True,
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, eps):
        s = load_score(f"el2n_ep{e}")
        ax.hist(s, bins=70, color=ps.ORANGE, alpha=0.9, lw=0)
        frac0 = float((s < 0.05).mean())
        ax.set_title(f"эпоха {e}", loc="left", fontsize=9.5, color=ps.INK)
        ax.text(0.97, 0.93, f"{frac0*100:.0f}% почти\nвыучено", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=ps.INK3)
        ax.set_xlabel("EL2N")
        ax.set_xlim(0, 1.45)
        ax.set_yscale("log")
        ax.set_ylim(1, 6e4)
    axes[0].set_ylabel("число примеров (log)")
    fig.suptitle(f"Распределение EL2N (среднее по {n_models_of('el2n_ep20')} моделям): "
                 f"масса уезжает к нулю, хвост остаётся",
                 x=0.0, ha="left", fontsize=10.5, fontweight="semibold", color=ps.INK)
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

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    ax = axes[0]
    ax.plot(eps, corr, color=ps.ORANGE, marker="s", mec=ps.SURFACE, mew=0.8)
    ax.set_xscale("log")
    ax.set_xticks(eps)
    ax.set_xticklabels([str(e) for e in eps], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlabel("эпоха, на которой посчитан EL2N")
    ax.set_ylabel("корреляция Спирмена с forgetting")
    ps.title(ax, "Скор информативен рано — и выдыхается к концу",
             "согласие с forgetting, который считается за всё обучение целиком")

    ax = axes[1]
    e20 = load_score("el2n_ep20")
    m = np.isfinite(ref)
    ax.hexbin(e20[m], ref[m], gridsize=45, cmap="Blues", bins="log", mincnt=1, lw=0)
    ax.grid(False)
    ax.set_xlabel("EL2N, эпоха 20")
    ax.set_ylabel("число забываний за обучение")
    ps.title(ax, f"Спирмен = {spearman(e20, ref):.2f}",
             "плотность примеров CIFAR-10, темнее — больше")

    ax = axes[2]
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color=ps.SURFACE if M[i, j] > 0.6 else ps.INK)
    fig.colorbar(im, ax=ax, label="Спирмен", shrink=0.85)
    ps.title(ax, "Скоры ранжируют похоже", "ранговая корреляция между скорами")
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
    for ax, t, yl, sub in (
            (axes[0], "Test accuracy", "test accuracy",
             "у всех запусков одинаковое число шагов и одно lr-расписание"),
            (axes[1], "Train accuracy на своём подмножестве", "train accuracy",
             "каждое подмножество выучивается до нуля ошибок — дело не в недообучении")):
        ax.set_xlabel("номинальная эпоха (шаг / 390)")
        ax.set_ylabel(yl)
        ps.title(ax, t, sub)
    axes[0].set_ylim(0.38, 0.98)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncols=len(l), bbox_to_anchor=(0.5, -0.06),
               frameon=False, fontsize=9)
    ps.finish(fig, FIGS / "11_training_curves.png")


# ---------------------------------------------------------------------------------------
#  11 — the two protocol controls
# ---------------------------------------------------------------------------------------
def fig_controls(df):
    """Bars show the DIFFERENCE from training on all data: that quantity has a true zero,
    so the bar lengths are honest (accuracies themselves would need a truncated axis)."""
    rows = []
    for prefix, lab in (("main", "100 эпох, bf16\n(наш протокол)"),
                        ("verify200", "200 эпох, bf16\n(расписание статьи)"),
                        ("fp32", "100 эпох, fp32")):
        sub = df[df.prefix == prefix]
        if sub.empty:
            continue
        full = sub[sub.kind == "full"]["final_test_acc"]
        if not len(full):
            continue
        pr = sub[(sub.kind == "prune") & (sub.keep_frac == 0.5)]
        r = dict(protocol=lab, full=full.mean(), n_full=len(full))
        for meth in ("random", "el2n_ep20"):
            v = pr[pr.method == meth]["final_test_acc"]
            r[meth] = v.mean() - full.mean() if len(v) else np.nan
        rows.append(r)
    if len(rows) < 2:
        return
    t = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = np.arange(len(t))
    w = 0.3
    for i, (meth, lab) in enumerate((("random", "случайные 50%"),
                                     ("el2n_ep20", "EL2N 50%"))):
        c = ps.METHOD_STYLE[meth][0]
        v = t[meth].values * 100
        ax.bar(xs + (i - 0.5) * (w + 0.02), v, w, color=c, label=lab)
        for x, y in zip(xs + (i - 0.5) * (w + 0.02), v):
            if np.isfinite(y):
                ax.text(x, y + (0.07 if y >= 0 else -0.07), f"{y:+.2f}", ha="center",
                        va="bottom" if y >= 0 else "top", fontsize=8.5, color=ps.INK2)
    ax.axhline(0, color=ps.INK2, lw=1.2)
    labels = [f"{p}\nвсе данные {f:.4f}" + (f" ({n} зап.)" if n > 1 else "")
              for p, f, n in zip(t.protocol, t.full, t.n_full)]
    ax.set_xticks(xs, labels, fontsize=8.5)
    lo, hi = np.nanmin(t[["random", "el2n_ep20"]].values) * 100, \
        np.nanmax(t[["random", "el2n_ep20"]].values) * 100
    ax.set_ylim(lo - 0.55, max(hi, 0) + 0.75)
    ax.set_ylabel("разница с обучением на полных данных, п.п.")
    ps.title(ax, "Контроль: вывод не создан отличиями протокола",
             "на расписании статьи эффект EL2N сильнее; в fp32 — в пределах шума по сидам")
    ax.legend(loc="lower left", ncols=2)
    ps.finish(fig, FIGS / "12_controls.png",
              note="Контрольные протоколы — по одному запуску на точку; шум по сидам на "
                   "базовой линии 0.19 п.п., поэтому разницы такого порядка между "
                   "протоколами этими данными не разрешаются.")
    out = t.copy()
    out["random"] = out.random * 100
    out["el2n_ep20"] = out.el2n_ep20 * 100
    out["protocol"] = out.protocol.str.replace("\n", " ")
    save_table("controls", out.rename(columns={
        "protocol": "протокол", "full": "все данные", "n_full": "запусков базы",
        "random": "случайные 50%, п.п.", "el2n_ep20": "EL2N 50%, п.п."}))


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
    jobs = [("prune", lambda: fig_prune(df)),
            ("grand_epoch", lambda: fig_grand_epoch(df)), ("score_epoch", lambda: fig_score_epoch(df)),
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
