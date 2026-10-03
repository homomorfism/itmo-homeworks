#!/usr/bin/env python
"""Every number quoted in the report and the slides, computed from results/.

Run after collect.py. Writes results/tables/key_numbers.md and prints the same.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import plots as P  # noqa: E402

OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


def pp(x):
    """accuracy -> percentage points, 2 decimals, signed"""
    return f"{x*100:+.2f}"


def main():
    root = ROOT / "results"
    P.FIGS, P.TABLES = root / "figures", root / "tables"
    P.SCORES, P.RUNS = root / "scores", root / "runs"
    df = pd.read_csv(root / "summary.csv")

    say("# Ключевые числа\n")
    say(f"Всего запусков: **{len(df)}**, суммарно "
        f"**{df.wall_minutes.sum()/60:.1f} GPU-часов** на RTX 3090.\n")
    say("| стадия | запусков | GPU-часов |")
    say("|---|---|---|")
    for g, s in df.groupby("group"):
        say(f"| {g} | {len(s)} | {s.wall_minutes.sum()/60:.2f} |")
    say()

    # ---- baseline --------------------------------------------------------------
    b = P.baseline(df)
    if b:
        bm, blo, bhi, bn, bsd = b
        say(f"## База\n\nПолные данные: **{bm:.4f}** (sd {bsd*100:.2f} п.п., "
            f"{bn} запусков, 16–84 перцентиль {blo:.4f}–{bhi:.4f}).\n")

    # ---- main pruning grid -----------------------------------------------------
    d = df[df.group == "prune"]
    if not d.empty and b:
        say("## Прунинг: итоговая test accuracy\n")
        piv = d.groupby(["method", "keep_frac"]).final_test_acc.mean().unstack(0)
        cols = [c for c in P.METHODS if c in piv.columns]
        piv = piv[cols]
        say("| оставлено | " + " | ".join(cols) + " |")
        say("|---" * (len(cols) + 1) + "|")
        for k, row in piv.iterrows():
            say(f"| {int(k*100)}% | " + " | ".join(f"{v:.4f}" for v in row) + " |")
        say()
        say("Разница с обучением на полных данных, п.п.:\n")
        say("| оставлено | " + " | ".join(cols) + " |")
        say("|---" * (len(cols) + 1) + "|")
        for k, row in piv.iterrows():
            say(f"| {int(k*100)}% | " + " | ".join(pp(v - bm) for v in row) + " |")
        say()
        tol = bsd  # "без потери" = в пределах одного sd базовой линии
        say(f"Наименьшая доля данных, на которой метод ещё держится в пределах одного sd "
            f"базы ({tol*100:.2f} п.п.):\n")
        for c in cols:
            ok = [k for k, v in piv[c].items() if v >= bm - tol]
            say(f"- **{c}**: {int(min(ok)*100)}% данных" if ok else
                f"- **{c}**: не достигает базы ни на одной из проверенных долей")
        say()

    # ---- score epoch -----------------------------------------------------------
    d = df[df.group == "score_epoch"]
    if not d.empty and b:
        keep = d.keep_frac.iloc[0]
        rnd = df[(df.group == "prune") & (df.method == "random") & (df.keep_frac == keep)]
        say(f"## Насколько рано можно считать скор ({int(keep*100)}% данных)\n")
        say("| эпоха EL2N | test acc | vs база, п.п. |")
        say("|---|---|---|")
        for ep, v in d.groupby("score_epoch").final_test_acc.mean().items():
            say(f"| {int(ep)} | {v:.4f} | {pp(v - bm)} |")
        if not rnd.empty:
            say(f"\nСлучайные {int(keep*100)}%: {rnd.final_test_acc.mean():.4f} "
                f"({pp(rnd.final_test_acc.mean() - bm)} п.п. к базе).\n")

    # ---- score averaging -------------------------------------------------------
    ref = P.load_score("el2n_ep20")
    if ref is not None:
        n_ref = P.n_models_of("el2n_ep20")
        singles = [np.load(p) for p in sorted(
            (P.RUNS / "scores").glob("run*/scores/el2n_ep20.npy"))]
        if len(singles) > 1:
            S = np.stack(singles)
            pair = [P.spearman(S[i], S[j]) for i in range(len(S))
                    for j in range(i + 1, len(S))]
            say("## Усреднение скора\n")
            say(f"- корреляция Спирмена между EL2N двух независимых моделей: "
                f"**{np.mean(pair):.3f}** (разброс {np.min(pair):.3f}–{np.max(pair):.3f})")
            say(f"- корреляция одной модели со средним по {n_ref}: "
                f"**{np.mean([P.spearman(s, ref) for s in S]):.3f}**\n")

    # ---- score agreement -------------------------------------------------------
    fst = P.forget_stem()
    if fst is not None:
        fr = P.load_score(fst)
        say("## Согласие скоров (Спирмен)\n")
        for stem, lab in [("el2n_ep1", "EL2N эпоха 1"), ("el2n_ep5", "EL2N эпоха 5"),
                          ("el2n_ep10", "EL2N эпоха 10"), ("el2n_ep20", "EL2N эпоха 20"),
                          ("grand_ep1", "GraNd эпоха 1"), ("grand_ep20", "GraNd эпоха 20")]:
            s = P.load_score(stem)
            if s is not None:
                say(f"- {lab} vs forgetting: **{P.spearman(s, fr):.3f}**")
        g1, e20 = P.load_score("grand_ep1"), P.load_score("el2n_ep20")
        if g1 is not None and e20 is not None:
            say(f"- GraNd эпоха 1 vs EL2N эпоха 20: **{P.spearman(g1, e20):.3f}**")
        say()

    # ---- window ----------------------------------------------------------------
    for g, lab in (("offset", "чистые метки"), ("noise10_offset", "10% шума в метках")):
        d = df[df.group == g]
        if d.empty:
            continue
        a = d.groupby("offset").final_test_acc.mean()
        say(f"## Скользящее окно, {lab}\n")
        say(f"- размер окна: {int(d.subset_size.iloc[0])} примеров "
            f"({d.keep_frac.iloc[0]*100:.0f}% CIFAR-10)")
        say(f"- offset 0 (только самые сложные): {a.loc[0]:.4f}")
        say(f"- лучший offset = **{int(a.idxmax())}**: {a.max():.4f} "
            f"(**{pp(a.max() - a.loc[0])} п.п.** к offset 0)\n")

    # ---- label noise -----------------------------------------------------------
    d = df[df.group == "noise10_prune"]
    if not d.empty:
        bn10 = P.baseline(df, "noise10_full")
        say("## Прунинг при 10% перемешанных меток\n")
        if bn10:
            say(f"Все шумные данные: **{bn10[0]:.4f}** ({bn10[3]} запусков)\n")
        piv = d.groupby(["method", "keep_frac"]).final_test_acc.mean().unstack(0)
        say("| оставлено | " + " | ".join(piv.columns) + " |")
        say("|---" * (len(piv.columns) + 1) + "|")
        for k, row in piv.iterrows():
            say(f"| {int(k*100)}% | " + " | ".join(f"{v:.4f}" for v in row) + " |")
        say()

    hits = (sorted(P.SCORES.glob("noise10_el2n_ep10_mean*.npy"))
            or sorted(P.SCORES.glob("noise10_el2n_ep*_mean*.npy")))
    if hits:
        from data_diet.data import load_raw, randomize_labels
        _, y, _, _ = load_raw("cifar10", str(ROOT / "data"))
        noisy = randomize_labels(y, 0.1, 7777) != y
        s = np.load(hits[-1])
        order = np.argsort(-s)
        cum = np.cumsum(noisy[order])
        say("## EL2N как детектор испорченных меток\n")
        say(f"Испорчено {noisy.sum()} меток из {len(y)} ({noisy.mean()*100:.2f}%).\n")
        say("| k верхних по EL2N | точность | полнота |")
        say("|---|---|---|")
        for k in (1000, 2500, 4500, 5000, 7500, 10000):
            say(f"| {k} | {cum[k-1]/k:.3f} | {cum[k-1]/noisy.sum():.3f} |")
        say()

    # ---- controls --------------------------------------------------------------
    say("## Контроли протокола\n")
    rows = []
    for prefix, lab in (("main", "100 эпох, bf16 (наш протокол)"),
                        ("verify200", "200 эпох, bf16 (расписание статьи)"),
                        ("fp32", "100 эпох, fp32")):
        sub = df[df.prefix == prefix]
        if sub.empty:
            continue
        full = sub[sub.kind == "full"].final_test_acc
        pr = sub[(sub.kind == "prune") & (sub.keep_frac == 0.5)]
        r = {"протокол": lab, "все данные": full.mean() if len(full) else np.nan}
        for m in ("random", "el2n_ep20"):
            v = pr[pr.method == m].final_test_acc
            r[m + " 50%"] = v.mean() if len(v) else np.nan
        rows.append(r)
    if rows:
        t = pd.DataFrame(rows)
        say(P.to_md(t))
        say()
        for r in rows:
            if not np.isnan(r.get("el2n_ep20 50%", np.nan)) and not np.isnan(r["все данные"]):
                say(f"- {r['протокол']}: EL2N 50% − все данные = "
                    f"**{pp(r['el2n_ep20 50%'] - r['все данные'])} п.п.**, "
                    f"случайные 50% − все данные = "
                    f"**{pp(r['random 50%'] - r['все данные'])} п.п.**")
        say()

    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "tables" / "key_numbers.md").write_text("\n".join(OUT) + "\n")
    print(f"\n-> {root/'tables'/'key_numbers.md'}")


if __name__ == "__main__":
    main()
