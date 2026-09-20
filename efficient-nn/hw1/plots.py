"""Figures: measured points + predicted curves/surfaces for FLOPs, Memory, Latency, Energy.

    python plots.py     # reads results/measurements.csv, results/theta.json, results/kernels.csv
                        # writes results/figures/*.png
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import equations as eq

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# palette (validated defaults, light surface): one blue sequential ramp for the
# ordered variable S, orange as the single accent, diverging blue<->red for signed error
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ACCENT = "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def s_color(S, all_S):
    """Sequential ramp keyed to the ordered variable S (step 250..700)."""
    idx = sorted(all_S).index(S)
    ramp = SEQ[3:]
    return ramp[int(round(idx * (len(ramp) - 1) / max(len(all_S) - 1, 1)))]


def curves_vs_B(ax, df, ycol, pred_fn, ylabel, title, all_S, per_image=False, unit_scale=1.0, legend_loc="upper left"):
    """Measured markers + predicted lines vs batch size, one line per S."""
    Bfine = np.geomspace(1, 256, 200)
    for S in all_S:
        d = df[(df.S == S) & (df.oom == 0)]
        if d.empty:
            continue
        c = s_color(S, all_S)
        div = Bfine if per_image else 1.0
        ax.plot(Bfine, pred_fn(S, Bfine) * unit_scale / div, color=c, lw=1.6, zorder=2)
        divm = d.B.values if per_image else 1.0
        cal, val = d[d.is_validation == 0], d[d.is_validation == 1]
        ax.plot(cal.B, cal[ycol] * unit_scale / (cal.B.values if per_image else 1.0), "o", ms=4, color=c,
                mec="white", mew=0.6, zorder=3)
        ax.plot(val.B, val[ycol] * unit_scale / (val.B.values if per_image else 1.0), "o", ms=5, mfc="white",
                mec=c, mew=1.3, zorder=4)
        ax.annotate(f"S={S}", (Bfine[-1], pred_fn(S, Bfine[-1]) * unit_scale / (Bfine[-1] if per_image else 1.0)),
                    xytext=(4, 0), textcoords="offset points", fontsize=7, color=c, va="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("batch size B"); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=10)
    ax.set_xticks([1, 4, 16, 64, 256]); ax.set_xticklabels(["1", "4", "16", "64", "256"])
    ax.plot([], [], "o", ms=4, color=INK2, mec="white", label="measured (calibration)")
    ax.plot([], [], "o", ms=5, mfc="white", mec=INK2, mew=1.3, label="measured (validation, unseen)")
    ax.plot([], [], "-", color=INK2, lw=1.6, label="predicted")
    ax.legend(loc=legend_loc, fontsize=7)


def error_heatmap(ax, df, ycol, pred_fn, title, all_S, all_B, vmax=0.5):
    """Signed relative error (pred/meas - 1) over the (S, B) grid; OOM cells hatched."""
    M = np.full((len(all_S), len(all_B)), np.nan)
    for i, S in enumerate(all_S):
        for j, B in enumerate(all_B):
            r = df[(df.S == S) & (df.B == B)]
            if len(r) and r.oom.values[0] == 0:
                M[i, j] = pred_fn(S, B) / r[ycol].values[0] - 1
    im = ax.imshow(M, cmap=DIV, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto", origin="lower")
    ax.set_xticks(range(len(all_B))); ax.set_xticklabels(all_B, fontsize=7)
    ax.set_yticks(range(len(all_S))); ax.set_yticklabels(all_S, fontsize=7)
    ax.set_xlabel("batch size B"); ax.set_ylabel("image size S"); ax.grid(False)
    ax.set_title(title + "   (boxed cells = validation / unseen points)", loc="left", fontsize=10)
    for i, S in enumerate(all_S):
        for j, B in enumerate(all_B):
            r = df[(df.S == S) & (df.B == B)]
            if len(r) and r.oom.values[0] == 1:
                ax.text(j, i, "OOM", ha="center", va="center", fontsize=6, color=INK2)
            elif not np.isnan(M[i, j]):
                ax.text(j, i, f"{100*M[i,j]:+.0f}", ha="center", va="center", fontsize=5.5,
                        color=INK if abs(M[i, j]) < 0.6 * vmax else "white")
            is_val = r.is_validation.values[0] == 1 if len(r) else False
            if is_val:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec=INK, lw=0.8))
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("predicted / measured − 1"); cb.outline.set_visible(False)


def parity(ax, df, ycol, pred_fn, label, unit):
    d = df[df.oom == 0]
    p = pred_fn(d.S.values, d.B.values)
    cal, val = d.is_validation == 0, d.is_validation == 1
    lo, hi = min(p.min(), d[ycol].min()) * 0.7, max(p.max(), d[ycol].max()) * 1.4
    ax.plot([lo, hi], [lo, hi], "-", color=INK2, lw=1); ax.plot([lo, hi], [lo * 1.2, hi * 1.2], ":", color=GRID, lw=1)
    ax.plot([lo, hi], [lo / 1.2, hi / 1.2], ":", color=GRID, lw=1)
    ax.plot(d[ycol][cal], p[cal], "o", ms=4, color=SEQ[7], mec="white", mew=0.5, label="calibration")
    ax.plot(d[ycol][val], p[val], "o", ms=5, mfc="white", mec=ACCENT, mew=1.3, label="validation (unseen)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"measured {label} [{unit}]"); ax.set_ylabel(f"predicted {label} [{unit}]")
    ax.set_title(f"{label}: predicted vs measured (dotted = ±20 %)", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(RESULTS_DIR, "measurements.csv"))
    th = json.load(open(os.path.join(RESULTS_DIR, "theta.json")))
    theta, theta_e, theta_add = th["theta"], th["theta_energy"], th["theta_additive"]
    all_S, all_B = sorted(df.S.unique()), sorted(df.B.unique())
    ok = df[df.oom == 0]

    lat = lambda S, B: eq.latency(S, B, theta)
    ene = lambda S, B: eq.energy(S, B, theta_e)
    mem = lambda S, B: eq.memory(S, B)

    # ------------------------------------------------------------------ FLOPs
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    curves_vs_B(axs[0], ok, "flops_profiler", eq.flops, "FLOPs per forward pass", "FLOPs(S, B) = 17793·B·S² + 314112·B  vs torch.profiler count",
                all_S, unit_scale=1.0)
    axs[0].set_ylabel("FLOPs")
    S_fine = np.geomspace(32, 512, 100)
    for B in [1, 16, 256]:
        axs[1].plot(S_fine, eq.flops(S_fine, B) / B / 1e9, color=SEQ[7], lw=1.6)
        axs[1].annotate(f"any B (per image)", (S_fine[-1], eq.flops(S_fine[-1], B) / B / 1e9), xytext=(4, 0), textcoords="offset points", fontsize=7, color=SEQ[7], va="center") if B == 1 else None
    d = ok.copy()
    axs[1].plot(d.S, d.flops_profiler / d.B / 1e9, "o", ms=4, color=ACCENT, mec="white", mew=0.5, label="profiler (conv+linear only) / B")
    axs[1].set_xscale("log", base=2); axs[1].set_yscale("log"); axs[1].set_xlabel("image size S"); axs[1].set_ylabel("GFLOPs per image")
    axs[1].set_xticks(all_S); axs[1].set_xticklabels(all_S, fontsize=6, rotation=45)
    axs[1].set_title("FLOPs per image ∝ S²  (slope 2 in log-log)", loc="left", fontsize=10); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "flops.png")); plt.close(fig)

    # ------------------------------------------------------------------ Memory
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1.3]})
    curves_vs_B(axs[0], df, "memory_bytes_warm", mem, "peak memory [MiB]", "Memory(S, B) = 76·B·S² + 13.75 MB", all_S, unit_scale=1 / 2**20)
    x = 76 * ok.B.values * ok.S.values.astype(float) ** 2
    resid = (ok.memory_bytes_warm.values - mem(ok.S.values, ok.B.values)) / 2**20
    axs[1].axhline(0, color=INK2, lw=1)
    axs[1].plot(x[ok.is_validation == 0] / 2**20, resid[ok.is_validation == 0], "o", ms=4, color=SEQ[7], mec="white", mew=0.5, label="calibration")
    axs[1].plot(x[ok.is_validation == 1] / 2**20, resid[ok.is_validation == 1], "o", ms=5, mfc="white", mec=ACCENT, mew=1.3, label="validation")
    axs[1].set_xscale("log"); axs[1].set_xlabel("predicted activation peak 76·B·S² [MiB]"); axs[1].set_ylabel("measured − predicted [MiB]")
    axs[1].set_title("Residual = un-modelled cuDNN conv2 workspace\n(only visible when activations are small)", loc="left", fontsize=10); axs[1].legend(fontsize=7)
    error_heatmap(axs[2], df, "memory_bytes_warm", mem, "Memory: relative error [%]", all_S, all_B, vmax=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "memory.png")); plt.close(fig)

    # ------------------------------------------------------------------ Latency
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    curves_vs_B(axs[0], df, "latency_s", lat, "latency [ms]",
                f"Latency: per-layer roofline, θ = (t₀={theta['t_launch']*1e6:.1f} µs, β={theta['bandwidth']/1e9:.0f} GB/s, π={theta['peak_flops']/1e12:.1f} TFLOP/s)",
                all_S, unit_scale=1e3)
    error_heatmap(axs[1], df, "latency_s", lat, "Latency: relative error [%]", all_S, all_B, vmax=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "latency.png")); plt.close(fig)

    # regimes: latency vs work (B·S²), with the three asymptotes
    fig, ax = plt.subplots(figsize=(8, 4.8))
    work = ok.B.values * ok.S.values.astype(float) ** 2
    for S in all_S:
        m = (ok.S == S).values
        ax.plot(work[m], ok.latency_s.values[m] * 1e3, "o", ms=4, color=s_color(S, all_S), mec="white", mew=0.5)
    w = np.geomspace(1e3, 1e8, 300)
    S_ref, B_ref = 256.0, w / 256.0 ** 2      # walk the B·S² axis along S=256
    ax.plot(w, lat(S_ref, B_ref) * 1e3, "-", color=INK, lw=1.6, label="roofline model, S=256")
    ax.plot(w, eq.latency_additive(S_ref, B_ref, theta_add) * 1e3, "--", color=INK2, lw=1.2, label="additive model (no overlap)")
    n_ops = len(eq.LAYERS)
    ax.axhline(n_ops * theta["t_launch"] * 1e3, color=ACCENT, lw=1, ls=":")
    ax.text(1.2e3, n_ops * theta["t_launch"] * 1e3 * 1.08, f"launch floor: {n_ops} ops × t₀", color=ACCENT, fontsize=7)
    ax.plot(w, eq.bytes_moved(S_ref, B_ref) / theta["bandwidth"] * 1e3, ":", color=SEQ[7], lw=1)
    ax.text(3e7, eq.bytes_moved(S_ref, 3e7 / 256 ** 2) / theta["bandwidth"] * 1e3 * 0.5, "Bytes/β (memory-bound)", color=SEQ[7], fontsize=7, ha="right")
    ax.plot(w, eq.flops(S_ref, B_ref) / theta["peak_flops"] * 1e3, ":", color="#008300", lw=1)
    ax.text(3e7, eq.flops(S_ref, 3e7 / 256 ** 2) / theta["peak_flops"] * 1e3 * 1.5, "FLOPs/π (compute-bound)", color="#008300", fontsize=7, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("work B·S²  (pixels per pass)"); ax.set_ylabel("latency [ms]")
    ax.set_title("Three regimes: launch-bound → memory-bound → compute-bound   (points coloured by S, light→dark)", loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "latency_regimes.png")); plt.close(fig)

    # ------------------------------------------------------------------ Energy
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    curves_vs_B(axs[0], df, "energy_j", ene, "energy per pass [mJ]",
                f"Energy: P₀={theta_e['p_static']:.0f} W · T + {theta_e['e_flop']*1e12:.1f} pJ/FLOP + {theta_e['e_byte']*1e12:.0f} pJ/byte",
                all_S, unit_scale=1e3)
    error_heatmap(axs[1], df, "energy_j", ene, "Energy: relative error [%]", all_S, all_B, vmax=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "energy.png")); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    curves_vs_B(axs[0], df, "energy_j", ene, "energy per image [mJ / image]", "Energy per image: batching amortises the static power", all_S, per_image=True, unit_scale=1e3, legend_loc="lower left")
    for S in all_S:
        d = ok[ok.S == S]
        axs[1].plot(d.B, d.mean_power_w, "o-", ms=3, lw=1, color=s_color(S, all_S))
    axs[1].axhline(th.get("idle_power_w", np.nan) if "idle_power_w" in th else json.load(open(os.path.join(RESULTS_DIR, "env.json")))["idle_power_w"], color=ACCENT, ls=":", lw=1)
    axs[1].text(1, json.load(open(os.path.join(RESULTS_DIR, "env.json")))["idle_power_w"] + 4, "idle power", color=ACCENT, fontsize=7)
    axs[1].set_xscale("log", base=2); axs[1].set_xlabel("batch size B"); axs[1].set_ylabel("mean board power during the loop [W]")
    axs[1].set_title("Measured mean power (lines coloured by S, light→dark)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "energy_per_image_power.png")); plt.close(fig)

    # ------------------------------------------------------------------ Parity (all four)
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    parity(axs[0, 0], ok, "flops_profiler", eq.flops, "FLOPs", "FLOP")
    parity(axs[0, 1], ok, "memory_bytes_warm", mem, "peak memory", "bytes")
    parity(axs[1, 0], ok, "latency_s", lat, "latency", "s")
    parity(axs[1, 1], ok, "energy_j", ene, "energy", "J")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "parity.png")); plt.close(fig)

    # ------------------------------------------------------------------ Per-layer roofline from profiler kernel times
    kf = os.path.join(RESULTS_DIR, "kernels.csv")
    if os.path.exists(kf):
        k = pd.read_csv(kf)
        k = k[k.kernel.notna()]
        dur = k.groupby(["S", "B", "layer"]).duration_us.sum().reset_index()
        fig, ax = plt.subplots(figsize=(8.5, 5))
        names = eq.LAYER_NAMES
        for i, layer in enumerate(names):
            d = dur[dur.layer == layer]
            if d.empty:
                continue
            fl = eq.layer_flops(d.S.values.astype(float), d.B.values.astype(float))[i]
            by = eq.layer_bytes(d.S.values.astype(float), d.B.values.astype(float))[i]
            ai, perf = fl / by, fl / (d.duration_us.values * 1e-6)
            kind = "conv" if layer.startswith("conv") else ("fc" if layer.startswith("fc") else "elementwise/pool")
            c = {"conv": SEQ[8], "fc": ACCENT, "elementwise/pool": "#1baf7a"}[kind]
            ax.plot(ai, perf / 1e9, "o", ms=3, color=c, alpha=0.5, mec="none")
        for kind, c in [("conv (1×1 … 7×7)", SEQ[8]), ("linear", ACCENT), ("BN / ReLU / pool", "#1baf7a")]:
            ax.plot([], [], "o", color=c, label=kind)
        ai_line = np.geomspace(0.05, 1e4, 200)
        ax.plot(ai_line, np.minimum(theta["peak_flops"], theta["bandwidth"] * ai_line) / 1e9, "-", color=INK, lw=1.6, label=f"fitted roofline: β={theta['bandwidth']/1e9:.0f} GB/s, π={theta['peak_flops']/1e12:.1f} TFLOP/s")
        ax.plot(ai_line, np.minimum(35.6e12, 936e9 * ai_line) / 1e9, "--", color=INK2, lw=1, label="RTX 3090 spec: 936 GB/s, 35.6 TFLOP/s FP32")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("arithmetic intensity [FLOP / byte]"); ax.set_ylabel("achieved throughput [GFLOP/s]")
        ax.set_title("Per-layer roofline (every layer of every configuration, kernel time from torch.profiler)", loc="left", fontsize=10)
        ax.legend(fontsize=7, loc="lower right"); ax.set_ylim(0.1, 1e5)
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "roofline_layers.png")); plt.close(fig)

    print("figures written to", FIG_DIR, sorted(os.listdir(FIG_DIR)))


if __name__ == "__main__":
    main()
