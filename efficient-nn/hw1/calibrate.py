"""Calibrate theta (latency) and theta_energy (energy) on the measured grid and
evaluate all four equations on calibration and validation points.

    python calibrate.py            # reads results/measurements.csv, writes results/theta.json

Fitting:
  * latency  : theta = (t_launch, bandwidth, peak_flops), positive; minimise the sum of
               squared log-ratios log(pred/meas) (i.e. relative error) with
               scipy.optimize.least_squares, several restarts (the per-layer max() is
               non-smooth).  Fitted on is_validation == 0 rows only.
  * energy   : theta_energy = (p_static, e_flop, e_byte) >= 0; linear in the parameters
               once latency(S,B,theta) is known -> weighted NNLS (rows weighted by 1/E,
               i.e. relative error), calibration rows only.
  * memory / flops : no parameters; only evaluated.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

import equations as eq

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def rel_err(pred, meas):
    return pred / meas - 1.0


def summarize(pred, meas):
    e = np.abs(rel_err(pred, meas))
    return {"n": int(len(e)), "median_abs_rel_err": float(np.median(e)),
            "mean_abs_rel_err": float(np.mean(e)), "max_abs_rel_err": float(np.max(e)),
            "rmse_log": float(np.sqrt(np.mean(np.log(pred / meas) ** 2)))}


def fit_latency(S, B, T, model=eq.latency):
    def unpack(p):
        return {"t_launch": np.exp(p[0]), "bandwidth": np.exp(p[1]), "peak_flops": np.exp(p[2])}

    def resid(p):
        return np.log(model(S, B, unpack(p)) / T)

    best = None
    for t0 in (5e-6, 15e-6, 40e-6):
        for bw in (300e9, 700e9, 900e9):
            for pf in (10e12, 20e12, 35e12):
                r = least_squares(resid, np.log([t0, bw, pf]), method="trf")
                if best is None or r.cost < best.cost:
                    best = r
    return unpack(best.x)


def fit_energy(S, B, E, theta):
    """E = p_static * T + e_flop * FLOPs (+ e_byte * Bytes).

    For a fixed architecture FLOPs and Bytes are both ~ B*S^2 (FLOPs/Bytes = 33.4 for
    every (S, B) except the tiny constant terms), so e_flop and e_byte are collinear and
    cannot be identified separately from measurements of this network alone.  We fit
    (p_static, e_flop) and report e_byte = 0: e_flop then means "dynamic energy per FLOP
    *including* the DRAM traffic that comes with it at 33 FLOP/byte".
    """
    T = eq.latency(S, B, theta)
    A = np.stack([T, eq.flops(S, B)], axis=1)
    w = 1.0 / E                        # relative-error weighting
    coef, _ = nnls(A * w[:, None], E * w)
    return {"p_static": float(coef[0]), "e_flop": float(coef[1]), "e_byte": 0.0,
            "e_flop_equivalent_pJ_per_byte_if_all_dynamic_energy_were_DRAM":
                float(coef[1] * eq.FLOPS_PER_BS2 / (4 * 133) * 1e12),
            "latency": theta}


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "measurements.csv"))
    ok = df[df.oom == 0].copy()
    cal, val = ok[ok.is_validation == 0], ok[ok.is_validation == 1]
    print(f"{len(df)} configs, {int(df.oom.sum())} OOM, {len(cal)} calibration, {len(val)} validation")

    S_c, B_c = cal.S.values.astype(float), cal.B.values.astype(float)
    S_v, B_v = val.S.values.astype(float), val.B.values.astype(float)
    from measure import BASE_B, BASE_S
    out = {"grid": {"S": sorted(df.S.unique().tolist()), "B": sorted(df.B.unique().tolist()),
                    "random_S": sorted(set(df.S.unique().tolist()) - set(BASE_S)),
                    "random_B": sorted(set(df.B.unique().tolist()) - set(BASE_B)),
                    "note": "is_validation = 1 iff S is a random S or B is a random B; theta is fitted on is_validation == 0 only"}}

    # ---- latency: per-layer roofline (main) and additive (baseline)
    theta = fit_latency(S_c, B_c, cal.latency_s.values)
    theta_add = fit_latency(S_c, B_c, cal.latency_s.values, model=eq.latency_additive)
    out["theta"] = theta
    out["theta_additive"] = theta_add
    out["latency_eval"] = {
        "roofline": {"calibration": summarize(eq.latency(S_c, B_c, theta), cal.latency_s.values),
                     "validation": summarize(eq.latency(S_v, B_v, theta), val.latency_s.values)},
        "additive": {"calibration": summarize(eq.latency_additive(S_c, B_c, theta_add), cal.latency_s.values),
                     "validation": summarize(eq.latency_additive(S_v, B_v, theta_add), val.latency_s.values)},
    }

    # ---- energy
    theta_e = fit_energy(S_c, B_c, cal.energy_j.values, theta)
    out["theta_energy"] = theta_e
    out["energy_eval"] = {"calibration": summarize(eq.energy(S_c, B_c, theta_e), cal.energy_j.values),
                          "validation": summarize(eq.energy(S_v, B_v, theta_e), val.energy_j.values)}
    # energy model with *measured* latency plugged in (isolates the energy model's own error)
    Tm_c, Tm_v = cal.latency_s.values, val.latency_s.values
    Em_c = theta_e["p_static"] * Tm_c + theta_e["e_flop"] * eq.flops(S_c, B_c) + theta_e["e_byte"] * eq.bytes_moved(S_c, B_c)
    Em_v = theta_e["p_static"] * Tm_v + theta_e["e_flop"] * eq.flops(S_v, B_v) + theta_e["e_byte"] * eq.bytes_moved(S_v, B_v)
    out["energy_eval_with_measured_latency"] = {"calibration": summarize(Em_c, cal.energy_j.values),
                                                "validation": summarize(Em_v, val.energy_j.values)}

    # ---- memory & flops (parameter-free)
    mem_col = "memory_bytes_warm"
    out["memory_eval"] = {"calibration": summarize(eq.memory(S_c, B_c), cal[mem_col].values),
                          "validation": summarize(eq.memory(S_v, B_v), val[mem_col].values),
                          "all_abs_err_MiB_max": float(np.max(np.abs(eq.memory(ok.S.values, ok.B.values) - ok[mem_col].values)) / 2**20)}
    if "flops_profiler" in ok:
        conv_lin = eq.CONV_FLOPS_PER_BS2 * ok.B.values * ok.S.values.astype(float) ** 2 + (eq.FLOPS_PER_B - eq.GAP_OUT_PER_B - 256) * ok.B.values
        out["flops_eval"] = {
            "formula_vs_profiler(all_ops)": summarize(eq.flops(ok.S.values, ok.B.values), ok.flops_profiler.values),
            "formula_conv+linear_only_vs_profiler": summarize(conv_lin, ok.flops_profiler.values),
        }

    # ---- OOM check against an emulated smaller GPU, if such a run exists
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.startswith("measurements_memlimit") and fn.endswith(".csv"):
            lim_gb = float(fn[len("measurements_memlimit"):-len("gb.csv")])
            dm = pd.read_csv(os.path.join(RESULTS_DIR, fn))
            pred_oom = eq.memory(dm.S.values, dm.B.values) > lim_gb * 2**30
            agree = (pred_oom == (dm.oom.values == 1))
            out.setdefault("oom_eval", {})[fn] = {
                "limit_gb": lim_gb, "n_oom_measured": int(dm.oom.sum()), "n_oom_predicted": int(pred_oom.sum()),
                "agreement": float(agree.mean()),
                "disagreements": [dict(S=int(r.S), B=int(r.B), oom=int(r.oom), pred_MiB=float(eq.memory(r.S, r.B) / 2**20))
                                  for r, a in zip(dm.itertuples(), agree) if not a]}

    # ---- derived hardware-ish numbers for the discussion
    out["derived"] = {
        "t_launch_us": theta["t_launch"] * 1e6,
        "bandwidth_GBps": theta["bandwidth"] / 1e9,
        "peak_TFLOPS": theta["peak_flops"] / 1e12,
        "ridge_point_flop_per_byte": theta["peak_flops"] / theta["bandwidth"],
        "p_static_W": theta_e["p_static"],
        "pJ_per_flop": theta_e["e_flop"] * 1e12,
        "saturated_power_W_(p_static + e_flop*peak_flops)": theta_e["p_static"] + theta_e["e_flop"] * theta["peak_flops"],
        "idle_power_W": json.load(open(os.path.join(RESULTS_DIR, "env.json")))["idle_power_w"],
        "measured_mean_power_W_min_max": [float(ok.mean_power_w.min()), float(ok.mean_power_w.max())],
        "sm_clock_MHz_min_max": [int(ok.sm_clock_mhz.min()), int(ok.sm_clock_mhz.max())],
    }
    with open(os.path.join(RESULTS_DIR, "theta.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
