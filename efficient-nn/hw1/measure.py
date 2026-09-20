"""HW1 measurement protocol: latency / peak memory / energy / kernel names over
the (S, B) grid, on one GPU.

    python measure.py                       # full grid -> results/measurements.csv, results/kernels.csv
    python measure.py --mem-limit-gb 2      # emulate a 2 GB GPU to exercise the OOM path
                                            #   -> results/measurements_memlimit2gb.csv (memory/OOM only)

Protocol (identical for every configuration):
  * flags: cudnn.benchmark=False, allow_tf32=False (cudnn + matmul); model.cuda().eval();
    torch.inference_mode(); FP32; random input tensor.
  * memory : torch.cuda.reset_peak_memory_stats() after the input has been allocated,
             then one forward -> torch.cuda.max_memory_allocated()  (weights + input included).
  * latency: warm-up passes, then per-pass wall time (perf_counter around forward +
             torch.cuda.synchronize()), median over N passes (N chosen so that the
             timed block lasts >= LAT_TARGET_S, 20 <= N <= 300).
  * energy : NVML nvmlDeviceGetTotalEnergyConsumption (whole board, mJ counter) read
             before/after a loop of K passes (each followed by torch.cuda.synchronize(),
             exactly like the latency protocol) lasting >= ENERGY_TARGET_S;
             energy per pass = delta / K.  Idle power is measured once at start.
  * kernels: one torch.profiler pass per configuration, CUDA kernels grouped by layer.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from models import build_model, layer_names

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
N_RANDOM_S, N_RANDOM_B = 4, 3
SEED = 2026

WARMUP = 10
LAT_TARGET_S = 0.7      # timed block length for the latency median
LAT_MIN_ITERS, LAT_MAX_ITERS = 20, 300
ENERGY_TARGET_S = 1.5   # integration window for the NVML energy counter
ENERGY_MIN_ITERS = 10


# ----------------------------------------------------------------------------
def set_flags():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def make_grid(seed=SEED):
    """Base grid + random validation points (fixed seed so the run is reproducible)."""
    rng = np.random.default_rng(seed)
    cand_S = [s for s in range(32, 513, 16) if s not in BASE_S]
    rand_S = sorted(int(s) for s in rng.choice(cand_S, N_RANDOM_S, replace=False))
    cand_B = [b for b in range(1, 257) if (b & (b - 1)) != 0]   # not powers of two
    rand_B = sorted(int(b) for b in rng.choice(cand_B, N_RANDOM_B, replace=False))
    all_S = sorted(BASE_S + rand_S)
    all_B = sorted(BASE_B + rand_B)
    return all_S, all_B, rand_S, rand_B


# ----------------------------------------------------------------------------
class NVML:
    def __init__(self):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    def energy_j(self):
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1e3   # mJ -> J

    def power_w(self):
        return self.nv.nvmlDeviceGetPowerUsage(self.h) / 1e3

    def clocks(self):
        sm = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)
        mem = self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_MEM)
        return sm, mem

    def temp(self):
        return self.nv.nvmlDeviceGetTemperature(self.h, self.nv.NVML_TEMPERATURE_GPU)

    def idle(self, seconds=5.0):
        torch.cuda.synchronize()
        time.sleep(1.0)
        e0, t0 = self.energy_j(), time.perf_counter()
        time.sleep(seconds)
        e1, t1 = self.energy_j(), time.perf_counter()
        return (e1 - e0) / (t1 - t0)


def env_info(nvml: NVML):
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total,power.limit,clocks.max.sm,clocks.max.memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": smi,
        "sm_count": p.multi_processor_count,
        "total_memory_bytes": p.total_memory,
        "l2_cache_bytes": p.L2_cache_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "idle_power_w": nvml.idle(),
        "flags": {"cudnn.benchmark": torch.backends.cudnn.benchmark,
                  "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
                  "matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "protocol": {"warmup": WARMUP, "lat_target_s": LAT_TARGET_S,
                     "lat_iters": [LAT_MIN_ITERS, LAT_MAX_ITERS],
                     "energy_target_s": ENERGY_TARGET_S, "energy_min_iters": ENERGY_MIN_ITERS,
                     "seed": SEED},
    }


# ----------------------------------------------------------------------------
def _kernels_under(ev):
    """CUDA kernels launched inside a CPU profiler event, deduplicated: the
    profiler attaches the same kernel list to an op and to its children, so
    take the kernels of the deepest ops only."""
    child = []
    for c in ev.cpu_children:
        child += _kernels_under(c)
    if child:
        return child
    return list(getattr(ev, "kernels", None) or [])


def profile_kernels(model, names, x):
    """One profiled forward.  Returns (rows [(layer, kernel_name, duration_us)], profiler_flops)."""
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
            h = x
            for n, mod in zip(names, model):
                with record_function(f"L::{n}"):
                    h = mod(h)
            torch.cuda.synchronize()
    rows, pflops = [], 0
    for ev in prof.events():
        if ev.device_type != torch.autograd.DeviceType.CPU:
            continue
        if ev.flops:
            pflops += ev.flops
        if ev.name.startswith("L::"):
            layer = ev.name[3:]
            ks = _kernels_under(ev)
            if not ks:
                rows.append((layer, "", 0.0))
            for k in ks:
                rows.append((layer, k.name, float(k.duration)))
    return rows, pflops


# ----------------------------------------------------------------------------
def measure_one(model, names, S, B, nvml: NVML | None, do_energy=True, do_profile=True):
    """Measure one (S, B).  Returns a dict; raises torch.cuda.OutOfMemoryError on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    x = torch.randn(B, 3, S, S, device="cuda")
    out = {}

    with torch.inference_mode():
        # --- peak memory of the first (cold, may allocate cuDNN workspace) pass
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        model(x)
        torch.cuda.synchronize()
        out["memory_bytes"] = torch.cuda.max_memory_allocated()
        out["memory_base_bytes"] = base            # weights + input, before the pass

        # --- warm-up
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()

        # --- peak memory of a warm pass (should equal the cold one)
        torch.cuda.reset_peak_memory_stats()
        model(x)
        torch.cuda.synchronize()
        out["memory_bytes_warm"] = torch.cuda.max_memory_allocated()

        # --- latency
        t = time.perf_counter(); model(x); torch.cuda.synchronize(); t_est = time.perf_counter() - t
        n = int(np.clip(LAT_TARGET_S / max(t_est, 1e-6), LAT_MIN_ITERS, LAT_MAX_ITERS))
        times = []
        for _ in range(n):
            t = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        out["latency_s"] = statistics.median(times)
        out["latency_min_s"] = min(times)
        out["latency_p90_s"] = float(np.percentile(times, 90))
        out["latency_iters"] = n

        # --- energy (NVML board energy counter around a back-to-back loop)
        if do_energy and nvml is not None:
            k = max(ENERGY_MIN_ITERS, int(ENERGY_TARGET_S / max(out["latency_s"], 1e-6)))
            torch.cuda.synchronize()
            e0, t0 = nvml.energy_j(), time.perf_counter()
            for _ in range(k):
                model(x)
                torch.cuda.synchronize()      # same protocol as the latency measurement
            e1, t1 = nvml.energy_j(), time.perf_counter()
            out["energy_j"] = (e1 - e0) / k
            out["energy_iters"] = k
            out["energy_window_s"] = t1 - t0
            out["mean_power_w"] = (e1 - e0) / (t1 - t0)
            out["latency_loop_s"] = (t1 - t0) / k           # mean time per pass inside the energy loop
            out["sm_clock_mhz"], out["mem_clock_mhz"] = nvml.clocks()
            out["gpu_temp_c"] = nvml.temp()

    # --- kernels
    if do_profile:
        rows, pflops = profile_kernels(model, names, x)
        out["kernel_rows"] = rows
        out["flops_profiler"] = pflops

    del x
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-limit-gb", type=float, default=None,
                    help="emulate a smaller GPU via torch.cuda.set_per_process_memory_fraction; "
                         "memory/OOM only, written to a separate csv")
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    set_flags()
    torch.manual_seed(SEED)
    model = build_model()
    names = layer_names(model)

    memlimit_mode = args.mem_limit_gb is not None
    if memlimit_mode:
        total = torch.cuda.get_device_properties(0).total_memory
        frac = args.mem_limit_gb * 2**30 / total
        torch.cuda.set_per_process_memory_fraction(frac)
        do_energy, do_profile = False, False
        out_csv = args.out or os.path.join(RESULTS_DIR, f"measurements_memlimit{args.mem_limit_gb:g}gb.csv")
        nvml = None
        print(f"emulating a {args.mem_limit_gb} GB GPU (fraction {frac:.4f}) -> {out_csv}")
    else:
        do_energy, do_profile = not args.no_energy, not args.no_profile
        out_csv = args.out or os.path.join(RESULTS_DIR, "measurements.csv")
        nvml = NVML()
        info = env_info(nvml)
        with open(os.path.join(RESULTS_DIR, "env.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(json.dumps(info, indent=2))

    all_S, all_B, rand_S, rand_B = make_grid()
    print("S grid:", all_S, " (validation:", rand_S, ")")
    print("B grid:", all_B, " (validation:", rand_B, ")")

    fields = ["S", "B", "is_validation", "oom", "latency_s", "latency_min_s", "latency_p90_s", "latency_iters",
              "latency_loop_s", "memory_bytes", "memory_bytes_warm", "memory_base_bytes",
              "energy_j", "energy_iters", "energy_window_s", "mean_power_w",
              "sm_clock_mhz", "mem_clock_mhz", "gpu_temp_c", "flops_profiler", "elapsed_s"]
    rows, krows = [], []
    t_start = time.time()
    for S in all_S:
        for B in all_B:
            is_val = int(S in rand_S or B in rand_B)
            row = {"S": S, "B": B, "is_validation": is_val, "oom": 0}
            t0 = time.time()
            try:
                r = measure_one(model, names, S, B, nvml, do_energy, do_profile)
                for k, v in r.items():
                    if k == "kernel_rows":
                        krows += [(S, B, layer, kname, dur) for (layer, kname, dur) in v]
                    else:
                        row[k] = v
                msg = (f"lat {r['latency_s']*1e3:8.3f} ms  mem {r['memory_bytes']/2**20:9.1f} MiB"
                       + (f"  E {r['energy_j']*1e3:9.2f} mJ  P {r['mean_power_w']:6.1f} W" if "energy_j" in r else ""))
            except torch.cuda.OutOfMemoryError:
                row["oom"] = 1
                msg = "OOM"
                torch.cuda.empty_cache()
            row["elapsed_s"] = round(time.time() - t0, 2)
            rows.append(row)
            print(f"S={S:4d} B={B:4d} val={is_val}  {msg}   [{row['elapsed_s']:.1f}s, total {time.time()-t_start:.0f}s]", flush=True)

            # write incrementally so a crash keeps partial results
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            if do_profile:
                with open(os.path.join(RESULTS_DIR, "kernels.csv"), "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["S", "B", "layer", "kernel", "duration_us"])
                    w.writerows(krows)
    print("done in %.0f s -> %s" % (time.time() - t_start, out_csv))


if __name__ == "__main__":
    main()
