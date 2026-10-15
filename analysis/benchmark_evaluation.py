"""Benchmark NumPy float64 batch evaluation of SR and Eggleton fairly.

Input generation and checks are outside the timer; allocation of the returned
array and expression temporaries is included for both implementations.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sr(q):
    return np.cbrt(q / (7.6 * np.sqrt(q) + 1.85 * q + 8.8))


def eggleton(q):
    # Reuse q^(1/3); do not charge the baseline for redundant powers.
    t = np.cbrt(q)
    t2 = t * t
    return 0.49 * t2 / (0.6 * t2 + np.log1p(t))


def cpu_name():
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10_000_000)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--cpu-name", help="Explicit hardware label if automatic detection is unavailable")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    if args.size < 1 or args.repeats < 4 or args.warmups < 1:
        parser.error("size >= 1, repeats >= 4, and warmups >= 1 are required")

    probe = np.logspace(-5, 5, 257, dtype=np.float64)
    literal_sr = (probe / (7.6 * np.sqrt(probe) + 1.85 * probe + 8.8))**(1 / 3)
    literal_egg = 0.49 * probe**(2 / 3) / (0.6 * probe**(2 / 3) + np.log1p(probe**(1 / 3)))
    np.testing.assert_allclose(sr(probe), literal_sr, rtol=2e-14, atol=0)
    np.testing.assert_allclose(eggleton(probe), literal_egg, rtol=2e-14, atol=0)

    rng = np.random.Generator(np.random.PCG64(args.seed))
    q = np.ascontiguousarray(10.0**rng.uniform(-5, 5, args.size), dtype=np.float64)
    digest = hashlib.sha256(memoryview(q)).hexdigest()
    methods = {"Egg": eggleton, "SR": sr}
    for _ in range(args.warmups):
        for method in methods.values():
            result = method(q)
            del result
    rows = []
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(args.repeats):
            order = ("Egg", "SR") if repeat % 2 == 0 else ("SR", "Egg")
            for position, name in enumerate(order, start=1):
                start = time.perf_counter_ns()
                result = methods[name](q)
                elapsed = (time.perf_counter_ns() - start) / 1e9
                # Inspect outside the timed region, and release before the next call.
                if result.shape != q.shape or result.dtype != np.float64 or not np.all(np.isfinite(result)):
                    raise RuntimeError("Invalid benchmark output")
                rows.append({"repeat": repeat + 1, "position": position,
                             "formula": name, "seconds": elapsed})
                del result
            print(f"Completed timing pair {repeat + 1}/{args.repeats}", flush=True)
    finally:
        if was_enabled:
            gc.enable()

    timing = {}
    for name in methods:
        values = np.array([row["seconds"] for row in rows if row["formula"] == name])
        timing[name] = {
            "median_seconds": float(np.median(values)),
            "q25_seconds": float(np.quantile(values, 0.25)),
            "q75_seconds": float(np.quantile(values, 0.75)),
            "min_seconds": float(values.min()), "max_seconds": float(values.max()),
            "all_seconds": values.tolist(),
        }
    speedup = timing["Egg"]["median_seconds"] / timing["SR"]["median_seconds"]
    report = {
        "run_date_Asia_Shanghai": time.strftime("%Y-%m-%d", time.gmtime(time.time() + 8 * 3600)),
        "environment": {"cpu": args.cpu_name or cpu_name(),
                        "logical_processors": os.cpu_count(), "platform": platform.platform(),
                        "python": platform.python_version(), "numpy": np.__version__,
                        "timer": "time.perf_counter_ns", "timer_resolution_seconds": time.get_clock_info("perf_counter").resolution},
        "protocol": {"batch_size": args.size, "dtype": "float64", "q_min": 1e-5,
                     "q_max": 1e5, "distribution": "uniform in log10(q)",
                     "rng": "PCG64", "seed": args.seed, "input_sha256": digest,
                     "warmup_calls_per_formula": args.warmups,
                     "timed_calls_per_formula": args.repeats,
                     "order": "alternating Egg/SR and SR/Egg",
                     "includes": "formula evaluation and all result/temporary-array allocations",
                     "excludes": "input generation, warmups, validation, file I/O and result deallocation",
                     "eggleton_implementation": "one shared cbrt(q), t*t, log1p(t)",
                     "sr_implementation": "sqrt(q), arithmetic denominator, cbrt of the ratio"},
        "timing": timing, "median_time_ratio_Egg_over_SR": speedup,
        "median_time_reduction_SR_percent": float(100 * (1 - 1 / speedup)),
        "correctness_check": "both implementations agree with literal formulas to rtol=2e-14 on 257 logarithmic probe points",
    }
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "timing_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_directory / "timing_trials.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
