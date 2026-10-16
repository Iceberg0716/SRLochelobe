"""Measure the effect of expression implementation using the same inputs."""

import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from benchmark_evaluation import cpu_name, eggleton, sr

ROOT = Path(__file__).resolve().parents[1]


def eggleton_no_reuse(q):
    return 0.49 * np.cbrt(q)**2 / (0.6 * np.cbrt(q)**2 + np.log1p(np.cbrt(q)))


def eggleton_literal(q):
    return 0.49 * q**(2 / 3) / (0.6 * q**(2 / 3) + np.log(1 + q**(1 / 3)))


def sr_literal(q):
    return (q / (7.6 * np.sqrt(q) + 1.85 * q + 8.8))**(1 / 3)


def main():
    methods = {'Egg_shared': eggleton, 'Egg_no_reuse': eggleton_no_reuse,
               'Egg_literal': eggleton_literal, 'SR_cbrt': sr, 'SR_literal': sr_literal}
    probe = np.logspace(-5, 5, 257)
    for name, method in methods.items():
        expected = eggleton_literal(probe) if name.startswith('Egg') else sr_literal(probe)
        np.testing.assert_allclose(method(probe), expected, rtol=2e-14, atol=0)
    rng = np.random.Generator(np.random.PCG64(20261016))
    q = np.ascontiguousarray(10.0**rng.uniform(-5, 5, 10_000_000), dtype=np.float64)
    digest = hashlib.sha256(memoryview(q)).hexdigest()
    for _ in range(2):
        for method in methods.values():
            result = method(q)
            del result
    rows = []
    names = list(methods)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for repeat in range(10):
            shift = repeat % len(names)
            order = names[shift:] + names[:shift]
            for position, name in enumerate(order, start=1):
                start = time.perf_counter_ns()
                result = methods[name](q)
                elapsed = (time.perf_counter_ns() - start) / 1e9
                assert result.shape == q.shape and np.all(np.isfinite(result))
                del result
                rows.append({'repeat': repeat + 1, 'position': position,
                             'implementation': name, 'seconds': elapsed})
            print(f'Completed implementation round {repeat + 1}/10', flush=True)
    finally:
        if was_enabled:
            gc.enable()
    timings = {}
    for name in names:
        values = np.array([row['seconds'] for row in rows if row['implementation'] == name])
        timings[name] = {'median_seconds': float(np.median(values)),
                         'q25_seconds': float(np.quantile(values, 0.25)),
                         'q75_seconds': float(np.quantile(values, 0.75)),
                         'all_seconds': values.tolist()}
    comparisons = {}
    for numerator, denominator in [('Egg_shared', 'SR_cbrt'), ('Egg_no_reuse', 'SR_cbrt'),
                                    ('Egg_literal', 'SR_cbrt'), ('Egg_literal', 'SR_literal')]:
        ratios = np.array(timings[numerator]['all_seconds']) / np.array(timings[denominator]['all_seconds'])
        comparisons[numerator + '_over_' + denominator] = {
            'ratio_of_median_times': timings[numerator]['median_seconds'] / timings[denominator]['median_seconds'],
            'maximum_same_round_ratio': float(ratios.max())}
    report = {'environment': {'cpu': cpu_name(), 'python': platform.python_version(),
                             'numpy': np.__version__, 'platform': platform.platform()},
              'protocol': {'batch_size': len(q), 'dtype': 'float64', 'seed': 20261016,
                           'input_sha256': digest, 'q_distribution': 'uniform log10(q) in [-5,5]',
                           'warmups_per_method': 2, 'trials_per_method': 10,
                           'order': 'cyclic rotation; each method occupies each position twice',
                           'timer': 'perf_counter_ns; formula evaluation and temporary/result allocations included',
                           'excludes': 'input generation, warmups, correctness checks, result deallocation, file I/O'},
              'implementations': {'Egg_shared': 'shared cbrt(q), t*t, log1p(t)',
                                  'Egg_no_reuse': 'three cbrt(q) calls; two squares; log1p; no saved intermediate',
                                  'Egg_literal': 'q**(2/3) twice; q**(1/3); log(1+...)',
                                  'SR_cbrt': 'sqrt and cbrt, as in preceding benchmarks',
                                  'SR_literal': 'sqrt and **(1/3)'},
              'correctness': 'all methods match their literal formula to rtol=2e-14 on 257 probes',
              'timing': timings, 'comparisons': comparisons}
    folder = ROOT / 'data/timing_implementations_20261016'
    folder.mkdir(exist_ok=True)
    (folder / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (folder / 'trials.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'medians': {name: item['median_seconds'] for name, item in timings.items()},
                      'comparisons': comparisons}, indent=2))


if __name__ == '__main__':
    main()
