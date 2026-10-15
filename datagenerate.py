"""Recompute volume-equivalent Roche-lobe radii with the original algorithm.

Input q=M1/M2 is dimensionless; the potential uses Q=1/q internally.
The saved input grid is used rather than guessing the original sampling code.
Original numerical defaults and initial guesses are retained.
"""

import argparse
import csv
from pathlib import Path
import warnings

import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import newton

ROOT = Path(__file__).resolve().parent


def roche_radius(q):
    """Return R_L/a and dblquad's volume error estimate for one q=M1/M2."""
    if not np.isfinite(q) or q <= 0:
        raise ValueError("Mass ratio must be finite and positive.")
    Q = 1.0 / q

    def lagrange_equation(z):
        return (1 + Q) * z**5 - (2 + 3 * Q) * z**4 + (1 + 3 * Q) * z**3 - z**2 + 2 * z - 1

    def omega(radius, z, theta):
        return (
            np.power(radius**2 + z**2, -0.5)
            + Q * np.power(radius**2 + (1 - z)**2, -0.5)
            + (1 + Q) / 2 * radius**2 * np.sin(theta)**2
            + (1 + Q) / 2 * z**2 - Q * z
        )

    z0 = newton(lagrange_equation, x0=0.5)
    critical = 1 / z0 + Q / (1 - z0) + (1 + Q) / 2 * z0**2 - Q * z0
    z1 = newton(lambda z: omega(0, z, 0) - critical, -z0)

    def integrand(theta, z):
        initial_radius = newton(lambda radius: omega(radius, z, 0) - critical, 0.5)
        radius = newton(lambda radius: omega(radius, z, theta) - critical, initial_radius)
        return 0.5 * radius**2

    volume, error = dblquad(integrand, z1, z0, 0, 2 * np.pi)
    return float(np.power(3 * volume / (4 * np.pi), 1 / 3)), float(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass-ratios", type=Path, default=ROOT / "Q.npy")
    parser.add_argument("--q", type=float, nargs="+", help="Explicit mass ratios instead of Q.npy")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N input ratios")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "recomputed_benchmark.csv")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ratios = np.asarray(args.q, dtype=float) if args.q else np.load(args.mass_ratios, allow_pickle=False)
    if ratios.ndim != 1 or ratios.size == 0 or not np.all(np.isfinite(ratios) & (ratios > 0)):
        parser.error("Expected a nonempty one-dimensional array of finite positive mass ratios.")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        ratios = ratios[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    with args.output.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["q", "R_L_over_a", "volume_error_estimate", "status"])
        for index, q in enumerate(ratios):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    radius, error = roche_radius(float(q))
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError("Non-positive or non-finite radius returned")
                status = "ok" if not caught else "warning: " + " | ".join(str(w.message) for w in caught)
                writer.writerow([float(q), radius, error, status])
                successes += 1
            except (RuntimeError, ValueError, OverflowError) as error:
                writer.writerow([float(q), "", "", "failed: " + str(error)])
            file.flush()
            print(f"Completed {index + 1}/{len(ratios)}; successful: {successes}", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
