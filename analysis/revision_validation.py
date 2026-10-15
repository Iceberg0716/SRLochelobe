"""Reproduce revision metrics, the comparison figure, and independent checks.

Run from any directory with Python, NumPy, SciPy, and Matplotlib installed.
The original Q.npy and R.npy files are read only. The independent quadrature
is a new validation calculation, not the historical dataset-generation code.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def predictions(q):
    t = np.cbrt(q)
    return {
        "SR": np.cbrt(q / (7.6 * np.sqrt(q) + 1.85 * q + 8.8)),
        "SR_opt": np.cbrt(q / (7.593 * np.sqrt(q) + 1.845 * q + 8.813)),
        "Egg": 0.49 * t**2 / (0.6 * t**2 + np.log1p(t)),
        "Egg_opt1": 0.48547 * t**2 / (0.596745 * t**2 + np.log(0.999511 + t)),
        "Egg_opt2": 0.495277 * t**2 / (0.6088 * t**2 + np.log1p(t)),
    }


def metrics(y, predicted):
    error = np.abs(predicted - y)
    relative = error / y
    return {
        "MaxAE": float(error.max()),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(error.mean()),
        "MeanRel_percent": float(100 * relative.mean()),
        "MaxRel_percent": float(100 * relative.max()),
    }


def exact_radius(q, order):
    """Full Roche potential; integrate the connected lobe using symmetry."""
    Q = 1.0 / q

    def polynomial(z):
        return (1 + Q) * z**5 - (2 + 3 * Q) * z**4 + (1 + 3 * Q) * z**3 - z**2 + 2 * z - 1

    z0 = brentq(polynomial, 1e-12, 1 - 1e-12, xtol=1e-14)

    def omega(u, z, theta):
        return (
            1 / np.sqrt(u + z**2)
            + Q / np.sqrt(u + (1 - z)**2)
            + (1 + Q) / 2 * u * np.sin(theta)**2
            + (1 + Q) / 2 * z**2 - Q * z
        )

    critical = omega(0, z0, 0)
    z1 = brentq(lambda z: omega(0, z, 0) - critical, -z0, -1e-12, xtol=1e-14)
    nodes, weights = leggauss(order)
    zs = (z0 - z1) / 2 * nodes + (z0 + z1) / 2
    thetas = np.pi / 4 * (nodes + 1)
    total = 0.0
    for z, wz in zip(zs, weights):
        for theta, wt in zip(thetas, weights):
            # u=R^2. If z0^2 lies beyond both surface crossings, use
            # the potential minimum to bracket the connected inner branch.
            upper = z0**2
            if omega(upper, z, theta) >= critical:
                def derivative(u):
                    return (-0.5 / (u + z**2)**1.5
                            - 0.5 * Q / (u + (1-z)**2)**1.5
                            + (1+Q) / 2 * np.sin(theta)**2)

                bound = max(1.0, upper)
                while derivative(bound) <= 0:
                    bound *= 4
                upper = brentq(derivative, 0, bound, xtol=1e-14)
            u = brentq(lambda u: omega(u, z, theta) - critical, 0, upper, xtol=1e-14)
            total += wz * wt * u
    volume = 2 * total * (z0 - z1) / 2 * np.pi / 4
    return float(np.cbrt(3 * volume / (4 * np.pi)))


def large_q_limit():
    # Q -> 0, critical potential = 3/2; mu is cosine of polar angle
    # measured from the rotation axis. The surface radius is spherical.
    def integrand(mu):
        radius = brentq(lambda r: 1 / r + (1 - mu**2) * r**2 / 2 - 1.5, 2 / 3, 1, xtol=1e-14)
        return radius**3

    volume_ratio, error = quad(integrand, 0, 1, epsabs=1e-12, epsrel=1e-12)
    return float(np.cbrt(volume_ratio)), float(error)


def comparison_plot(q, radius):
    values = predictions(q)
    styles = {
        "SR": ("#d62728", "--", 2.1, "SR"),
        "SR_opt": ("#d62728", ":", 1.2, "SR optimized"),
        "Egg": ("#167ac6", "-", 2.1, "Eggleton"),
        "Egg_opt1": ("#78b6df", "--", 1.1, "Eggleton optimized 1"),
        "Egg_opt2": ("#78b6df", ":", 1.1, "Eggleton optimized 2"),
    }
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.1), constrained_layout=True)
    for row, lower in enumerate((False, True)):
        mask = (q >= 1e-3 * (1 - 1e-12)) & (q <= 3) if lower else np.ones(q.size, bool)
        x, y = q[mask], radius[mask]
        axes[row, 0].plot(x, y, color="#414141", lw=1.8, label="Numerical reference")
        # Draw the primary comparison last so thin variants cannot cover it.
        for key in ("Egg_opt1", "Egg_opt2", "SR_opt", "Egg", "SR"):
            color, linestyle, lw, label = styles[key]
            pred = values[key][mask]
            axes[row, 0].plot(x, pred, color=color, ls=linestyle, lw=lw, label=label)
            axes[row, 1].plot(x, 100 * np.abs(pred / y - 1), color=color, ls=linestyle, lw=lw)
        for ax in axes[row]:
            ax.grid(alpha=0.2)
            ax.set_xlabel(r"Mass ratio $q=M_1/M_2$")
            if not lower:
                ax.set_xscale("log")
                ax.set_xlim(1e-5, 1e5)
            else:
                ax.set_xlim(1e-3, 3)
        axes[row, 0].set_ylabel(r"Normalized radius $R_{\rm L}/a$")
        axes[row, 1].set_ylabel("Absolute relative error (%)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title(r"Reference samples: $10^{-5}\leq q\leq10^5$")
    axes[0, 1].set_title("Relative errors on the same samples")
    axes[1, 0].set_title(r"Intermediate range: $10^{-3}\leq q\leq3$")
    axes[1, 1].set_title("Relative errors in the intermediate range")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = dict(zip(labels, handles))
    order = ["Numerical reference", "SR", "SR optimized", "Eggleton",
             "Eggleton optimized 1", "Eggleton optimized 2"]
    axes[0, 0].legend([legend[label] for label in order], order, loc="lower right")
    fig.savefig(ROOT / "figures" / "Comparison_revised.png", dpi=240)
    plt.close(fig)


def main():
    q = np.load(ROOT / "Q.npy", allow_pickle=False)
    radius = np.load(ROOT / "R.npy", allow_pickle=False)
    assert q.shape == radius.shape == (1434,)
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(radius))
    assert np.all(q > 0) and np.all(radius > 0)
    report = {"numpy_version": np.__version__, "scipy_version": scipy.__version__}
    table_rows = []
    for name, lo, hi in (("full", 1e-5, 1e5), ("typical", 1e-3, 1e3)):
        mask = (q >= lo * (1 - 1e-12)) & (q <= hi)
        report[name] = {"N": int(mask.sum()), "metrics": {}}
        for model, predicted in predictions(q[mask]).items():
            result = metrics(radius[mask], predicted)
            report[name]["metrics"][model] = result
            table_rows.append({"domain": name, "N": int(mask.sum()), "formula": model, **result})
    with (DATA / "revision_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    comparison_plot(q[(q >= 1e-5 * (1 - 1e-12)) & (q <= 1e5)], radius[(q >= 1e-5 * (1 - 1e-12)) & (q <= 1e5)])

    fresh_q = np.logspace(-4.973, 4.967, 41)
    assert not np.any(np.isclose(fresh_q[:, None], q[None, :], rtol=1e-10, atol=0))
    reference_lo, reference_hi = [], []
    for index, qi in enumerate(fresh_q):
        reference_lo.append(exact_radius(float(qi), 48))
        reference_hi.append(exact_radius(float(qi), 96))
        if (index + 1) % 10 == 0:
            print(f"Independent reference: {index + 1}/41", flush=True)
    reference_lo, reference_hi = np.array(reference_lo), np.array(reference_hi)
    convergence = np.abs(reference_hi / reference_lo - 1)
    if convergence.max() > 1e-6:
        raise RuntimeError(f"Reference quadrature requires refinement: {convergence.max()}")
    report["independent"] = {
        "N": 41,
        "q_min": float(fresh_q.min()), "q_max": float(fresh_q.max()),
        "orders": [48, 96], "max_relative_quadrature_change": float(convergence.max()),
        "metrics": {key: metrics(reference_hi, value) for key, value in predictions(fresh_q).items()},
    }
    with (DATA / "independent_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["q", "radius_order48", "radius_order96", "relative_change"])
        writer.writerows(zip(fresh_q, reference_lo, reference_hi, convergence))
    limiting_radius, error = large_q_limit()
    report["large_q_limit"] = {
        "numerical_radius": limiting_radius, "integral_error_estimate": error,
        "SR": float(1.85**(-1 / 3)), "Egg": float(0.49 / 0.6),
        "SR_relative_error_percent": float(100 * abs(1.85**(-1 / 3) / limiting_radius - 1)),
        "Egg_relative_error_percent": float(100 * abs((0.49 / 0.6) / limiting_radius - 1)),
    }
    (DATA / "revision_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
