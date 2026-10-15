"""Run the supplied PySR search on all saved dimensionless benchmark pairs."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--check-data", action="store_true", help="Check inputs without importing PySR or starting Julia")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    q = np.load(ROOT / "Q.npy", allow_pickle=False)
    radius = np.load(ROOT / "R.npy", allow_pickle=False)
    if q.ndim != 1 or radius.shape != q.shape or q.size == 0:
        raise ValueError("Q.npy and R.npy must be paired, nonempty 1-D arrays.")
    if not np.all(np.isfinite(q) & (q > 0)) or not np.all(np.isfinite(radius) & (radius > 0)):
        raise ValueError("All inputs and targets must be finite and positive.")
    config = {
        "n_samples": int(q.size),
        "q_min": float(q.min()), "q_max": float(q.max()),
        "input": "q = M1/M2 (dimensionless)",
        "target": "R_L/a (dimensionless)",
        "maxsize": 20, "niterations": args.iterations,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sqrt", "cbrt"],
        "elementwise_loss": "loss(x, y) = (x - y)^2",
    }
    print(json.dumps(config, indent=2))
    if args.check_data:
        return

    from pysr import PySRRegressor

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    model = PySRRegressor(
        maxsize=config["maxsize"], niterations=config["niterations"],
        binary_operators=config["binary_operators"],
        unary_operators=config["unary_operators"],
        elementwise_loss=config["elementwise_loss"],
        output_directory=str(output),
    )
    model.fit(q.reshape(-1, 1), radius)
    prediction = model.predict(q.reshape(-1, 1))
    residual = np.sum((prediction - radius)**2)
    total = np.sum((radius - radius.mean())**2)
    config.update({"pysr_version": version("pysr"), "selected_expression": str(model.sympy()),
                   "training_R2": float(1 - residual / total)})
    (output / "search_summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.save(output / "training_predictions.npy", prediction)
    print("Selected expression:", model.sympy())
    print("Training R2:", config["training_R2"])


if __name__ == "__main__":
    main()
