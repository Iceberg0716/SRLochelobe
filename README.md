# SRLochelobe

Numerical data and reproducible programs accompanying **Symbolic Regression Fitting for Roche Lobe Radius**.

This repository provides the saved Roche-lobe benchmark, a standalone implementation of its numerical generation algorithm, the PySR search configuration, and the revised accuracy and physical-limit checks. It concerns the classical circular, synchronously rotating Roche model.

## Data and conventions

- The input is the dimensionless mass ratio **q = M1/M2**. The target is the dimensionless volume-equivalent radius **R_L/a**, for the lobe surrounding M1.
- `Q.npy` contains the input q values despite its uppercase filename. The Roche potential uses the reciprocal **Q = M2/M1 = 1/q** internally.
- `R.npy` contains the corresponding normalized radii. Both arrays contain 1434 entries, with 1432 distinct mass ratios; q=1 and q=100 each occur twice and are retained.
- The stored mass-ratio range is approximately 1e-5 to 3.6909169e5. All 1434 pairs enter the symbolic-regression search.
- The main error comparison uses 1364 rows with 1e-5 <= q <= 1e5. The restricted comparison uses 914 rows with 1e-3 <= q <= 1e3.
- Sampling is non-uniform in logarithmic space, with greater density between 1 and 100. The saved grid is supplied explicitly; it is not replaced with a guessed uniform grid. Means are arithmetic averages over the selected rows.
- `data/benchmark.csv` provides the same input/target pairs in a portable text format.

## Install

Use standard Python 3.13 and an isolated environment. The numerical checks were run with NumPy 2.2.6, SciPy 1.16.2, and Matplotlib 3.10.6.

```bash
python -m venv .venv
# Activate .venv for your operating system, then:
python -m pip install -r requirements.txt
```

On Windows, `py -3.13` can be used to select the standard Python installation. The free-threaded interpreter needs compatible binary packages.

## Recompute numerical radii

`datagenerate.py` uses the full Roche potential. It locates the inner Lagrangian point and the opposite intersection of its critical equipotential, solves for the cylindrical surface radius, integrates R(theta,z)^2/2 over z and theta, and converts the resulting volume to R_L/a.

The SciPy `newton` calls do not receive derivatives and therefore use the secant method, with their default `tol=1.48e-8`, `rtol=0`, and `maxiter=50`. `dblquad` uses its default absolute and relative tolerance parameters of 1.49e-8. These settings are not guaranteed error bounds on the final radius.

Check two reference values first:

```bash
python datagenerate.py --q 1 100 --output outputs/spotcheck.csv
```

The saved reference radii are 0.3798632409273554 and 0.7181690286376814, respectively. To process the full saved grid:

```bash
python datagenerate.py --output outputs/recomputed_benchmark.csv
```

The complete calculation can take considerably longer than a spot check. Each row records its mass ratio, radius, integration error estimate, and status. Numerical failures and warnings are recorded explicitly. Existing output files are protected unless `--overwrite` is passed. The original arrays are never overwritten. Differences across numerical-library versions remain possible; the historical library versions were not recorded.

## Symbolic regression

The standalone search reads the two arrays directly. It retains the supplied arithmetic/root operator set, maximum complexity 20, 500 search iterations, and elementwise squared-error loss. All samples are used without a separate train/test split.

First check the inputs without importing PySR:

```bash
python sr.py --check-data
```

Install the optional search dependencies and run:

```bash
python -m pip install -r requirements-sr.txt
python sr.py
```

PySR may download and initialize its Julia environment on first use. The public environment specifies PySR 1.5.9; this is not a claim about the unrecorded historical version. Search is stochastic and the original seed and candidate history were not saved, so a rerun need not return the identical expression. The remaining search and selection settings use the installed library defaults. New summaries and predictions are saved under `outputs/`.

The displayed SR expression is:

```text
R_SR(q) = [q / (7.6 sqrt(q) + 1.85 q + 8.8)]^(1/3)
```

This is a dimensionless empirical approximation to a known Roche-model relation. It is not a new physical law or a first-principles derivation of the coefficients.

## Reproduce the revised comparisons and independent checks

```bash
python analysis/revision_validation.py
```

This recomputes the five displayed formulas with their coefficients fixed, regenerates the comparison figure, and calculates 41 new reference points absent from the training arrays. It produces:

- `data/revision_metrics.csv`: error metrics in the main and restricted domains.
- `data/independent_validation.csv`: new reference radii at quadrature orders 48 and 96.
- `data/revision_validation.json`: metrics, convergence comparisons, the large-q limiting-model calculation, and numerical-library versions.
- `figures/Comparison_revised.png`: normalized-radius and relative-error comparisons.

The independent calculation uses bracketed root finding and tensor-product Gauss-Legendre quadrature rather than the historical secant/adaptive-quadrature procedure. It checks the same full Roche model using a different numerical implementation. On the 41 new points, increasing the quadrature order from 48 to 96 changes radii by at most approximately 9.44e-8 in relative terms. This comparison is a convergence check, not a rigorous error bound or a proof of extrapolation accuracy.

The limiting model as q tends to infinity gives R_L/a approximately 0.8148856768; the SR expression tends to 0.8145969078. Full details and error metrics are in the numerical output files.

## Evaluation-speed benchmark

```bash
python analysis/benchmark_evaluation.py
```

Both formulas evaluate the same contiguous array of 10 million float64 mass ratios, sampled uniformly in log10(q) on [-5, 5] with PCG64 seed 20261016. Each formula receives two full-size warm-up calls and ten timed calls, with alternating order. The Eggleton implementation computes cbrt(q) once, reuses its square, and uses log1p. Timing includes formula evaluation and temporary/result-array allocations, excluding input generation, warm-up, validation, file output, and result deallocation. The script checks agreement with the literal formulas before measuring.

The recorded experiment used an Intel Core i7-13650HX, Windows 11, Python 3.13.7, and NumPy 2.2.6 on 2026-10-16:

| Formula | Median (s) | Interquartile range (s) |
| --- | --- | --- |
| Eggleton | 0.33071655 | 0.3138468–0.337934175 |
| SR | 0.25778330 | 0.255482825–0.261442975 |

The ratio of median times is 1.2829, corresponding to a 22.1% reduction for SR. The script writes `data/timing_benchmark.json` with the protocol and environment, and `data/timing_trials.csv` with all 20 trial times. Batch size, repeats, warm-ups, seed, and output directory can be changed through command-line options. Running the same protocol on another system produces a new timing record.

## Method references

- Eggleton (1983), [Approximations to the radii of Roche lobes](https://doi.org/10.1086/160960).
- Cranmer (2023), [Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl](https://arxiv.org/abs/2305.01582).
- Virtanen et al. (2020), [SciPy 1.0](https://doi.org/10.1038/s41592-019-0686-2).

The paper manuscript and referee correspondence are not included in this numerical-code repository.
