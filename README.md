# Poisson Matching for Coupling Metropolis–Hastings Chains

This repository contains implementations of Poisson Matching and maximal coupling for Random-Walk Metropolis–Hastings chains with Gaussian and Student's $begin:math:text$t$end:math:text$ target distributions.

## Requirements

- Python 3
- PyTorch >= 2.10.0

## Repository Structure

```text
.
├── rwmh_gaussian/
├── rwmh_tstudent/
└── README.md
```

## Running Experiments

From the repository root, run (for maximal coupling baselines):

```bash
cd rwmh_gaussian
python3 maximal_run_gaussian_joint.py --config config/d8_nchain32.yaml
```

or (for Poisson Matching baselines)

```bash
cd rwmh_gaussian
python3 pmc_run_gaussian.py --config config/d8_nchain32.yaml
```

Replace `rwmh_gaussian` with `rwmh_tstudent` to run the corresponding Student's experiments.
