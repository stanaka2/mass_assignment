# mass_assignment

A small Python extension module for single-process mass assignment on a 3D mesh with optional OpenMP parallelization.

This package provides:
- `dens`: mass density field
- `velc`: mass-weighted mean velocity field
- `sigma`: mass-weighted velocity dispersion tensor (second order moment)

Supported assignment schemes:
- 1: NGP
- 2: CIC
- 3: TSC
- 4: PCS (cubic B-spline)

## Installation

```bash
pip install mass-assignment
```
