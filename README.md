# mass_assignment

A Python library for 3D mass assignment and velocity moment estimation using C++ and OpenMP.

## Features

- **Multiple Assignment Schemes**: Supports NGP, CIC, TSC, and PCS.
- **High-Order Moments**: Compute Density (0th order), Velocity (1st), Velocity Dispersion (2nd), Skewness (3rd), and Kurtosis (4th).
- **Optimized Performance**: Core routines implemented in C++ with OpenMP for efficient multi-threading.

## Installation

```bash
pip install mass-assignment
```


## Usage

1. Density Field

- Map particle masses to a 128^3 3D mesh.

```python
import numpy as np
import mass_assignment as ma

pos = np.random.uniform(0, 100, (1000000, 3))
mass = np.random.uniform(0.5, 1.5, 1000000)

# Compute density
# lbox: Box size, nmesh: Grid resolution
grid_density = ma.dens(pos, lbox=100.0, nmesh=128, method="TSC", mass=mass)
```

2. Velocity Dispersion Field

- Compute the 2nd-order moment tensor

```python
import numpy as np
import mass_assignment as ma

pos = np.random.uniform(0, 100, (1000000, 3))
vel = np.random.normal(0, 10, (1000000, 3))

# Returns an array of shape (nmesh, nmesh, nmesh, 6)
# Component order: xx, xy, xz, yy, yz, zz
dispersion_tensor = ma.sigma_norm(pos, vel, lbox=100.0, nmesh=128, method="TSC")
```


## API Parameters

- Assignment Schemes (`method`)
  You can pass the scheme as a string or an integer.
  - `NGP` (1): Nearest Grid Point
  - `CIC` (2): Cloud-In-Cell
  - `TSC` (3): Triangular Shaped Cloud
  - `PCS` (4): Piecewise Cubic Spline

- Mass (`mass`)
  - If mass is set to None (default), the library assumes uniform mass (weight = 1.0 for every particle).

- Data Precision (`dtype`)
  - `f4` or np.float32 (Default): Memory efficient and faster.
  - `f8` or np.float64: Recommended for higher numerical precision.

- Parallelization (`nthreads`)
  - 0 (Default): Automatically uses all available CPU threads via OpenMP.
  - N: Uses a specific number of threads.

- Normalization Mode (`norm_mode`)
  This parameter defines how high-order moments are normalized in `_norm` functions:
  - `diag_norm` (Default): Normalizes each tensor component by the mass in that cell. This corresponds to the Frobenius norm calculation for each independent component.
  - `tr_norm`: Normalizes the tensor using the trace-based scaling.


*Disclaimer*: The implementations for Skewness (3rd order) and Kurtosis (4th order) are currently untested.
