# mass_assignment

A Python library for 3D mass assignment and velocity moment estimation using C++ and OpenMP.

## Features

- **Multiple Assignment Schemes**: Supports NGP, CIC, TSC, and PCS.
- **High-Order Moments**: Compute Density (0th order), Velocity (1st), Velocity Dispersion (2nd), Skewness (3rd), and Kurtosis (4th).
- **Optimized Performance**: Core routines implemented in C++ with OpenMP for efficient multi-threading.

The established `dens`, `velc`, `sigma`, `skewness`, and `kurtosis` APIs remain
available. `moment2`, `moment3`, and `moment4` are clearer aliases for the same
normalized central moments and match the snapshot dataset names used by
`vlasov_cpp` and `nbody_cpp`.

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
grid_density = ma.dens(pos, nmesh=128, lbox=100.0, method="TSC", mass=mass)
```

`dens` returns the summed mass of each cell, not the density contrast. Two-point statistics are
usually wanted for the contrast, which is one line away:

```python
rho = ma.dens(pos, nmesh=128, lbox=100.0, method="TSC", mass=mass)
delta = rho / rho.mean() - 1.0
```

Feeding `rho` straight into a power spectrum is not an error and gives a spectrum dominated by the
mean, so the step is worth making explicit.

- A scalar `nmesh` or `lbox` means the same value on all three axes. Both also accept a length-3
  sequence, so the mesh and the box need not be cubic.

```python
grid_density = ma.dens(
    pos,
    nmesh=(128, 64, 32),
    lbox=(100.0, 50.0, 25.0),
    method="TSC",
    mass=mass,
    bc=("periodic", "open", "periodic"),
)
```

2. Velocity Dispersion Field

- Compute the 2nd-order moment tensor

```python
import numpy as np
import mass_assignment as ma

pos = np.random.uniform(0, 100, (1000000, 3))
vel = np.random.normal(0, 10, (1000000, 3))

# Returns the 6 independent components as a tuple, each of shape (nmesh, nmesh, nmesh)
# Component order: xx, xy, xz, yy, yz, zz
sxx, sxy, sxz, syy, syz, szz = ma.sigma(pos, vel, nmesh=128, lbox=100.0, method="TSC")

# np.asarray() stacks them into (6, nmesh, nmesh, nmesh) if you want one array
dispersion_tensor = np.asarray(ma.sigma(pos, vel, nmesh=128, lbox=100.0, method="TSC"))

# sigma_norm returns a single scalar field (nmesh, nmesh, nmesh) instead
dispersion_norm = ma.sigma_norm(pos, vel, nmesh=128, lbox=100.0, method="TSC")
```

The default output remains `float32` for compatibility with 1.0.x. Select
`dtype=np.float64` when a small velocity dispersion sits on a large bulk
velocity, or whenever fourth-order accuracy matters:

```python
m4 = ma.moment4(
    pos,
    vel,
    nmesh=128,
    lbox=100.0,
    method="TSC",
    mass=mass,
    dtype=np.float64,
)

# Equivalent order-selecting interface
m4 = ma.central_moment(pos, vel, order=4, nmesh=128, lbox=100.0, dtype=np.float64)
assert ma.moment_components(4) == (
    "xxxx", "xxxy", "xxxz", "xxyy", "xxyz", "xxzz",
    "xyyy", "xyyz", "xyzz", "xzzz", "yyyy", "yyyz",
    "yyzz", "yzzz", "zzzz",
)
```

Despite their historical names, `skewness` and `kurtosis` return third- and
fourth-order normalized central moments, not dimensionless standardized
skewness or excess kurtosis. They are retained unchanged for compatibility;
new code may prefer `moment3` and `moment4`.

- `velc`, `sigma`, `skewness` and `kurtosis` return one array per component. Keeping the
  components in separate allocations matters: a single stacked block would place them exactly
  `nx*ny*nz` elements apart, which maps them onto the same cache set for a power-of-two mesh.

3. Sample a mesh field at particle positions

- `mesh_to_ptcl` interpolates a scalar field defined on a regular 3D grid to arbitrary particle positions.

```python
import numpy as np
import mass_assignment as ma

nmesh = 128
lbox = 1.0

# Example scalar field on the mesh (nmesh, nmesh, nmesh)
mesh = np.random.randn(nmesh, nmesh, nmesh).astype(np.float32)

# Particle positions (N,3) in [0, lbox)
pos = np.random.rand(200000, 3).astype(np.float32) * lbox
val = ma.mesh_to_ptcl(pos, mesh, lbox=lbox, method="TSC", nthreads=0)
print(val.shape)  # (N,)
```

- The grid is taken from the shape of `mesh`, so a non-cubic mesh works as it is.

4. Mesh gradient

```python
gx, gy, gz = ma.mesh_diff(
    mesh,
    lbox=(1.0, 2.0, 0.5),
    order=4,
    nthreads=0,
    bc=("periodic", "open", "periodic"),
)
```

- `order` is 2, 4, 6 or 8. A periodic axis uses a centered stencil; an open axis switches to a
  shifted stencil near the edges so that the requested order is kept there. An open axis therefore
  needs at least `order + 1` cells.


## API Parameters

- Assignment Schemes (`method`)
  You can pass the scheme as a string or an integer.
  - `NGP` (1): Nearest Grid Point
  - `CIC` (2): Cloud-In-Cell
  - `TSC` (3): Triangular Shaped Cloud
  - `PCS` (4): Piecewise Cubic Spline

- Mass (`mass`)
  - If mass is set to None (default), the library assumes uniform mass (weight = 1.0 for every particle).

- Grid (`nmesh`, `lbox`)
  - A scalar applies to all three axes. A length-3 sequence sets each axis separately.
  - `lbox` is in the same units as `pos`, and only sets the scale the coordinates are divided by.
    It is a declaration, not something the library can check: positions normalized to [0,1) with
    `lbox=100` are read as a box whose particles all sit in one corner, and no error is raised.
  - Positions normalized to [0,1) are honestly described by `lbox=1`, and the mesh that comes out is
    the same one the physical coordinates would give. The physical length is a separate statement
    and is made where it is needed. `field2pt` takes it as `lbox=` next to the `Grid`, which then
    supplies the assignment order and the mesh shape but not the box.

- Boundary Condition (`bc`, keyword only)
  - `periodic` (default): any periodic image of a coordinate is accepted and the stencil wraps.
  - `open`: coordinates are not wrapped and the part of the cloud that leaves the domain is
    discarded. The remaining weights are left as they are and never renormalized, so a particle
    on the boundary deposits less than its full mass.
  - A string applies to all three axes; a length-3 sequence sets each axis separately.
  - This is the boundary of the global physical domain, not the edge of a local array.


- Parallelization (`nthreads`)
  - 0 (Default): Automatically uses all available CPU threads via OpenMP.
  - N: Uses a specific number of threads.

- Output precision (`dtype`, keyword only)
  - `np.float32` (default): preserves the 1.0.x return dtype and uses less memory.
  - `np.float64`: accumulates and returns fields in double precision.

- Normalization Mode (`norm_mode`)
  This parameter defines how high-order moments are normalized in `_norm` functions:
  - `diag_norm` (Default): Normalizes each tensor component by the mass in that cell. This corresponds to the Frobenius norm calculation for each independent component.
  - `tr_norm`: Normalizes the tensor using the trace-based scaling.


## Grid

The mesh geometry and the assignment settings do not usually change within one analysis, and
repeating them on every call is a chance for them to drift apart with nothing raised. A `Grid`
holds them once and reports them back.

```python
g = ma.Grid(nmesh=512, lbox=1000.0, method="TSC", bc="periodic", nthreads=8)

rho = g.dens(pos)
vx, vy, vz = g.velc(pos, vel)
sigma = g.sigma(pos, vel)

g.nmesh    # (512, 512, 512)
g.lbox     # (1000.0, 1000.0, 1000.0)
g.method   # "TSC"
g.p        # 3, the assignment order
g.bc       # ("periodic", "periodic", "periodic")
```

Whatever consumes the mesh needs `lbox` and the assignment order to undo the assignment window, so
handing it the `Grid` keeps the two steps from disagreeing.

```python
import field2pt as f2

res = f2.power_spectrum(rho / rho.mean() - 1.0, grid=g)
```

Nothing here depends on that package: it reads `.lbox`, `.p` and `.nmesh` off the object it is
given. The module level functions are unchanged; a `Grid` only groups their settings.

### Shot noise

The Poisson term of the density contrast is `V * sum(w^2) / sum(w)^2`, which is `V / N` for equal
weights. Both halves are already at hand: the volume from the `Grid`, the rest from the particles
that were deposited.

```python
rho = g.dens(pos, mass=mass)
sn = g.shotnoise(pos, mass=mass)
```

`shotnoise` takes no state from the deposit, so a `Grid` can be used for several particle sets
without any of them being remembered. This is the leading term; close to the Nyquist frequency the
aliased images make the true shot noise k-dependent, and interlacing is what removes that.

The volume is this `Grid`'s box unless `lbox` says otherwise. The override is there because this
`lbox` may be a normalization rather than a length, in which case the shot noise comes back in
those same units:

```python
g = ma.Grid(nmesh=512, lbox=1.0, method="TSC")   # positions in [0,1)
sn = g.shotnoise(pos, lbox=1000.0)               # Mpc/h, to match the spectrum
```

Safer still is not to carry the number across at all. `field2pt` takes `npart=` or `weights=` and
works the volume out from the box it is measuring in, which cannot disagree with itself.

## Moment Component Order

```
m2: xx, xy, xz, yy, yz, zz
m3: xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
m4: xxxx, xxxy, xxxz, xxyy, xxyz, xxzz,
    xyyy, xyyz, xyzz, xzzz,
    yyyy, yyyz, yyzz, yzzz, zzzz
```

Second and higher moments are accumulated about the mean velocity of each cell instead of being
recovered by subtracting large raw moments. The local validation covers every one of the 35
components against direct NumPy accumulation, including large bulk motion and unequal weights.
