#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "domain.hpp"

static inline double wrap_periodic(double x, const double lbox)
{
  const double inv = 1.0 / lbox;
  x -= std::floor(x * inv) * lbox;
  if(x >= lbox) x -= lbox;
  if(x < 0.0) x += lbox;
  return x;
}

// A periodic axis accepts any periodic image of the coordinate.
// An open axis is left alone: a particle outside the domain may still reach inside with part of
// its stencil, so it must not be rejected on the coordinate alone.
static inline double axis_coord(const double x, const double lbox, const BoundaryType bc)
{
  return (bc == BoundaryType::Periodic) ? wrap_periodic(x, lbox) : x;
}

// Stencil of the assignment window, without any knowledge of the boundary condition.
// idx[] may fall outside [0, nmesh).
static inline int assign_axis_raw(const double xg, const int method, int idx[4], double w[4])
{
  // xg: coordinate in grid units (0..nmesh)

  if(method == 1) {
    // NGP
    const int ic = (int)(std::floor(xg + 0.5)); // need +0.5
    idx[0] = ic;
    w[0] = 1.0;
    return 1;
  }

  if(method == 2) {
    // CIC
    const int i0 = (int)(std::floor(xg));
    const double f = xg - (double)(i0);
    idx[0] = i0;
    idx[1] = i0 + 1;
    w[0] = 1.0 - f;
    w[1] = f;
    return 2;
  }

  if(method == 3) {
    // TSC
    const int ic = (int)(std::floor(xg + 0.5)); // need +0.5
    const double d = xg - (double)(ic);
    idx[0] = ic - 1;
    idx[1] = ic;
    idx[2] = ic + 1;

    w[0] = 0.5 * (0.5 - d) * (0.5 - d);
    w[1] = 0.75 - d * d;
    w[2] = 0.5 * (0.5 + d) * (0.5 + d);
    return 3;
  }

  if(method == 4) {
    // PCS (cubic B-spline, 4-point)
    const int i0 = (int)(std::floor(xg));
    const double u = xg - (double)(i0); // [0,1)
    idx[0] = i0 - 1;
    idx[1] = i0;
    idx[2] = i0 + 1;
    idx[3] = i0 + 2;

    w[0] = (1.0 / 6.0) * (1.0 - u) * (1.0 - u) * (1.0 - u);
    w[1] = (1.0 / 6.0) * (3.0 * u * u * u - 6.0 * u * u + 4.0);
    w[2] = (1.0 / 6.0) * (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0);
    w[3] = (1.0 / 6.0) * (u * u * u);
    return 4;
  }
  throw std::runtime_error("method must be 1(NGP), 2(CIC), 3(TSC), or 4(PCS)");
}

// Periodic: keep every stencil point and wrap the index.
// Open: drop the stencil points that fall outside the domain and leave the remaining weights
// untouched. The contribution that left the domain is discarded, never renormalized.
static inline int apply_axis_bc(const int raw_idx[4], const double raw_w[4], const int nin, const int nmesh,
                                const BoundaryType bc, int idx[4], double w[4])
{
  if(bc == BoundaryType::Periodic) {
    for(int i = 0; i < nin; i++) {
      idx[i] = (raw_idx[i] % nmesh + nmesh) % nmesh;
      w[i] = raw_w[i];
    }
    return nin;
  }

  int nout = 0;
  for(int i = 0; i < nin; i++) {
    if(raw_idx[i] < 0 || raw_idx[i] >= nmesh) continue;
    idx[nout] = raw_idx[i];
    w[nout] = raw_w[i];
    nout++;
  }
  return nout;
}

static inline int assign_axis_1d(const double xg, const int nmesh, const int method, const BoundaryType bc, int idx[4],
                                 double w[4])
{
  int raw_idx[4];
  double raw_w[4];
  const int nin = assign_axis_raw(xg, method, raw_idx, raw_w);
  return apply_axis_bc(raw_idx, raw_w, nin, nmesh, bc, idx, w);
}
