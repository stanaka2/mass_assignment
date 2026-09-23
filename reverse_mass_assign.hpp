#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <omp.h>

#include "domain.hpp"
#include "assignment_kernel.hpp"
#include "ptcl.hpp"

// Uses the same stencil and boundary logic as the deposition, so that an open axis behaves
// consistently in both directions.
template <typename T, typename U>
static void scalar_from_mesh(const Ptcl<T> &p, const U *mesh, T *out, const GridSpec &grid, const int method,
                             const int nthreads)
{
  if(nthreads > 0) omp_set_num_threads(nthreads);
  {
    py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
    for(int64_t ip = 0; ip < p.n; ip++) {
      const double x = axis_coord(p.x(ip), grid.lbox(0), grid.bc(0));
      const double y = axis_coord(p.y(ip), grid.lbox(1), grid.bc(1));
      const double z = axis_coord(p.z(ip), grid.lbox(2), grid.bc(2));

      int idx_x[4], idx_y[4], idx_z[4];
      double w_x[4], w_y[4], w_z[4];

      const int nax = assign_axis_1d(x * grid.inv_dx(0), grid.n[0], method, grid.bc(0), idx_x, w_x);
      const int nay = assign_axis_1d(y * grid.inv_dx(1), grid.n[1], method, grid.bc(1), idx_y, w_y);
      const int naz = assign_axis_1d(z * grid.inv_dx(2), grid.n[2], method, grid.bc(2), idx_z, w_z);

      double sum = 0.0;

      for(int ix = 0; ix < nax; ix++) {
        const double wx = w_x[ix];
        for(int iy = 0; iy < nay; iy++) {
          const double wxy = wx * w_y[iy];
          for(int iz = 0; iz < naz; iz++) {
            const double w = wxy * w_z[iz];
            const int64_t idx = idx3(idx_x[ix], idx_y[iy], idx_z[iz], grid.n);
            sum += w * (double)mesh[idx];
          }
        }
      }
      out[ip] = (T)sum;
    } // particles loop
  } // gil release
}

template <typename T, typename U>
static py::object mesh_to_ptcl_impl(const Ptcl<T> &p, const U *mesh, const GridSpec &grid, const int method,
                                    const int nthreads)
{
  py::array_t<T> p_arr({p.n});
  T *out = static_cast<T *>(p_arr.request().ptr);
  std::fill_n(out, p.n, static_cast<T>(0.0));
  scalar_from_mesh(p, mesh, out, grid, method, nthreads);
  return p_arr;
}
