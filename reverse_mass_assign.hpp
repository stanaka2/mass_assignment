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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mass_assign.hpp"

template <typename T, typename U>
static void scalar_from_mesh(const T *pos, const U *mesh, T *out, const int64_t n, const double lbox, const int nmesh,
                             const int method, const int nthreads)
{
#ifdef _OPENMP
  if(nthreads > 0) omp_set_num_threads(nthreads);
#endif

  const double inv_dx = (double)nmesh / lbox;
  {
    py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
    for(int64_t ip = 0; ip < n; ip++) {
      const double x = wrap_periodic((double)pos[3 * ip + 0], lbox);
      const double y = wrap_periodic((double)pos[3 * ip + 1], lbox);
      const double z = wrap_periodic((double)pos[3 * ip + 2], lbox);

      int idx_x[4], idx_y[4], idx_z[4];
      double w_x[4], w_y[4], w_z[4];

      const int nax = assign_axis_1d(x * inv_dx, nmesh, method, idx_x, w_x);
      const int nay = assign_axis_1d(y * inv_dx, nmesh, method, idx_y, w_y);
      const int naz = assign_axis_1d(z * inv_dx, nmesh, method, idx_z, w_z);

      double s = 0.0;

      for(int ix = 0; ix < nax; ix++) {
        const double wx = w_x[ix];
        for(int iy = 0; iy < nay; iy++) {
          const double wxy = wx * w_y[iy];
          for(int iz = 0; iz < naz; iz++) {
            const double w = wxy * w_z[iz];
            const int64_t idx =
                (int64_t)idx_z[iz] + (int64_t)nmesh * ((int64_t)idx_y[iy] + (int64_t)nmesh * (int64_t)idx_x[ix]);
            s += w * (double)mesh[idx];
          }
        }
      }
      out[ip] = (T)s;
    } // particles loop
  } // gil release
}

template <typename T, typename U>
static py::object mesh_to_ptcl_impl(const T *pos, const U *mesh, const double lbox, const int64_t n, const int nmesh,
                                    const int method, const int nthreads)
{
  py::array_t<T> p_arr({n});
  T *out = static_cast<T *>(p_arr.request().ptr);
  std::fill_n(out, n, static_cast<T>(0.0));
  scalar_from_mesh(pos, mesh, out, n, lbox, nmesh, method, nthreads);
  return p_arr;
}
