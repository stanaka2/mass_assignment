#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#include <omp.h>

#include "domain.hpp"

// Centered coefficients for the first derivative. c[s-1] multiplies (f[i+s] - f[i-s]).
template <int ACC>
struct FD;

template <>
struct FD<2> {
  static constexpr int r = 1;
  static constexpr std::array<double, r> c = {0.5}; // 1/2
};

template <>
struct FD<4> {
  static constexpr int r = 2;
  static constexpr std::array<double, r> c = {2.0 / 3.0, -1.0 / 12.0};
};

template <>
struct FD<6> {
  static constexpr int r = 3;
  static constexpr std::array<double, r> c = {3.0 / 4.0, -3.0 / 20.0, 1.0 / 60.0};
};

template <>
struct FD<8> {
  static constexpr int r = 4;
  static constexpr std::array<double, r> c = {4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0};
};

// Shifted coefficients used near an open boundary, so that the requested order is kept there.
// Row d is for a cell at distance d from the left edge and multiplies f[i-d+k], k = 0..ACC.
// The right edge uses the same rows mirrored and negated.
template <int ACC>
struct OneSidedFD;

template <>
struct OneSidedFD<2> {
  static constexpr int w = 3;
  static constexpr std::array<std::array<double, w>, 1> c = {{
      {-3.0 / 2.0, 2.0, -1.0 / 2.0},
  }};
};

template <>
struct OneSidedFD<4> {
  static constexpr int w = 5;
  static constexpr std::array<std::array<double, w>, 2> c = {{
      {-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0},
      {-1.0 / 4.0, -5.0 / 6.0, 3.0 / 2.0, -1.0 / 2.0, 1.0 / 12.0},
  }};
};

template <>
struct OneSidedFD<6> {
  static constexpr int w = 7;
  static constexpr std::array<std::array<double, w>, 3> c = {{
      {-49.0 / 20.0, 6.0, -15.0 / 2.0, 20.0 / 3.0, -15.0 / 4.0, 6.0 / 5.0, -1.0 / 6.0},
      {-1.0 / 6.0, -77.0 / 60.0, 5.0 / 2.0, -5.0 / 3.0, 5.0 / 6.0, -1.0 / 4.0, 1.0 / 30.0},
      {1.0 / 30.0, -2.0 / 5.0, -7.0 / 12.0, 4.0 / 3.0, -1.0 / 2.0, 2.0 / 15.0, -1.0 / 60.0},
  }};
};

template <>
struct OneSidedFD<8> {
  static constexpr int w = 9;
  static constexpr std::array<std::array<double, w>, 4> c = {{
      {-761.0 / 280.0, 8.0, -14.0, 56.0 / 3.0, -35.0 / 2.0, 56.0 / 5.0, -14.0 / 3.0, 8.0 / 7.0, -1.0 / 8.0},
      {-1.0 / 8.0, -223.0 / 140.0, 7.0 / 2.0, -7.0 / 2.0, 35.0 / 12.0, -7.0 / 4.0, 7.0 / 10.0, -1.0 / 6.0, 1.0 / 56.0},
      {1.0 / 56.0, -2.0 / 7.0, -19.0 / 20.0, 2.0, -5.0 / 4.0, 2.0 / 3.0, -1.0 / 4.0, 2.0 / 35.0, -1.0 / 168.0},
      {-1.0 / 168.0, 1.0 / 14.0, -1.0 / 2.0, -9.0 / 20.0, 5.0 / 4.0, -1.0 / 2.0, 1.0 / 6.0, -1.0 / 28.0, 1.0 / 280.0},
  }};
};

// Offsets and weights of the first-derivative stencil at index i of an axis with n cells.
// A periodic axis always uses the centered stencil. An open axis uses it in the interior and a
// shifted stencil within r cells of either edge.
template <int ACC>
static inline int fd_stencil(const int i, const int n, const BoundaryType bc, int off[ACC + 1], double c[ACC + 1])
{
  constexpr int r = FD<ACC>::r;

  if(bc == BoundaryType::Periodic || (i >= r && i < n - r)) {
    int k = 0;
    for(int s = 1; s <= r; s++) {
      off[k] = -s;
      c[k] = -FD<ACC>::c[s - 1];
      k++;
      off[k] = s;
      c[k] = FD<ACC>::c[s - 1];
      k++;
    }
    return k;
  }

  constexpr int w = OneSidedFD<ACC>::w;

  if(i < r) {
    for(int k = 0; k < w; k++) {
      off[k] = k - i;
      c[k] = OneSidedFD<ACC>::c[i][k];
    }
    return w;
  }

  const int d = n - 1 - i;
  for(int k = 0; k < w; k++) {
    off[k] = d - k;
    c[k] = -OneSidedFD<ACC>::c[d][k];
  }
  return w;
}

// An open axis never leaves the domain by construction, so only a periodic axis wraps.
static inline int fd_index(const int i, const int off, const int n, const BoundaryType bc)
{
  const int j = i + off;
  return (bc == BoundaryType::Periodic) ? ((j % n + n) % n) : j;
}

template <int ACC, typename T>
static void build_grad_mesh(const T *mesh, T *retx, T *rety, T *retz, const GridSpec &grid, const int nthreads)
{
  static_assert(ACC == 2 || ACC == 4 || ACC == 6 || ACC == 8, "ACC must be 2,4,6,8");

  for(int j = 0; j < 3; j++) {
    if(grid.bc(j) == BoundaryType::Open && grid.n[j] < ACC + 1)
      throw std::runtime_error("an open axis needs at least order+1 cells for the boundary stencils");
  }

  if(nthreads > 0) omp_set_num_threads(nthreads);

  const int nx = grid.n[0];
  const int ny = grid.n[1];
  const int nz = grid.n[2];

  {
    py::gil_scoped_release release;

#pragma omp parallel for collapse(3) schedule(static)
    for(int i = 0; i < nx; i++) {
      for(int j = 0; j < ny; j++) {
        for(int k = 0; k < nz; k++) {

          int off[ACC + 1];
          double c[ACC + 1];

          // X derivative
          double sx = 0.0;
          int ns = fd_stencil<ACC>(i, nx, grid.bc(0), off, c);
          for(int s = 0; s < ns; s++) {
            const int ii = fd_index(i, off[s], nx, grid.bc(0));
            sx += c[s] * (double)mesh[idx3(ii, j, k, grid.n)];
          }

          // Y derivative
          double sy = 0.0;
          ns = fd_stencil<ACC>(j, ny, grid.bc(1), off, c);
          for(int s = 0; s < ns; s++) {
            const int jj = fd_index(j, off[s], ny, grid.bc(1));
            sy += c[s] * (double)mesh[idx3(i, jj, k, grid.n)];
          }

          // Z derivative
          double sz = 0.0;
          ns = fd_stencil<ACC>(k, nz, grid.bc(2), off, c);
          for(int s = 0; s < ns; s++) {
            const int kk = fd_index(k, off[s], nz, grid.bc(2));
            sz += c[s] * (double)mesh[idx3(i, j, kk, grid.n)];
          }

          const int64_t idx = idx3(i, j, k, grid.n);

          retx[idx] = (T)(sx * grid.inv_dx(0));
          rety[idx] = (T)(sy * grid.inv_dx(1));
          retz[idx] = (T)(sz * grid.inv_dx(2));
        }
      }
    }
  } // gil release
}

template <typename T>
static py::object mesh_diff_impl(const T *mesh, const GridSpec &grid, const int order, const int nthreads)
{
  const int64_t size = grid.size();
  py::array_t<T> diff_x({grid.n[0], grid.n[1], grid.n[2]}), diff_y({grid.n[0], grid.n[1], grid.n[2]}),
      diff_z({grid.n[0], grid.n[1], grid.n[2]});
  T *di_x = static_cast<T *>(diff_x.request().ptr);
  T *di_y = static_cast<T *>(diff_y.request().ptr);
  T *di_z = static_cast<T *>(diff_z.request().ptr);
  std::fill_n(di_x, size, 0.0);
  std::fill_n(di_y, size, 0.0);
  std::fill_n(di_z, size, 0.0);

  switch(order) {
  case 2:
    build_grad_mesh<2>(mesh, di_x, di_y, di_z, grid, nthreads);
    break;
  case 4:
    build_grad_mesh<4>(mesh, di_x, di_y, di_z, grid, nthreads);
    break;
  case 6:
    build_grad_mesh<6>(mesh, di_x, di_y, di_z, grid, nthreads);
    break;
  case 8:
    build_grad_mesh<8>(mesh, di_x, di_y, di_z, grid, nthreads);
    break;
  default:
    throw std::runtime_error("order must be 2,4,6,8");
  }

  return py::make_tuple(diff_x, diff_y, diff_z);
}
