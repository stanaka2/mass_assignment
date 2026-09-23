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

#include "domain.hpp"
#include "assignment_kernel.hpp"
#include "ptcl.hpp"

#include <omp.h>

namespace py = pybind11;

template <typename OutT>
struct Moments {
  OutT *s = nullptr;           // Scalar field
  OutT *m0 = nullptr;          // Mass/Density field
  std::array<OutT *, 3> m1{};  // Momentum (x, y, z)
  std::array<OutT *, 6> m2{};  // Second Moment (xx, xy, xz, yy, yz, zz)
  std::array<OutT *, 10> m3{}; // Third Moment
  std::array<OutT *, 15> m4{}; // Fourth Moment
};

template <typename T, typename OutT>
static void deposit_scalar(const Ptcl<T> &p, const T *scalar, const Moments<OutT> &M, const GridSpec &grid,
                           const int method, const int nthreads)
{
  if(nthreads > 0) omp_set_num_threads(nthreads);
  {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
    for(int64_t ip = 0; ip < p.n; ip++) {
      const double x = axis_coord(p.x(ip), grid.lbox(0), grid.bc(0));
      const double y = axis_coord(p.y(ip), grid.lbox(1), grid.bc(1));
      const double z = axis_coord(p.z(ip), grid.lbox(2), grid.bc(2));
      const double m = p.pmass(ip);

      int idx_x[4], idx_y[4], idx_z[4];
      double w_x[4], w_y[4], w_z[4];

      const int nax = assign_axis_1d(x * grid.inv_dx(0), grid.n[0], method, grid.bc(0), idx_x, w_x);
      const int nay = assign_axis_1d(y * grid.inv_dx(1), grid.n[1], method, grid.bc(1), idx_y, w_y);
      const int naz = assign_axis_1d(z * grid.inv_dx(2), grid.n[2], method, grid.bc(2), idx_z, w_z);
      const double s = (double)scalar[ip];

      for(int ix = 0; ix < nax; ix++) {
        const double wx = w_x[ix];
        for(int iy = 0; iy < nay; iy++) {
          const double wy = w_y[iy];
          for(int iz = 0; iz < naz; iz++) {
            const double wz = w_z[iz];
            const double ww = wx * wy * wz * m;
            const int64_t idx = idx3(idx_x[ix], idx_y[iy], idx_z[iz], grid.n);

#pragma omp atomic update
            M.m0[idx] += (OutT)ww; // weighted density
#pragma omp atomic update
            M.s[idx] += (OutT)(ww * s); // weighted scalar
          }
        }
      } // ix,iy,iz
    } // particles loop
  } // gil release
}

// First pass: mass field (ORDER 0) and momentum (ORDER 1).
// Second and higher moments are handled by deposit_central_moments().
template <int ORDER, typename T, typename OutT>
static void deposit_moments(const Ptcl<T> &p, const Moments<OutT> &M, const GridSpec &grid, const int method,
                            const int nthreads)
{
  static_assert(0 <= ORDER && ORDER <= 1, "ORDER must be 0..1");

  if(nthreads > 0) omp_set_num_threads(nthreads);
  {
    py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
    for(int64_t ip = 0; ip < p.n; ip++) {
      const double x = axis_coord(p.x(ip), grid.lbox(0), grid.bc(0));
      const double y = axis_coord(p.y(ip), grid.lbox(1), grid.bc(1));
      const double z = axis_coord(p.z(ip), grid.lbox(2), grid.bc(2));
      const double m = p.pmass(ip);

      int idx_x[4], idx_y[4], idx_z[4];
      double w_x[4], w_y[4], w_z[4];

      const int nax = assign_axis_1d(x * grid.inv_dx(0), grid.n[0], method, grid.bc(0), idx_x, w_x);
      const int nay = assign_axis_1d(y * grid.inv_dx(1), grid.n[1], method, grid.bc(1), idx_y, w_y);
      const int naz = assign_axis_1d(z * grid.inv_dx(2), grid.n[2], method, grid.bc(2), idx_z, w_z);

      double v[3];
      if constexpr(ORDER >= 1) {
        v[0] = p.vx(ip);
        v[1] = p.vy(ip);
        v[2] = p.vz(ip);
      }

      for(int ix = 0; ix < nax; ix++) {
        const double wx = w_x[ix];
        for(int iy = 0; iy < nay; iy++) {
          const double wy = w_y[iy];
          for(int iz = 0; iz < naz; iz++) {
            const double wz = w_z[iz];
            const double ww = wx * wy * wz * m;
            const int64_t idx = idx3(idx_x[ix], idx_y[iy], idx_z[iz], grid.n);

#pragma omp atomic update
            M.m0[idx] += (OutT)ww;

            if constexpr(ORDER >= 1) {
              for(int j = 0; j < 3; j++) {
#pragma omp atomic update
                M.m1[j][idx] += (OutT)(ww * v[j]);
              }
            } // ORDER
          }
        }
      } // ix,iy,iz
    } // particles loop
  } // gil release
}

// Second pass for high-order moments.
// M.m0 and M.m1 must already hold the mass field and the mean velocity of each cell.
// Central moments are accumulated directly, so the raw-to-central expansion is not needed.
template <int ORDER, typename T, typename OutT>
static void deposit_central_moments(const Ptcl<T> &p, const Moments<OutT> &M, const GridSpec &grid, const int method,
                                    const int nthreads)
{
  static_assert(2 <= ORDER && ORDER <= 4, "ORDER must be 2..4");

  if(nthreads > 0) omp_set_num_threads(nthreads);
  {
    py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
    for(int64_t ip = 0; ip < p.n; ip++) {
      const double x = axis_coord(p.x(ip), grid.lbox(0), grid.bc(0));
      const double y = axis_coord(p.y(ip), grid.lbox(1), grid.bc(1));
      const double z = axis_coord(p.z(ip), grid.lbox(2), grid.bc(2));
      const double m = p.pmass(ip);

      int idx_x[4], idx_y[4], idx_z[4];
      double w_x[4], w_y[4], w_z[4];

      const int nax = assign_axis_1d(x * grid.inv_dx(0), grid.n[0], method, grid.bc(0), idx_x, w_x);
      const int nay = assign_axis_1d(y * grid.inv_dx(1), grid.n[1], method, grid.bc(1), idx_y, w_y);
      const int naz = assign_axis_1d(z * grid.inv_dx(2), grid.n[2], method, grid.bc(2), idx_z, w_z);

      const double v[3] = {p.vx(ip), p.vy(ip), p.vz(ip)};

      for(int ix = 0; ix < nax; ix++) {
        const double wx = w_x[ix];
        for(int iy = 0; iy < nay; iy++) {
          const double wy = w_y[iy];
          for(int iz = 0; iz < naz; iz++) {
            const double wz = w_z[iz];
            const double ww = wx * wy * wz * m;
            const int64_t idx = idx3(idx_x[ix], idx_y[iy], idx_z[iz], grid.n);

            // the mean velocity differs from cell to cell, so dv is formed inside the stencil loop
            const double dv[3] = {v[0] - (double)M.m1[0][idx], v[1] - (double)M.m1[1][idx],
                                  v[2] - (double)M.m1[2][idx]};

            double dd[6];
            dd[0] = dv[0] * dv[0]; // xx
            dd[1] = dv[0] * dv[1]; // xy
            dd[2] = dv[0] * dv[2]; // xz
            dd[3] = dv[1] * dv[1]; // yy
            dd[4] = dv[1] * dv[2]; // yz
            dd[5] = dv[2] * dv[2]; // zz

            if constexpr(ORDER == 2) {
              for(int j = 0; j < 6; j++) {
#pragma omp atomic update
                M.m2[j][idx] += (OutT)(ww * dd[j]);
              }
            }

            if constexpr(ORDER >= 3) {
              double ddd[10];
              ddd[0] = dd[0] * dv[0]; // xxx
              ddd[1] = dd[0] * dv[1]; // xxy
              ddd[2] = dd[0] * dv[2]; // xxz
              ddd[3] = dd[1] * dv[1]; // xyy
              ddd[4] = dd[1] * dv[2]; // xyz
              ddd[5] = dd[2] * dv[2]; // xzz
              ddd[6] = dd[3] * dv[1]; // yyy
              ddd[7] = dd[3] * dv[2]; // yyz
              ddd[8] = dd[4] * dv[2]; // yzz
              ddd[9] = dd[5] * dv[2]; // zzz

              if constexpr(ORDER == 3) {
                for(int j = 0; j < 10; j++) {
#pragma omp atomic update
                  M.m3[j][idx] += (OutT)(ww * ddd[j]);
                }
              }

              if constexpr(ORDER == 4) {
                double dddd[15];
                // xxxx, xxxy, xxxz, xxyy, xxyz, xxzz
                dddd[0] = ddd[0] * dv[0];
                dddd[1] = ddd[0] * dv[1];
                dddd[2] = ddd[0] * dv[2];
                dddd[3] = ddd[1] * dv[1];
                dddd[4] = ddd[1] * dv[2];
                dddd[5] = ddd[2] * dv[2];
                // xyyy, xyyz, xyzz, xzzz
                dddd[6] = ddd[3] * dv[1];
                dddd[7] = ddd[3] * dv[2];
                dddd[8] = ddd[4] * dv[2];
                dddd[9] = ddd[5] * dv[2];
                // yyyy, yyyz, yyzz, yzzz, zzzz
                dddd[10] = ddd[6] * dv[1];
                dddd[11] = ddd[6] * dv[2];
                dddd[12] = ddd[7] * dv[2];
                dddd[13] = ddd[8] * dv[2];
                dddd[14] = ddd[9] * dv[2];

                for(int j = 0; j < 15; j++) {
#pragma omp atomic update
                  M.m4[j][idx] += (OutT)(ww * dddd[j]);
                }
              }
            }
          }
        }
      } // ix,iy,iz
    } // particles loop
  } // gil release
}

template <typename OutT>
static void calc_scalar_field(Moments<OutT> &M, const int64_t size)
{
  py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size; i++) {
    const OutT r = M.m0[i];

    if(r > 0.0) {
      const OutT inv = 1.0 / r;
      M.s[i] *= inv;
    } else {
      M.s[i] = 0.0;
    }
  }
}

template <typename OutT>
static void calc_velocity_components(Moments<OutT> &M, const int64_t size)
{
  py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size; i++) {
    const OutT r = M.m0[i];

    if(r > 0.0) {
      const OutT inv = 1.0 / r;
      M.m1[0][i] *= inv;
      M.m1[1][i] *= inv;
      M.m1[2][i] *= inv;

    } else {
      for(int j = 0; j < 3; ++j) M.m1[j][i] = 0.0;
    }
  }
}

template <typename OutT>
static void calc_velocity_norm(Moments<OutT> &M, OutT *norm_out, const int64_t size)
{
  py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size; i++) {
    const OutT r = M.m0[i];

    if(r > 0.0) {
      const OutT inv = 1.0 / r;
      const OutT ux = M.m1[0][i] * inv;
      const OutT uy = M.m1[1][i] * inv;
      const OutT uz = M.m1[2][i] * inv;
      norm_out[i] = std::sqrt(ux * ux + uy * uy + uz * uz);

    } else {
      norm_out[i] = 0.0;
    }
  }
}

template <typename OutT>
static void calc_sigma_components(Moments<OutT> &M, const int64_t size)
{
  py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size; i++) {
    const OutT r = M.m0[i];

    if(r > 0.0) {
      const OutT inv = 1.0 / r;
      for(int j = 0; j < 6; ++j) M.m2[j][i] *= inv;

    } else {
      for(int j = 0; j < 6; ++j) M.m2[j][i] = 0.0;
    }
  }
}

template <typename OutT>
static void calc_sigma_norm(Moments<OutT> &M, OutT *norm_out, const int64_t size, const int norm_mode)
{
  py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size; i++) {

    const OutT r = M.m0[i];
    if(r > 0.0) {
      const OutT inv = 1.0 / r;
      const OutT sxx = M.m2[0][i] * inv;
      const OutT syy = M.m2[3][i] * inv;
      const OutT szz = M.m2[5][i] * inv;

      if(norm_mode == 0) { // tr_norm
        norm_out[i] = std::sqrt(std::max((OutT)0.0, sxx + syy + szz));

      } else { // diag_norm (Frobenius)
        const OutT sxy = M.m2[1][i] * inv;
        const OutT sxz = M.m2[2][i] * inv;
        const OutT syz = M.m2[4][i] * inv;
        const OutT ss = sxx * sxx + syy * syy + szz * szz + 2.0 * (sxy * sxy + sxz * sxz + syz * syz);
        norm_out[i] = std::sqrt(std::max((OutT)0.0, ss));
      }

    } else {
      norm_out[i] = 0.0;
    }
  }
}

template <typename OutT>
static void calc_skewness_components(Moments<OutT> &M, const int64_t size)
{
  py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size; i++) {
    const OutT r = M.m0[i];

    if(r > 0.0) {
      const OutT inv = 1.0 / r;
      for(int j = 0; j < 10; ++j) M.m3[j][i] *= inv;

    } else {
      for(int j = 0; j < 10; ++j) M.m3[j][i] = 0.0;
    }
  }
}

template <typename OutT>
static void calc_skewness_norm(Moments<OutT> &M, OutT *norm_out, const int64_t size, const int norm_mode)
{
  py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size; i++) {
    const OutT r = M.m0[i];

    if(r > 0.0) {
      const OutT inv = 1.0 / r;
      OutT s[10];
      for(int j = 0; j < 10; ++j) s[j] = M.m3[j][i] * inv;

      if(norm_mode == 0) { // tr_norm
        norm_out[i] = std::sqrt(std::max((OutT)0.0, s[0] + s[6] + s[9]));
      } else { // diag_norm
        OutT sum2 = s[0] * s[0] + s[6] * s[6] + s[9] * s[9] +
                    3.0 * (s[1] * s[1] + s[2] * s[2] + s[3] * s[3] + s[5] * s[5] + s[7] * s[7] + s[8] * s[8]) +
                    6.0 * (s[4] * s[4]);
        norm_out[i] = std::sqrt(std::max((OutT)0.0, sum2));
      }

    } else {
      norm_out[i] = 0.0;
    }
  }
}

template <typename OutT>
static void calc_kurtosis_components(Moments<OutT> &M, const int64_t size)
{
  py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size; i++) {
    const OutT r = M.m0[i];

    if(r > 0.0) {
      const OutT inv = 1.0 / r;
      for(int j = 0; j < 15; ++j) M.m4[j][i] *= inv;

    } else {
      for(int j = 0; j < 15; ++j) M.m4[j][i] = 0.0;
    }
  }
}

template <typename OutT>
static void calc_kurtosis_norm(Moments<OutT> &M, OutT *norm_out, const int64_t size, const int norm_mode)
{
  py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
  for(int64_t i = 0; i < size; i++) {
    const OutT r = M.m0[i];

    if(r > 0.0) {
      const OutT inv = 1.0 / r;
      OutT s[15];
      for(int j = 0; j < 15; ++j) s[j] = M.m4[j][i] * inv;

      if(norm_mode == 0) { // tr_norm
        norm_out[i] = std::sqrt(std::max((OutT)0.0, s[0] + s[10] + s[14]));
      } else { // diag_norm
        OutT sum2 = s[0] * s[0] + s[10] * s[10] + s[14] * s[14] +
                    4.0 * (s[1] * s[1] + s[2] * s[2] + s[6] * s[6] + s[9] * s[9] + s[11] * s[11] + s[13] * s[13]) +
                    6.0 * (s[3] * s[3] + s[5] * s[5] + s[12] * s[12]) +
                    12.0 * (s[4] * s[4] + s[7] * s[7] + s[8] * s[8]);
        norm_out[i] = std::sqrt(std::max((OutT)0.0, sum2));
      }

    } else {
      norm_out[i] = 0.0;
    }
  }
}

template <typename T, typename OutT>
static py::array_t<OutT> density_impl(const Ptcl<T> &p, const GridSpec &grid, const int method, const int nthreads)
{
  const int64_t size = grid.size();
  py::array_t<OutT> rho_arr({grid.n[0], grid.n[1], grid.n[2]});
  OutT *rho = static_cast<OutT *>(rho_arr.request().ptr);
  std::fill_n(rho, size, static_cast<OutT>(0.0));

  Moments<OutT> M;
  M.m0 = rho;
  deposit_moments<0, T, OutT>(p, M, grid, method, nthreads);
  return rho_arr;
}

template <typename T, typename OutT>
static py::array_t<OutT> scalar_impl(const Ptcl<T> &p, const T *scalar, const GridSpec &grid, const int method,
                                     const int nthreads)
{
  const int64_t size = grid.size();
  py::array_t<OutT> scalar_arr({grid.n[0], grid.n[1], grid.n[2]});
  OutT *scalar_ptr = static_cast<OutT *>(scalar_arr.request().ptr);
  std::fill_n(scalar_ptr, size, static_cast<OutT>(0.0));
  std::vector<OutT> rho(size, 0.0);

  Moments<OutT> M;
  M.m0 = rho.data(); // sum_w
  M.s = scalar_ptr;  // sum_w * scalar
  deposit_scalar(p, scalar, M, grid, method, nthreads);
  calc_scalar_field(M, size); // s /= m0
  return scalar_arr;
}

template <typename T, typename OutT>
static py::object velocity_impl(const Ptcl<T> &p, const GridSpec &grid, const int method, const int nthreads)
{
  const int64_t size = grid.size();
  std::vector<OutT> rho(size, 0.0);
  // one array per component. A single (k,nx,ny,nz) block would put every component exactly
  // nx*ny*nz elements apart, which maps them all onto the same cache set for a power-of-two mesh.
  std::array<py::array_t<OutT>, 3> arrs;

  Moments<OutT> M;
  M.m0 = rho.data();
  for(int i = 0; i < 3; ++i) {
    arrs[i] = py::array_t<OutT>({grid.n[0], grid.n[1], grid.n[2]});
    M.m1[i] = static_cast<OutT *>(arrs[i].request().ptr);
    std::fill_n(M.m1[i], size, static_cast<OutT>(0.0));
  }

  deposit_moments<1, T, OutT>(p, M, grid, method, nthreads);
  calc_velocity_components(M, size);
  py::tuple t(3);
  for(int i = 0; i < 3; ++i) t[i] = arrs[i];
  return t;
}

template <typename T, typename OutT>
static py::array_t<OutT> velocity_norm_impl(const Ptcl<T> &p, const GridSpec &grid, const int method,
                                            const int nthreads)
{
  const int64_t size = grid.size();
  std::vector<OutT> rho(size, 0.0);
  std::vector<OutT> mx(size, 0.0), my(size, 0.0), mz(size, 0.0);

  py::array_t<OutT> norm_arr({grid.n[0], grid.n[1], grid.n[2]});
  OutT *out_ptr = static_cast<OutT *>(norm_arr.request().ptr);

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};

  deposit_moments<1, T, OutT>(p, M, grid, method, nthreads);
  calc_velocity_norm(M, out_ptr, size);
  return norm_arr;
}

template <typename T, typename OutT>
static py::object sigma_impl(const Ptcl<T> &p, const GridSpec &grid, const int method, const int nthreads)
{
  const int64_t size = grid.size();
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  // one array per component. A single (k,nx,ny,nz) block would put every component exactly
  // nx*ny*nz elements apart, which maps them all onto the same cache set for a power-of-two mesh.
  std::array<py::array_t<OutT>, 6> arrs;

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 6; ++i) {
    arrs[i] = py::array_t<OutT>({grid.n[0], grid.n[1], grid.n[2]});
    M.m2[i] = static_cast<OutT *>(arrs[i].request().ptr);
    std::fill_n(M.m2[i], size, static_cast<OutT>(0.0));
  }

  deposit_moments<1, T, OutT>(p, M, grid, method, nthreads);
  calc_velocity_components(M, size); // m1 holds the mean velocity from here on
  deposit_central_moments<2, T, OutT>(p, M, grid, method, nthreads);
  calc_sigma_components(M, size);
  py::tuple t(6);
  for(int i = 0; i < 6; ++i) t[i] = arrs[i];
  return t;
}

template <typename T, typename OutT>
static py::array_t<OutT> sigma_norm_impl(const Ptcl<T> &p, const GridSpec &grid, const int method, const int nthreads,
                                         int norm_mode)
{
  const int64_t size = grid.size();
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  std::vector<OutT> m2[6];
  for(int i = 0; i < 6; ++i) m2[i].resize(size, 0.0);

  py::array_t<OutT> norm_arr({grid.n[0], grid.n[1], grid.n[2]});
  OutT *out_ptr = static_cast<OutT *>(norm_arr.request().ptr);

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 6; ++i) M.m2[i] = m2[i].data();

  deposit_moments<1, T, OutT>(p, M, grid, method, nthreads);
  calc_velocity_components(M, size); // m1 holds the mean velocity from here on
  deposit_central_moments<2, T, OutT>(p, M, grid, method, nthreads);
  calc_sigma_norm(M, out_ptr, size, norm_mode);
  return norm_arr;
}

template <typename T, typename OutT>
static py::object skewness_impl(const Ptcl<T> &p, const GridSpec &grid, const int method, const int nthreads)
{
  const int64_t size = grid.size();
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);

  // one array per component. A single (k,nx,ny,nz) block would put every component exactly
  // nx*ny*nz elements apart, which maps them all onto the same cache set for a power-of-two mesh.
  std::array<py::array_t<OutT>, 10> arrs;

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 10; ++i) {
    arrs[i] = py::array_t<OutT>({grid.n[0], grid.n[1], grid.n[2]});
    M.m3[i] = static_cast<OutT *>(arrs[i].request().ptr);
    std::fill_n(M.m3[i], size, static_cast<OutT>(0.0));
  }

  deposit_moments<1, T, OutT>(p, M, grid, method, nthreads);
  calc_velocity_components(M, size); // m1 holds the mean velocity from here on
  deposit_central_moments<3, T, OutT>(p, M, grid, method, nthreads);
  calc_skewness_components(M, size);

  py::tuple t(10);
  for(int i = 0; i < 10; ++i) t[i] = arrs[i];
  return t;
}

template <typename T, typename OutT>
static py::array_t<OutT> skewness_norm_impl(const Ptcl<T> &p, const GridSpec &grid, const int method,
                                            const int nthreads, int norm_mode)
{
  const int64_t size = grid.size();
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  std::vector<OutT> m3[10];
  for(int i = 0; i < 10; ++i) m3[i].resize(size, 0.0);

  py::array_t<OutT> norm_arr({grid.n[0], grid.n[1], grid.n[2]});
  OutT *out_ptr = static_cast<OutT *>(norm_arr.request().ptr);

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 10; ++i) M.m3[i] = m3[i].data();

  deposit_moments<1, T, OutT>(p, M, grid, method, nthreads);
  calc_velocity_components(M, size); // m1 holds the mean velocity from here on
  deposit_central_moments<3, T, OutT>(p, M, grid, method, nthreads);
  calc_skewness_norm(M, out_ptr, size, norm_mode);
  return norm_arr;
}

template <typename T, typename OutT>
static py::object kurtosis_impl(const Ptcl<T> &p, const GridSpec &grid, const int method, const int nthreads)
{
  const int64_t size = grid.size();
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);

  // one array per component. A single (k,nx,ny,nz) block would put every component exactly
  // nx*ny*nz elements apart, which maps them all onto the same cache set for a power-of-two mesh.
  std::array<py::array_t<OutT>, 15> arrs;

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 15; ++i) {
    arrs[i] = py::array_t<OutT>({grid.n[0], grid.n[1], grid.n[2]});
    M.m4[i] = static_cast<OutT *>(arrs[i].request().ptr);
    std::fill_n(M.m4[i], size, static_cast<OutT>(0.0));
  }

  deposit_moments<1, T, OutT>(p, M, grid, method, nthreads);
  calc_velocity_components(M, size); // m1 holds the mean velocity from here on
  deposit_central_moments<4, T, OutT>(p, M, grid, method, nthreads);
  calc_kurtosis_components(M, size);

  py::tuple t(15);
  for(int i = 0; i < 15; ++i) t[i] = arrs[i];
  return t;
}

template <typename T, typename OutT>
static py::array_t<OutT> kurtosis_norm_impl(const Ptcl<T> &p, const GridSpec &grid, const int method,
                                            const int nthreads, int norm_mode)
{
  const int64_t size = grid.size();
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  std::vector<OutT> m4[15];
  for(int i = 0; i < 15; ++i) m4[i].resize(size, 0.0);

  py::array_t<OutT> norm_arr({grid.n[0], grid.n[1], grid.n[2]});
  OutT *out_ptr = static_cast<OutT *>(norm_arr.request().ptr);

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 15; ++i) M.m4[i] = m4[i].data();

  deposit_moments<1, T, OutT>(p, M, grid, method, nthreads);
  calc_velocity_components(M, size); // m1 holds the mean velocity from here on
  deposit_central_moments<4, T, OutT>(p, M, grid, method, nthreads);
  calc_kurtosis_norm(M, out_ptr, size, norm_mode);
  return norm_arr;
}
