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

namespace py = pybind11;

template <typename OutT>
struct Moments {
  OutT *m0 = nullptr;          // Mass/Density
  std::array<OutT *, 3> m1{};  // Momentum (x, y, z)
  std::array<OutT *, 6> m2{};  // Second Moment (xx, xy, xz, yy, yz, zz)
  std::array<OutT *, 10> m3{}; // Third Moment
  std::array<OutT *, 15> m4{}; // Fourth Moment
};

static inline double wrap_periodic(double x, const double lbox)
{
  const double inv = 1.0 / lbox;
  x -= std::floor(x * inv) * lbox;
  if(x >= lbox) x -= lbox;
  if(x < 0.0) x += lbox;
  return x;
}

static inline int assign_axis_1d(double xg, const int nmesh, const int method, int idx[4], double w[4])
{
  // xg: coordinate in grid units (0..nmesh)

  if(method == 1) {
    // NGP
    int ic = (int)(std::floor(xg + 0.5)); // need +0.5
    ic = (ic % nmesh + nmesh) % nmesh;
    idx[0] = ic;
    w[0] = 1.0;
    return 1;
  }

  if(method == 2) {
    // CIC
    int i0 = (int)(std::floor(xg));
    double f = xg - (double)(i0);
    int i1 = i0 + 1;
    i0 = (i0 % nmesh + nmesh) % nmesh;
    i1 = (i1 % nmesh + nmesh) % nmesh;
    idx[0] = i0;
    idx[1] = i1;
    w[0] = 1.0 - f;
    w[1] = f;
    return 2;
  }

  if(method == 3) {
    // TSC
    int ic = (int)(std::floor(xg + 0.5)); // need +0.5
    double d = xg - (double)(ic);
    int im1 = ic - 1;
    int ip1 = ic + 1;

    im1 = (im1 % nmesh + nmesh) % nmesh;
    ic = (ic % nmesh + nmesh) % nmesh;
    ip1 = (ip1 % nmesh + nmesh) % nmesh;

    idx[0] = im1;
    idx[1] = ic;
    idx[2] = ip1;

    w[0] = 0.5 * (0.5 - d) * (0.5 - d);
    w[1] = 0.75 - d * d;
    w[2] = 0.5 * (0.5 + d) * (0.5 + d);
    return 3;
  }

  if(method == 4) {
    // PCS (cubic B-spline, 4-point)
    int i0 = (int)(std::floor(xg));
    double u = xg - (double)(i0); // [0,1)
    int im1 = i0 - 1;
    int ip1 = i0 + 1;
    int ip2 = i0 + 2;

    i0 = (i0 % nmesh + nmesh) % nmesh;
    im1 = (im1 % nmesh + nmesh) % nmesh;
    ip1 = (ip1 % nmesh + nmesh) % nmesh;
    ip2 = (ip2 % nmesh + nmesh) % nmesh;

    idx[0] = im1;
    idx[1] = i0;
    idx[2] = ip1;
    idx[3] = ip2;

    w[0] = (1.0 / 6.0) * (1.0 - u) * (1.0 - u) * (1.0 - u);
    w[1] = (1.0 / 6.0) * (3.0 * u * u * u - 6.0 * u * u + 4.0);
    w[2] = (1.0 / 6.0) * (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0);
    w[3] = (1.0 / 6.0) * (u * u * u);
    return 4;
  }
  throw std::runtime_error("method must be 1(NGP), 2(CIC), 3(TSC), or 4(PCS)");
}

template <int ORDER, typename T, typename OutT>
static void deposit_moments(const T *pos, const T *vel, const T *mass, const Moments<OutT> &M, const int64_t n,
                            const double lbox, const int nmesh, const int method, const int nthreads)
{
  static_assert(0 <= ORDER && ORDER <= 4, "ORDER must be 0..4");

#ifdef _OPENMP
  if(nthreads > 0) omp_set_num_threads(nthreads);
#endif
  const double inv_dx = (double)nmesh / lbox;

  {
    py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
    for(int64_t ip = 0; ip < n; ip++) {
      const double x = wrap_periodic(pos[3 * ip + 0], lbox);
      const double y = wrap_periodic(pos[3 * ip + 1], lbox);
      const double z = wrap_periodic(pos[3 * ip + 2], lbox);
      const double m = (mass != nullptr) ? (double)mass[ip] : 1.0;

      int idx_x[4], idx_y[4], idx_z[4];
      double w_x[4], w_y[4], w_z[4];

      const int nax = assign_axis_1d(x * inv_dx, nmesh, method, idx_x, w_x);
      const int nay = assign_axis_1d(y * inv_dx, nmesh, method, idx_y, w_y);
      const int naz = assign_axis_1d(z * inv_dx, nmesh, method, idx_z, w_z);

      double v[3], vv[6], vvv[10], vvvv[15];
      if constexpr(ORDER >= 1) {
        v[0] = (double)vel[3 * ip + 0];
        v[1] = (double)vel[3 * ip + 1];
        v[2] = (double)vel[3 * ip + 2];

        if constexpr(ORDER >= 2) {
          vv[0] = v[0] * v[0]; // xx
          vv[1] = v[0] * v[1]; // xy
          vv[2] = v[0] * v[2]; // xz
          vv[3] = v[1] * v[1]; // yy
          vv[4] = v[1] * v[2]; // yz
          vv[5] = v[2] * v[2]; // zz

          if constexpr(ORDER >= 3) {
            vvv[0] = vv[0] * v[0]; // xxx
            vvv[1] = vv[0] * v[1]; // xxy
            vvv[2] = vv[0] * v[2]; // xxz
            vvv[3] = vv[1] * v[1]; // xyy
            vvv[4] = vv[1] * v[2]; // xyz
            vvv[5] = vv[2] * v[2]; // xzz
            vvv[6] = vv[3] * v[1]; // yyy
            vvv[7] = vv[3] * v[2]; // yyz
            vvv[8] = vv[4] * v[2]; // yzz
            vvv[9] = vv[5] * v[2]; // zzz

            if constexpr(ORDER >= 4) {
              // xxxx, xxxy, xxxz, xxyy, xxyz, xxzz
              vvvv[0] = vvv[0] * v[0];
              vvvv[1] = vvv[0] * v[1];
              vvvv[2] = vvv[0] * v[2];
              vvvv[3] = vvv[1] * v[1];
              vvvv[4] = vvv[1] * v[2];
              vvvv[5] = vvv[2] * v[2];
              // xyyy, xyyz, xyzz, xzzz
              vvvv[6] = vvv[3] * v[1];
              vvvv[7] = vvv[3] * v[2];
              vvvv[8] = vvv[4] * v[2];
              vvvv[9] = vvv[5] * v[2];
              // yyyy, yyyz, yyzz, yzzz, zzzz
              vvvv[10] = vvv[6] * v[1];
              vvvv[11] = vvv[6] * v[2];
              vvvv[12] = vvv[7] * v[2];
              vvvv[13] = vvv[8] * v[2];
              vvvv[14] = vvv[9] * v[2];
            }
          }
        }
      }

      for(int ix = 0; ix < nax; ix++) {
        const double wx = w_x[ix];
        for(int iy = 0; iy < nay; iy++) {
          const double wy = w_y[iy];
          for(int iz = 0; iz < naz; iz++) {
            const double wz = w_z[iz];
            const double ww = wx * wy * wz * m;
            const int64_t idx =
                (int64_t)idx_z[iz] + (int64_t)nmesh * ((int64_t)idx_y[iy] + (int64_t)nmesh * (int64_t)idx_x[ix]);

#pragma omp atomic update
            M.m0[idx] += (OutT)ww;

            if constexpr(ORDER >= 1) {
              for(int j = 0; j < 3; j++) {
#pragma omp atomic update
                M.m1[j][idx] += (OutT)(ww * v[j]);
              }
              if constexpr(ORDER >= 2) {
                for(int j = 0; j < 6; j++) {
#pragma omp atomic update
                  M.m2[j][idx] += (OutT)(ww * vv[j]);
                }
                if constexpr(ORDER >= 3) {
                  for(int j = 0; j < 10; j++) {
#pragma omp atomic update
                    M.m3[j][idx] += (OutT)(ww * vvv[j]);
                  }
                  if constexpr(ORDER >= 4) {
                    for(int j = 0; j < 15; j++) {
#pragma omp atomic update
                      M.m4[j][idx] += (OutT)(ww * vvvv[j]);
                    }
                  }
                }
              }
            } // ORDER
          }
        }
      } // ix,iy,iz
    } // particles loop
  } // gil release
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
      const OutT ux = M.m1[0][i] * inv;
      const OutT uy = M.m1[1][i] * inv;
      const OutT uz = M.m1[2][i] * inv;
      M.m2[0][i] = (M.m2[0][i] * inv) - ux * ux; // xx
      M.m2[1][i] = (M.m2[1][i] * inv) - ux * uy; // xy
      M.m2[2][i] = (M.m2[2][i] * inv) - ux * uz; // xz
      M.m2[3][i] = (M.m2[3][i] * inv) - uy * uy; // yy
      M.m2[4][i] = (M.m2[4][i] * inv) - uy * uz; // yz
      M.m2[5][i] = (M.m2[5][i] * inv) - uz * uz; // zz

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
      const OutT ux = M.m1[0][i] * inv;
      const OutT uy = M.m1[1][i] * inv;
      const OutT uz = M.m1[2][i] * inv;

      const OutT sxx = (M.m2[0][i] * inv) - ux * ux;
      const OutT syy = (M.m2[3][i] * inv) - uy * uy;
      const OutT szz = (M.m2[5][i] * inv) - uz * uz;

      if(norm_mode == 0) { // tr_norm
        norm_out[i] = std::sqrt(std::max((OutT)0.0, sxx + syy + szz));

      } else { // diag_norm (Frobenius)
        const OutT sxy = (M.m2[1][i] * inv) - ux * uy;
        const OutT sxz = (M.m2[2][i] * inv) - ux * uz;
        const OutT syz = (M.m2[4][i] * inv) - uy * uz;
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
      const OutT u[3] = {M.m1[0][i] * inv, M.m1[1][i] * inv, M.m1[2][i] * inv};
      const OutT e2[6] = {M.m2[0][i] * inv, M.m2[1][i] * inv, M.m2[2][i] * inv,
                          M.m2[3][i] * inv, M.m2[4][i] * inv, M.m2[5][i] * inv};
      const OutT u2[3] = {u[0] * u[0], u[1] * u[1], u[2] * u[2]};

      M.m3[0][i] = (M.m3[0][i] * inv) - 3.0 * e2[0] * u[0] + 2.0 * u[0] * u2[0];                                 // xxx
      M.m3[1][i] = (M.m3[1][i] * inv) - (2.0 * e2[1] * u[0] + e2[0] * u[1]) + 2.0 * u2[0] * u[1];                // xxy
      M.m3[2][i] = (M.m3[2][i] * inv) - (2.0 * e2[2] * u[0] + e2[0] * u[2]) + 2.0 * u2[0] * u[2];                // xxz
      M.m3[3][i] = (M.m3[3][i] * inv) - (2.0 * e2[1] * u[1] + e2[3] * u[0]) + 2.0 * u2[1] * u[0];                // xyy
      M.m3[4][i] = (M.m3[4][i] * inv) - (e2[4] * u[0] + e2[2] * u[1] + e2[1] * u[2]) + 2.0 * u[0] * u[1] * u[2]; // xyz
      M.m3[5][i] = (M.m3[5][i] * inv) - (2.0 * e2[2] * u[2] + e2[5] * u[0]) + 2.0 * u2[2] * u[0];                // xzz
      M.m3[6][i] = (M.m3[6][i] * inv) - 3.0 * e2[3] * u[1] + 2.0 * u[1] * u2[1];                                 // yyy
      M.m3[7][i] = (M.m3[7][i] * inv) - (2.0 * e2[4] * u[1] + e2[3] * u[2]) + 2.0 * u2[1] * u[2];                // yyz
      M.m3[8][i] = (M.m3[8][i] * inv) - (2.0 * e2[4] * u[2] + e2[5] * u[1]) + 2.0 * u2[2] * u[1];                // yzz
      M.m3[9][i] = (M.m3[9][i] * inv) - 3.0 * e2[5] * u[2] + 2.0 * u[2] * u2[2];                                 // zzz

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
      const OutT u[3] = {M.m1[0][i] * inv, M.m1[1][i] * inv, M.m1[2][i] * inv};
      const OutT e2[6] = {M.m2[0][i] * inv, M.m2[1][i] * inv, M.m2[2][i] * inv,
                          M.m2[3][i] * inv, M.m2[4][i] * inv, M.m2[5][i] * inv};
      const OutT u2[3] = {u[0] * u[0], u[1] * u[1], u[2] * u[2]};

      OutT s[10];
      s[0] = (M.m3[0][i] * inv) - 3.0 * e2[0] * u[0] + 2.0 * u[0] * u2[0];
      s[6] = (M.m3[6][i] * inv) - 3.0 * e2[3] * u[1] + 2.0 * u[1] * u2[1];
      s[9] = (M.m3[9][i] * inv) - 3.0 * e2[5] * u[2] + 2.0 * u[2] * u2[2];

      if(norm_mode == 0) { // tr_norm
        norm_out[i] = std::sqrt(std::max((OutT)0.0, s[0] + s[6] + s[9]));
      } else { // diag_norm
        s[1] = (M.m3[1][i] * inv) - (2.0 * e2[1] * u[0] + e2[0] * u[1]) + 2.0 * u2[0] * u[1];
        s[2] = (M.m3[2][i] * inv) - (2.0 * e2[2] * u[0] + e2[0] * u[2]) + 2.0 * u2[0] * u[2];
        s[3] = (M.m3[3][i] * inv) - (2.0 * e2[1] * u[1] + e2[3] * u[0]) + 2.0 * u2[1] * u[0];
        s[4] = (M.m3[4][i] * inv) - (e2[4] * u[0] + e2[2] * u[1] + e2[1] * u[2]) + 2.0 * u[0] * u[1] * u[2];
        s[5] = (M.m3[5][i] * inv) - (2.0 * e2[2] * u[2] + e2[5] * u[0]) + 2.0 * u2[2] * u[0];
        s[7] = (M.m3[7][i] * inv) - (2.0 * e2[4] * u[1] + e2[3] * u[2]) + 2.0 * u2[1] * u[2];
        s[8] = (M.m3[8][i] * inv) - (2.0 * e2[4] * u[2] + e2[5] * u[1]) + 2.0 * u2[2] * u[1];

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
      const OutT u[3] = {M.m1[0][i] * inv, M.m1[1][i] * inv, M.m1[2][i] * inv};
      const OutT e2[6] = {M.m2[0][i] * inv, M.m2[1][i] * inv, M.m2[2][i] * inv,
                          M.m2[3][i] * inv, M.m2[4][i] * inv, M.m2[5][i] * inv};
      const OutT e3[10] = {M.m3[0][i] * inv, M.m3[1][i] * inv, M.m3[2][i] * inv, M.m3[3][i] * inv, M.m3[4][i] * inv,
                           M.m3[5][i] * inv, M.m3[6][i] * inv, M.m3[7][i] * inv, M.m3[8][i] * inv, M.m3[9][i] * inv};
      const OutT u2[3] = {u[0] * u[0], u[1] * u[1], u[2] * u[2]};
      const OutT u3[3] = {u2[0] * u[0], u2[1] * u[1], u2[2] * u[2]};
      const OutT u4[3] = {u2[0] * u2[0], u2[1] * u2[1], u2[2] * u2[2]};

      M.m4[0][i] = (M.m4[0][i] * inv) - 4.0 * e3[0] * u[0] + 6.0 * e2[0] * u2[0] - 3.0 * u4[0];   // xxxx
      M.m4[10][i] = (M.m4[10][i] * inv) - 4.0 * e3[6] * u[1] + 6.0 * e2[3] * u2[1] - 3.0 * u4[1]; // yyyy
      M.m4[14][i] = (M.m4[14][i] * inv) - 4.0 * e3[9] * u[2] + 6.0 * e2[5] * u2[2] - 3.0 * u4[2]; // zzzz
      // Full 15 components calculation ...
      M.m4[1][i] = (M.m4[1][i] * inv) - (3.0 * e3[1] * u[0] + e3[0] * u[1]) +
                   (3.0 * e2[1] * u2[0] + 3.0 * e2[0] * u[0] * u[1]) - 3.0 * u3[0] * u[1];
      M.m4[2][i] = (M.m4[2][i] * inv) - (3.0 * e3[2] * u[0] + e3[0] * u[2]) +
                   (3.0 * e2[2] * u2[0] + 3.0 * e2[0] * u[0] * u[2]) - 3.0 * u3[0] * u[2];
      M.m4[3][i] = (M.m4[3][i] * inv) - (2.0 * e3[3] * u[0] + 2.0 * e3[1] * u[1]) +
                   (e2[3] * u2[0] + 4.0 * e2[1] * u[0] * u[1] + e2[0] * u2[1]) - 3.0 * u2[0] * u2[1];
      M.m4[4][i] = (M.m4[4][i] * inv) - (2.0 * e3[4] * u[0] + e3[2] * u[1] + e3[1] * u[2]) +
                   (e2[4] * u2[0] + 2.0 * e2[2] * u[0] * u[1] + 2.0 * e2[1] * u[0] * u[2] + e2[0] * u[1] * u[2]) -
                   3.0 * u2[0] * u[1] * u[2];
      M.m4[5][i] = (M.m4[5][i] * inv) - (2.0 * e3[5] * u[0] + 2.0 * e3[2] * u[2]) +
                   (e2[5] * u2[0] + 4.0 * e2[2] * u[0] * u[2] + e2[0] * u2[2]) - 3.0 * u2[0] * u2[2];
      M.m4[6][i] = (M.m4[6][i] * inv) - (3.0 * e3[3] * u[1] + e3[6] * u[0]) +
                   (3.0 * e2[1] * u2[1] + 3.0 * e2[3] * u[0] * u[1]) - 3.0 * u[0] * u3[1];
      M.m4[7][i] = (M.m4[7][i] * inv) - (2.0 * e3[4] * u[1] + e3[7] * u[0] + e3[3] * u[2]) +
                   (e2[4] * u2[1] + 2.0 * e2[1] * u[1] * u[2] + 2.0 * e2[3] * u[0] * u[2] + e2[4] * u[0] * u[1]) -
                   3.0 * u[0] * u2[1] * u[2];
      M.m4[8][i] = (M.m4[8][i] * inv) - (2.0 * e3[4] * u[2] + e3[9] * u[0] + e3[5] * u[1]) +
                   (e2[4] * u2[2] + 2.0 * e2[1] * u[2] * u[1] + 2.0 * e2[5] * u[0] * u[1] + e2[2] * u[1] * u[2]) -
                   3.0 * u[0] * u[1] * u2[2];
      M.m4[9][i] = (M.m4[9][i] * inv) - (3.0 * e3[5] * u[2] + e3[9] * u[0]) +
                   (3.0 * e2[2] * u2[2] + 3.0 * e2[5] * u[0] * u[2]) - 3.0 * u[0] * u3[2];
      M.m4[11][i] = (M.m4[11][i] * inv) - (3.0 * e3[7] * u[1] + e3[6] * u[2]) +
                    (3.0 * e2[4] * u2[1] + 3.0 * e2[3] * u[1] * u[2]) - 3.0 * u3[1] * u[2];
      M.m4[12][i] = (M.m4[12][i] * inv) - (2.0 * e3[8] * u[1] + 2.0 * e3[7] * u[2]) +
                    (e2[5] * u2[1] + 4.0 * e2[4] * u[1] * u[2] + e2[3] * u2[2]) - 3.0 * u2[1] * u2[2];
      M.m4[13][i] = (M.m4[13][i] * inv) - (3.0 * e3[8] * u[2] + e3[9] * u[1]) +
                    (3.0 * e2[4] * u2[2] + 3.0 * e2[5] * u[1] * u[2]) - 3.0 * u[1] * u3[2];

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
      const OutT u[3] = {M.m1[0][i] * inv, M.m1[1][i] * inv, M.m1[2][i] * inv};
      const OutT e2[6] = {M.m2[0][i] * inv, M.m2[1][i] * inv, M.m2[2][i] * inv,
                          M.m2[3][i] * inv, M.m2[4][i] * inv, M.m2[5][i] * inv};
      const OutT e3[10] = {M.m3[0][i] * inv, M.m3[1][i] * inv, M.m3[2][i] * inv, M.m3[3][i] * inv, M.m3[4][i] * inv,
                           M.m3[5][i] * inv, M.m3[6][i] * inv, M.m3[7][i] * inv, M.m3[8][i] * inv, M.m3[9][i] * inv};
      const OutT u2[3] = {u[0] * u[0], u[1] * u[1], u[2] * u[2]};
      const OutT u3[3] = {u2[0] * u[0], u2[1] * u[1], u2[2] * u[2]};
      const OutT u4[3] = {u2[0] * u2[0], u2[1] * u2[1], u2[2] * u2[2]};

      OutT s[15];
      s[0] = (M.m4[0][i] * inv) - 4.0 * e3[0] * u[0] + 6.0 * e2[0] * u2[0] - 3.0 * u4[0];
      s[10] = (M.m4[10][i] * inv) - 4.0 * e3[6] * u[1] + 6.0 * e2[3] * u2[1] - 3.0 * u4[1];
      s[14] = (M.m4[14][i] * inv) - 4.0 * e3[9] * u[2] + 6.0 * e2[5] * u2[2] - 3.0 * u4[2];

      if(norm_mode == 0) { // tr_norm
        norm_out[i] = std::sqrt(std::max((OutT)0.0, s[0] + s[10] + s[14]));
      } else { // diag_norm
        // Calculate remaining terms ...
        s[1] = (M.m4[1][i] * inv) - (3.0 * e3[1] * u[0] + e3[0] * u[1]) +
               (3.0 * e2[1] * u2[0] + 3.0 * e2[0] * u[0] * u[1]) - 3.0 * u3[0] * u[1];
        s[2] = (M.m4[2][i] * inv) - (3.0 * e3[2] * u[0] + e3[0] * u[2]) +
               (3.0 * e2[2] * u2[0] + 3.0 * e2[0] * u[0] * u[2]) - 3.0 * u3[0] * u[2];
        s[3] = (M.m4[3][i] * inv) - (2.0 * e3[3] * u[0] + 2.0 * e3[1] * u[1]) +
               (e2[3] * u2[0] + 4.0 * e2[1] * u[0] * u[1] + e2[0] * u2[1]) - 3.0 * u2[0] * u2[1];
        s[4] = (M.m4[4][i] * inv) - (2.0 * e3[4] * u[0] + e3[2] * u[1] + e3[1] * u[2]) +
               (e2[4] * u2[0] + 2.0 * e2[2] * u[0] * u[1] + 2.0 * e2[1] * u[0] * u[2] + e2[0] * u[1] * u[2]) -
               3.0 * u2[0] * u[1] * u[2];
        s[5] = (M.m4[5][i] * inv) - (2.0 * e3[5] * u[0] + 2.0 * e3[2] * u[2]) +
               (e2[5] * u2[0] + 4.0 * e2[2] * u[0] * u[2] + e2[0] * u2[2]) - 3.0 * u2[0] * u2[2];
        s[6] = (M.m4[6][i] * inv) - (3.0 * e3[3] * u[1] + e3[6] * u[0]) +
               (3.0 * e2[1] * u2[1] + 3.0 * e2[3] * u[0] * u[1]) - 3.0 * u[0] * u3[1];
        s[7] = (M.m4[7][i] * inv) - (2.0 * e3[4] * u[1] + e3[7] * u[0] + e3[3] * u[2]) +
               (e2[4] * u2[1] + 2.0 * e2[1] * u[1] * u[2] + 2.0 * e2[3] * u[0] * u[2] + e2[4] * u[0] * u[1]) -
               3.0 * u[0] * u2[1] * u[2];
        s[8] = (M.m4[8][i] * inv) - (2.0 * e3[4] * u[2] + e3[9] * u[0] + e3[5] * u[1]) +
               (e2[4] * u2[2] + 2.0 * e2[1] * u[2] * u[1] + 2.0 * e2[5] * u[0] * u[1] + e2[2] * u[1] * u[2]) -
               3.0 * u[0] * u[1] * u2[2];
        s[9] = (M.m4[9][i] * inv) - (3.0 * e3[5] * u[2] + e3[9] * u[0]) +
               (3.0 * e2[2] * u2[2] + 3.0 * e2[5] * u[0] * u[2]) - 3.0 * u[0] * u3[2];
        s[11] = (M.m4[11][i] * inv) - (3.0 * e3[7] * u[1] + e3[6] * u[2]) +
                (3.0 * e2[4] * u2[1] + 3.0 * e2[3] * u[1] * u[2]) - 3.0 * u3[1] * u[2];
        s[12] = (M.m4[12][i] * inv) - (2.0 * e3[8] * u[1] + 2.0 * e3[7] * u[2]) +
                (e2[5] * u2[1] + 4.0 * e2[4] * u[1] * u[2] + e2[3] * u2[2]) - 3.0 * u2[1] * u2[2];
        s[13] = (M.m4[13][i] * inv) - (3.0 * e3[8] * u[2] + e3[9] * u[1]) +
                (3.0 * e2[4] * u2[2] + 3.0 * e2[5] * u[1] * u[2]) - 3.0 * u[1] * u3[2];

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
static py::array_t<OutT> density_impl(const T *pos, const T *mass, const int64_t n, const double lbox, const int nmesh,
                                      const int method, const int nthreads)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  py::array_t<OutT> rho_arr({nmesh, nmesh, nmesh});
  OutT *rho = static_cast<OutT *>(rho_arr.request().ptr);
  std::fill_n(rho, size, static_cast<OutT>(0.0));

  Moments<OutT> M;
  M.m0 = rho;
  deposit_moments<0, T, OutT>(pos, nullptr, mass, M, n, lbox, nmesh, method, nthreads);
  return rho_arr;
}

template <typename T, typename OutT>
static py::object velocity_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                                const int nmesh, const int method, const int nthreads)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  std::vector<OutT> rho(size, 0.0);
  py::array_t<OutT> vx({nmesh, nmesh, nmesh}), vy({nmesh, nmesh, nmesh}), vz({nmesh, nmesh, nmesh});

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1[0] = static_cast<OutT *>(vx.request().ptr);
  M.m1[1] = static_cast<OutT *>(vy.request().ptr);
  M.m1[2] = static_cast<OutT *>(vz.request().ptr);
  std::fill_n(M.m1[0], size, 0.0);
  std::fill_n(M.m1[1], size, 0.0);
  std::fill_n(M.m1[2], size, 0.0);

  deposit_moments<1, T, OutT>(pos, vel, mass, M, n, lbox, nmesh, method, nthreads);
  calc_velocity_components(M, size);
  return py::make_tuple(vx, vy, vz);
}

template <typename T, typename OutT>
static py::object velocity_norm_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                                     const int nmesh, const int method, const int nthreads)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  std::vector<OutT> rho(size, 0.0);
  std::vector<OutT> mx(size, 0.0), my(size, 0.0), mz(size, 0.0);

  py::array_t<OutT> norm_arr({nmesh, nmesh, nmesh});
  OutT *out_ptr = static_cast<OutT *>(norm_arr.request().ptr);

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};

  deposit_moments<1, T, OutT>(pos, vel, mass, M, n, lbox, nmesh, method, nthreads);
  calc_velocity_norm(M, out_ptr, size);
  return norm_arr;
}

template <typename T, typename OutT>
static py::object sigma_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                             const int nmesh, const int method, const int nthreads)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  std::array<py::array_t<OutT>, 6> arrs;

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 6; ++i) {
    arrs[i] = py::array_t<OutT>({nmesh, nmesh, nmesh});
    M.m2[i] = static_cast<OutT *>(arrs[i].request().ptr);
    std::fill_n(M.m2[i], size, 0.0);
  }

  deposit_moments<2, T, OutT>(pos, vel, mass, M, n, lbox, nmesh, method, nthreads);
  calc_sigma_components(M, size);
  return py::make_tuple(arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], arrs[5]);
}

template <typename T, typename OutT>
static py::object sigma_norm_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                                  const int nmesh, const int method, const int nthreads, int norm_mode)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  std::vector<OutT> m2[6];
  for(int i = 0; i < 6; ++i) m2[i].resize(size, 0.0);

  py::array_t<OutT> norm_arr({nmesh, nmesh, nmesh});
  OutT *out_ptr = static_cast<OutT *>(norm_arr.request().ptr);

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 6; ++i) M.m2[i] = m2[i].data();

  deposit_moments<2, T, OutT>(pos, vel, mass, M, n, lbox, nmesh, method, nthreads);
  calc_sigma_norm(M, out_ptr, size, norm_mode);
  return norm_arr;
}

template <typename T, typename OutT>
static py::object skewness_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                                const int nmesh, const int method, const int nthreads)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  std::vector<OutT> m2[6];
  for(int i = 0; i < 6; ++i) m2[i].resize(size, 0.0);

  std::array<py::array_t<OutT>, 10> arrs;
  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 6; ++i) M.m2[i] = m2[i].data();
  for(int i = 0; i < 10; ++i) {
    arrs[i] = py::array_t<OutT>({nmesh, nmesh, nmesh});
    M.m3[i] = static_cast<OutT *>(arrs[i].request().ptr);
    std::fill_n(M.m3[i], size, 0.0);
  }

  deposit_moments<3, T, OutT>(pos, vel, mass, M, n, lbox, nmesh, method, nthreads);
  calc_skewness_components(M, size);

  py::tuple t(10);
  for(int i = 0; i < 10; ++i) t[i] = arrs[i];
  return t;
}

template <typename T, typename OutT>
static py::object skewness_norm_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                                     const int nmesh, const int method, const int nthreads, int norm_mode)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  std::vector<OutT> m2[6];
  for(int i = 0; i < 6; ++i) m2[i].resize(size, 0.0);
  std::vector<OutT> m3[10];
  for(int i = 0; i < 10; ++i) m3[i].resize(size, 0.0);

  py::array_t<OutT> norm_arr({nmesh, nmesh, nmesh});
  OutT *out_ptr = static_cast<OutT *>(norm_arr.request().ptr);

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 6; ++i) M.m2[i] = m2[i].data();
  for(int i = 0; i < 10; ++i) M.m3[i] = m3[i].data();

  deposit_moments<3, T, OutT>(pos, vel, mass, M, n, lbox, nmesh, method, nthreads);
  calc_skewness_norm(M, out_ptr, size, norm_mode);
  return norm_arr;
}

template <typename T, typename OutT>
static py::object kurtosis_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                                const int nmesh, const int method, const int nthreads)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  std::vector<OutT> m2[6], m3[10];
  for(int i = 0; i < 6; ++i) m2[i].resize(size, 0.0);
  for(int i = 0; i < 10; ++i) m3[i].resize(size, 0.0);

  std::array<py::array_t<OutT>, 15> arrs;
  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 6; ++i) M.m2[i] = m2[i].data();
  for(int i = 0; i < 10; ++i) M.m3[i] = m3[i].data();
  for(int i = 0; i < 15; ++i) {
    arrs[i] = py::array_t<OutT>({nmesh, nmesh, nmesh});
    M.m4[i] = static_cast<OutT *>(arrs[i].request().ptr);
    std::fill_n(M.m4[i], size, 0.0);
  }

  deposit_moments<4, T, OutT>(pos, vel, mass, M, n, lbox, nmesh, method, nthreads);
  calc_kurtosis_components(M, size);

  py::tuple t(15);
  for(int i = 0; i < 15; ++i) t[i] = arrs[i];
  return t;
}

template <typename T, typename OutT>
static py::object kurtosis_norm_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                                     const int nmesh, const int method, const int nthreads, int norm_mode)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  std::vector<OutT> rho(size, 0.0), mx(size, 0.0), my(size, 0.0), mz(size, 0.0);
  std::vector<OutT> m2[6], m3[10], m4[15];
  for(int i = 0; i < 6; ++i) m2[i].resize(size, 0.0);
  for(int i = 0; i < 10; ++i) m3[i].resize(size, 0.0);
  for(int i = 0; i < 15; ++i) m4[i].resize(size, 0.0);

  py::array_t<OutT> norm_arr({nmesh, nmesh, nmesh});
  OutT *out_ptr = static_cast<OutT *>(norm_arr.request().ptr);

  Moments<OutT> M;
  M.m0 = rho.data();
  M.m1 = {mx.data(), my.data(), mz.data()};
  for(int i = 0; i < 6; ++i) M.m2[i] = m2[i].data();
  for(int i = 0; i < 10; ++i) M.m3[i] = m3[i].data();
  for(int i = 0; i < 15; ++i) M.m4[i] = m4[i].data();

  deposit_moments<4, T, OutT>(pos, vel, mass, M, n, lbox, nmesh, method, nthreads);
  calc_kurtosis_norm(M, out_ptr, size, norm_mode);
  return norm_arr;
}
