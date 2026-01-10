#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

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

static inline int64_t idx3(int i, int j, int k, int n)
{
  return (int64_t)k + (int64_t)n * ((int64_t)j + (int64_t)n * (int64_t)i);
}

static inline int wrap_pm_index(int i, int n)
{
  if(i >= n) i -= n;
  if(i < 0) i += n;
  return i;
}

template <int ACC, typename T>
void build_grad_mesh_periodic(const T *mesh, T *retx, T *rety, T *retz, const int nmesh, const double lbox,
                              const int nthreads)
{
  static_assert(ACC == 2 || ACC == 4 || ACC == 6 || ACC == 8, "ACC must be 2,4,6,8");

#ifdef _OPENMP
  if(nthreads > 0) omp_set_num_threads(nthreads);
#endif

  const double dx = lbox / (double)nmesh;
  const double inv_dx = 1.0 / dx;
  constexpr int r = FD<ACC>::r;

  {
    py::gil_scoped_release release;

#pragma omp parallel for collapse(3) schedule(static)
    for(int i = 0; i < nmesh; i++) {
      for(int j = 0; j < nmesh; j++) {
        for(int k = 0; k < nmesh; k++) {

          double sx = 0.0;
          double sy = 0.0;
          double sz = 0.0;

          // X derivative
          for(int s = 1; s <= r; s++) {
            const int ip = wrap_pm_index(i + s, nmesh);
            const int im = wrap_pm_index(i - s, nmesh);
            const double fp = (double)mesh[idx3(ip, j, k, nmesh)];
            const double fm = (double)mesh[idx3(im, j, k, nmesh)];
            sx += FD<ACC>::c[s - 1] * (fp - fm);
          }

          // Y derivative
          for(int s = 1; s <= r; s++) {
            const int jp = wrap_pm_index(j + s, nmesh);
            const int jm = wrap_pm_index(j - s, nmesh);
            const double fp = (double)mesh[idx3(i, jp, k, nmesh)];
            const double fm = (double)mesh[idx3(i, jm, k, nmesh)];
            sy += FD<ACC>::c[s - 1] * (fp - fm);
          }

          // Z derivative
          for(int s = 1; s <= r; s++) {
            const int kp = wrap_pm_index(k + s, nmesh);
            const int km = wrap_pm_index(k - s, nmesh);
            const double fp = (double)mesh[idx3(i, j, kp, nmesh)];
            const double fm = (double)mesh[idx3(i, j, km, nmesh)];
            sz += FD<ACC>::c[s - 1] * (fp - fm);
          }

          const int64_t idx = idx3(i, j, k, nmesh);

          retx[idx] = (T)(sx * inv_dx);
          rety[idx] = (T)(sy * inv_dx);
          retz[idx] = (T)(sz * inv_dx);
        }
      }
    }
  } // gil release
}

template <typename T>
static py::object mesh_diff_impl(const T *mesh, const int nmesh, const double lbox, const int order, const int nthreads)
{
  const int64_t size = (int64_t)nmesh * (int64_t)nmesh * (int64_t)nmesh;
  py::array_t<T> diff_x({nmesh, nmesh, nmesh}), diff_y({nmesh, nmesh, nmesh}), diff_z({nmesh, nmesh, nmesh});
  T *di_x = static_cast<T *>(diff_x.request().ptr);
  T *di_y = static_cast<T *>(diff_y.request().ptr);
  T *di_z = static_cast<T *>(diff_z.request().ptr);
  std::fill_n(di_x, size, 0.0);
  std::fill_n(di_y, size, 0.0);
  std::fill_n(di_z, size, 0.0);

  switch(order) {
  case 2:
    build_grad_mesh_periodic<2>(mesh, di_x, di_y, di_z, nmesh, lbox, nthreads);
    break;
  case 4:
    build_grad_mesh_periodic<4>(mesh, di_x, di_y, di_z, nmesh, lbox, nthreads);
    break;
  case 6:
    build_grad_mesh_periodic<6>(mesh, di_x, di_y, di_z, nmesh, lbox, nthreads);
    break;
  case 8:
    build_grad_mesh_periodic<8>(mesh, di_x, di_y, di_z, nmesh, lbox, nthreads);
    break;
  default:
    throw std::runtime_error("order must be 2,4,6,8");
  }

  return py::make_tuple(diff_x, diff_y, diff_z);
}
