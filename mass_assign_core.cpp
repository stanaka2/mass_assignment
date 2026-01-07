#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

constexpr double eps = 1e-30;

static inline double wrap_periodic(double x, const double lbox)
{
  // map to [0, lbox)
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
    int ic = static_cast<int>(std::floor(xg + 0.5)); // need +0.5
    ic = (ic % nmesh + nmesh) % nmesh;
    idx[0] = ic;
    w[0] = 1.0;
    return 1;
  }

  if(method == 2) {
    // CIC
    int i0 = static_cast<int>(std::floor(xg));
    double f = xg - static_cast<double>(i0);
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
    int ic = static_cast<int>(std::floor(xg + 0.5)); // need +0.5
    double d = xg - static_cast<double>(ic);
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
    int i0 = static_cast<int>(std::floor(xg));
    double u = xg - static_cast<double>(i0); // [0,1)
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

  throw std::runtime_error("method must be 1(NGP),2(CIC),3(TSC),4(PCS)");
}

template <typename T, typename OutT>
static py::array_t<OutT> density_impl(const T *pos, const T *mass, const int64_t n, const double lbox, const int nmesh,
                                      const int method, const int nthreads)
{
  py::array_t<OutT> rho_arr({nmesh, nmesh, nmesh});
  auto rho_info = rho_arr.request();
  auto *rho = static_cast<OutT *>(rho_info.ptr);

  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;
  for(int64_t i = 0; i < size; i++) rho[i] = static_cast<OutT>(0);

#ifdef _OPENMP
  if(nthreads > 0) omp_set_num_threads(nthreads);
#endif

  const double inv_dx = static_cast<double>(nmesh) / lbox;

  {
    py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
    for(int64_t ip = 0; ip < n; ip++) {
      const double x = wrap_periodic(pos[3 * ip + 0], lbox);
      const double y = wrap_periodic(pos[3 * ip + 1], lbox);
      const double z = wrap_periodic(pos[3 * ip + 2], lbox);
      const double m = mass[ip];

      const double xg = x * inv_dx;
      const double yg = y * inv_dx;
      const double zg = z * inv_dx;

      int idx_x[4], idx_y[4], idx_z[4];
      double w_x[4], w_y[4], w_z[4];

      const int nax = assign_axis_1d(xg, nmesh, method, idx_x, w_x);
      const int nay = assign_axis_1d(yg, nmesh, method, idx_y, w_y);
      const int naz = assign_axis_1d(zg, nmesh, method, idx_z, w_z);

      for(int ix = 0; ix < nax; ix++) {
        for(int iy = 0; iy < nay; iy++) {
          const double wxy = w_x[ix] * w_y[iy];
          for(int iz = 0; iz < naz; iz++) {
            const double w = wxy * w_z[iz] * m;
            const int64_t idx =
                static_cast<int64_t>(idx_z[iz]) +
                static_cast<int64_t>(nmesh) *
                    (static_cast<int64_t>(idx_y[iy]) + static_cast<int64_t>(nmesh) * static_cast<int64_t>(idx_x[ix]));

#pragma omp atomic update
            rho[idx] += static_cast<OutT>(w);
          }
        }
      }
    }
  }

  return rho_arr;
}

template <typename T, typename OutT>
static py::tuple velocity_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                               const int nmesh, const int method, const int nthreads)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;

  std::vector<OutT> rho(size);

  py::array_t<OutT> vx_arr({nmesh, nmesh, nmesh});
  py::array_t<OutT> vy_arr({nmesh, nmesh, nmesh});
  py::array_t<OutT> vz_arr({nmesh, nmesh, nmesh});
  auto vx_info = vx_arr.request();
  auto vy_info = vy_arr.request();
  auto vz_info = vz_arr.request();
  OutT *vxo = static_cast<OutT *>(vx_info.ptr);
  OutT *vyo = static_cast<OutT *>(vy_info.ptr);
  OutT *vzo = static_cast<OutT *>(vz_info.ptr);

  for(int64_t i = 0; i < size; i++) {
    rho[i] = static_cast<OutT>(0);
    vxo[i] = static_cast<OutT>(0);
    vyo[i] = static_cast<OutT>(0);
    vzo[i] = static_cast<OutT>(0);
  }

#ifdef _OPENMP
  if(nthreads > 0) omp_set_num_threads(nthreads);
#endif

  const double inv_dx = static_cast<double>(nmesh) / lbox;

  {
    py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
    for(int64_t ip = 0; ip < n; ip++) {
      const double x = wrap_periodic(pos[3 * ip + 0], lbox);
      const double y = wrap_periodic(pos[3 * ip + 1], lbox);
      const double z = wrap_periodic(pos[3 * ip + 2], lbox);
      const double vx = vel[3 * ip + 0];
      const double vy = vel[3 * ip + 1];
      const double vz = vel[3 * ip + 2];
      const double m = mass[ip];

      const double xg = x * inv_dx;
      const double yg = y * inv_dx;
      const double zg = z * inv_dx;

      int idx_x[4], idx_y[4], idx_z[4];
      double w_x[4], w_y[4], w_z[4];

      const int nax = assign_axis_1d(xg, nmesh, method, idx_x, w_x);
      const int nay = assign_axis_1d(yg, nmesh, method, idx_y, w_y);
      const int naz = assign_axis_1d(zg, nmesh, method, idx_z, w_z);

      for(int ix = 0; ix < nax; ix++) {
        for(int iy = 0; iy < nay; iy++) {
          const double wxy = w_x[ix] * w_y[iy];
          for(int iz = 0; iz < naz; iz++) {
            const double w = wxy * w_z[iz] * m;
            const int64_t idx =
                static_cast<int64_t>(idx_z[iz]) +
                static_cast<int64_t>(nmesh) *
                    (static_cast<int64_t>(idx_y[iy]) + static_cast<int64_t>(nmesh) * static_cast<int64_t>(idx_x[ix]));

#pragma omp atomic update
            rho[idx] += static_cast<OutT>(w);
#pragma omp atomic update
            vxo[idx] += static_cast<OutT>(w * vx);
#pragma omp atomic update
            vyo[idx] += static_cast<OutT>(w * vy);
#pragma omp atomic update
            vzo[idx] += static_cast<OutT>(w * vz);
          }
        }
      }
    }

#pragma omp parallel for schedule(static)
    for(int64_t i = 0; i < size; i++) {
      vxo[i] = vxo[i] / (rho[i] + eps);
      vyo[i] = vyo[i] / (rho[i] + eps);
      vzo[i] = vzo[i] / (rho[i] + eps);
    }
  }

  return py::make_tuple(vx_arr, vy_arr, vz_arr);
}

template <typename T, typename OutT>
static py::tuple velocity_sigma_impl(const T *pos, const T *vel, const T *mass, const int64_t n, const double lbox,
                                     const int nmesh, const int method, const int nthreads)
{
  const int64_t size = static_cast<int64_t>(nmesh) * nmesh * nmesh;

  std::vector<OutT> rho(size);
  std::vector<OutT> px(size);
  std::vector<OutT> py(size);
  std::vector<OutT> pz(size);

  // raw 2nd moments: <v_i v_j> * rho  (symmetric 6 components)
  py::array_t<OutT> mxx_arr({nmesh, nmesh, nmesh});
  py::array_t<OutT> mxy_arr({nmesh, nmesh, nmesh});
  py::array_t<OutT> mxz_arr({nmesh, nmesh, nmesh});
  py::array_t<OutT> myy_arr({nmesh, nmesh, nmesh});
  py::array_t<OutT> myz_arr({nmesh, nmesh, nmesh});
  py::array_t<OutT> mzz_arr({nmesh, nmesh, nmesh});
  auto mxx_info = mxx_arr.request();
  auto mxy_info = mxy_arr.request();
  auto mxz_info = mxz_arr.request();
  auto myy_info = myy_arr.request();
  auto myz_info = myz_arr.request();
  auto mzz_info = mzz_arr.request();
  OutT *mxx = static_cast<OutT *>(mxx_info.ptr);
  OutT *mxy = static_cast<OutT *>(mxy_info.ptr);
  OutT *mxz = static_cast<OutT *>(mxz_info.ptr);
  OutT *myy = static_cast<OutT *>(myy_info.ptr);
  OutT *myz = static_cast<OutT *>(myz_info.ptr);
  OutT *mzz = static_cast<OutT *>(mzz_info.ptr);

  for(int64_t i = 0; i < size; i++) {
    rho[i] = px[i] = py[i] = pz[i] = static_cast<OutT>(0);
    mxx[i] = mxy[i] = mxz[i] = myy[i] = myz[i] = mzz[i] = static_cast<OutT>(0);
  }

#ifdef _OPENMP
  if(nthreads > 0) omp_set_num_threads(nthreads);
#endif

  const double inv_dx = static_cast<double>(nmesh) / lbox;

  {
    py::gil_scoped_release release;

#pragma omp parallel for schedule(static)
    for(int64_t ip = 0; ip < n; ip++) {
      const double x = wrap_periodic(pos[3 * ip + 0], lbox);
      const double y = wrap_periodic(pos[3 * ip + 1], lbox);
      const double z = wrap_periodic(pos[3 * ip + 2], lbox);
      const double xg = x * inv_dx;
      const double yg = y * inv_dx;
      const double zg = z * inv_dx;

      int idx_x[4], idx_y[4], idx_z[4];
      double w_x[4], w_y[4], w_z[4];

      const int nax = assign_axis_1d(xg, nmesh, method, idx_x, w_x);
      const int nay = assign_axis_1d(yg, nmesh, method, idx_y, w_y);
      const int naz = assign_axis_1d(zg, nmesh, method, idx_z, w_z);

      const double m = mass[ip];
      const double vx = static_cast<double>(vel[3 * ip + 0]);
      const double vy = static_cast<double>(vel[3 * ip + 1]);
      const double vz = static_cast<double>(vel[3 * ip + 2]);

      for(int ix = 0; ix < nax; ix++) {
        for(int iy = 0; iy < nay; iy++) {
          const double wxy = w_x[ix] * w_y[iy];
          for(int iz = 0; iz < naz; iz++) {
            const double ww = wxy * w_z[iz] * m;
            const int64_t idx =
                static_cast<int64_t>(idx_z[iz]) +
                static_cast<int64_t>(nmesh) *
                    (static_cast<int64_t>(idx_y[iy]) + static_cast<int64_t>(nmesh) * static_cast<int64_t>(idx_x[ix]));

#pragma omp atomic update
            rho[idx] += static_cast<OutT>(ww);
#pragma omp atomic update
            px[idx] += static_cast<OutT>(ww * vx);
#pragma omp atomic update
            py[idx] += static_cast<OutT>(ww * vy);
#pragma omp atomic update
            pz[idx] += static_cast<OutT>(ww * vz);

#pragma omp atomic update
            mxx[idx] += static_cast<OutT>(ww * vx * vx);
#pragma omp atomic update
            mxy[idx] += static_cast<OutT>(ww * vx * vy);
#pragma omp atomic update
            mxz[idx] += static_cast<OutT>(ww * vx * vz);
#pragma omp atomic update
            myy[idx] += static_cast<OutT>(ww * vy * vy);
#pragma omp atomic update
            myz[idx] += static_cast<OutT>(ww * vy * vz);
#pragma omp atomic update
            mzz[idx] += static_cast<OutT>(ww * vz * vz);
          }
        }
      }
    }

#pragma omp parallel for schedule(static)
    for(int64_t i = 0; i < size; i++) {
      const OutT inv = static_cast<OutT>(1) / (rho[i] + eps);
      const OutT ux = px[i] * inv;
      const OutT uy = py[i] * inv;
      const OutT uz = pz[i] * inv;

      // <v_i v_j>
      const OutT exx = mxx[i] * inv;
      const OutT exy = mxy[i] * inv;
      const OutT exz = mxz[i] * inv;
      const OutT eyy = myy[i] * inv;
      const OutT eyz = myz[i] * inv;
      const OutT ezz = mzz[i] * inv;

      // central 2nd moment (velocity dispersion tensor)
      mxx[i] = exx - ux * ux;
      mxy[i] = exy - ux * uy;
      mxz[i] = exz - ux * uz;
      myy[i] = eyy - uy * uy;
      myz[i] = eyz - uy * uz;
      mzz[i] = ezz - uz * uz;
    }
  }

  return py::make_tuple(mxx_arr, mxy_arr, mxz_arr, myy_arr, myz_arr, mzz_arr);
}

enum class OutKind { F32, F64 };

static inline OutKind parse_out_kind(const std::string &dtype)
{
  if(dtype == "f4" || dtype == "float32") return OutKind::F32;
  if(dtype == "f8" || dtype == "float64") return OutKind::F64;
  throw std::runtime_error("dtype must be 'f4'/'float32' or 'f8'/'float64'");
}

template <typename T>
static void bind_dtype(py::module_ &m)
{
  // dens
  m.def(
      "dens",
      [](py::array_t<T, py::array::c_style> pos, double lbox, int nmesh, int method,
         py::array_t<T, py::array::c_style> mass, int nthreads, std::string dtype) -> py::array {
        const int64_t n = static_cast<int64_t>(pos.shape(0));

        const auto outk = parse_out_kind(dtype);
        if(outk == OutKind::F32) {
          return density_impl<T, float>(pos.data(), mass.data(), n, lbox, nmesh, method, nthreads);
        } else {
          return density_impl<T, double>(pos.data(), mass.data(), n, lbox, nmesh, method, nthreads);
        }
      },
      py::arg("pos"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 2, py::arg("mass"),
      py::arg("nthreads") = 0, py::arg("dtype") = "f4");

  // velc
  m.def(
      "velc",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, double lbox, int nmesh,
         int method, py::array_t<T, py::array::c_style> mass, int nthreads, std::string dtype) -> py::tuple {
        const int64_t n = static_cast<int64_t>(pos.shape(0));
        const auto outk = parse_out_kind(dtype);
        if(outk == OutKind::F32) {
          return velocity_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads);
        } else {
          return velocity_impl<T, double>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads);
        }
      },
      py::arg("pos"), py::arg("vel"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 2, py::arg("mass"),
      py::arg("nthreads") = 0, py::arg("dtype") = "f4");

  // sigma
  m.def(
      "sigma",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, double lbox, int nmesh,
         int method, py::array_t<T, py::array::c_style> mass, int nthreads, std::string dtype) -> py::tuple {
        const int64_t n = static_cast<int64_t>(pos.shape(0));
        const auto outk = parse_out_kind(dtype);
        if(outk == OutKind::F32) {
          return velocity_sigma_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads);
        } else {
          return velocity_sigma_impl<T, double>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads);
        }
      },
      py::arg("pos"), py::arg("vel"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 2, py::arg("mass"),
      py::arg("nthreads") = 0, py::arg("dtype") = "f4");
}

PYBIND11_MODULE(_mass_assign_core, m)
{
  m.doc() = "Single-process mass assignment (NGP/CIC/TSC/PCS) with OpenMP support";

  bind_dtype<float>(m);
  bind_dtype<double>(m);
}
