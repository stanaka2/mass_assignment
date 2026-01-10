#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "mass_assign.hpp"
#include "reverse_mass_assign.hpp"
#include "mesh_finite_difference.hpp"

namespace py = pybind11;

// --- Python Bindings ---

template <typename T>
static void bind_dtype(py::module_ &m)
{
  // --- 1. Density ---
  m.def(
      "dens",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> mass, double lbox, int nmesh,
         int method, int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        assert(pos.shape(1) == 3);

        return density_impl<T, float>(pos.data(), mass.data(), n, lbox, nmesh, method, nthreads);
      },
      py::arg("pos"), py::arg("mass"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 3,
      py::arg("nthreads") = 0);

  // --- 2. Velocity ---
  m.def(
      "velc",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel,
         py::array_t<T, py::array::c_style> mass, double lbox, int nmesh, int method, int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        assert(pos.shape(1) == 3);

        return velocity_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 3,
      py::arg("nthreads") = 0);

  m.def(
      "velc_norm",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel,
         py::array_t<T, py::array::c_style> mass, double lbox, int nmesh, int method, int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        assert(pos.shape(1) == 3);

        return velocity_norm_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 3,
      py::arg("nthreads") = 0);

  // --- 3. Sigma ---
  m.def(
      "sigma",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel,
         py::array_t<T, py::array::c_style> mass, double lbox, int nmesh, int method, int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        assert(pos.shape(1) == 3);

        return sigma_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 3,
      py::arg("nthreads") = 0);

  m.def(
      "sigma_norm",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel,
         py::array_t<T, py::array::c_style> mass, double lbox, int nmesh, int method, int norm_mode,
         int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        assert(pos.shape(1) == 3);

        return sigma_norm_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads,
                                         norm_mode);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 3,
      py::arg("norm_mode") = 1, py::arg("nthreads") = 0);

  // --- 4. Skewness ---
  m.def(
      "skewness",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel,
         py::array_t<T, py::array::c_style> mass, double lbox, int nmesh, int method, int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        assert(pos.shape(1) == 3);

        return skewness_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 3,
      py::arg("nthreads") = 0);

  m.def(
      "skewness_norm",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel,
         py::array_t<T, py::array::c_style> mass, double lbox, int nmesh, int method, int norm_mode,
         int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        assert(pos.shape(1) == 3);

        return skewness_norm_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads,
                                            norm_mode);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 3,
      py::arg("norm_mode") = 1, py::arg("nthreads") = 0);

  // --- 5. Kurtosis ---
  m.def(
      "kurtosis",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel,
         py::array_t<T, py::array::c_style> mass, double lbox, int nmesh, int method, int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        assert(pos.shape(1) == 3);

        return kurtosis_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 3,
      py::arg("nthreads") = 0);

  m.def(
      "kurtosis_norm",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel,
         py::array_t<T, py::array::c_style> mass, double lbox, int nmesh, int method, int norm_mode,
         int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        assert(pos.shape(1) == 3);

        return kurtosis_norm_impl<T, float>(pos.data(), vel.data(), mass.data(), n, lbox, nmesh, method, nthreads,
                                            norm_mode);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("lbox"), py::arg("nmesh"), py::arg("method") = 3,
      py::arg("norm_mode") = 1, py::arg("nthreads") = 0);

  m.def(
      "mesh_to_ptcl",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<float, py::array::c_style> mesh, double lbox, int method,
         int nthreads) -> py::object {
        int64_t n = pos.shape(0);
        int nmesh = mesh.shape(0);
        assert(pos.shape(1) == 3);
        assert(mesh.ndim() == 3);

        return mesh_to_ptcl_impl<T, float>(pos.data(), mesh.data(), lbox, n, nmesh, method, nthreads);
      },
      py::arg("pos"), py::arg("mesh"), py::arg("lbox"), py::arg("method") = 3, py::arg("nthreads") = 0);

  m.def(
      "mesh_diff",
      [](py::array_t<float, py::array::c_style> mesh, double lbox, int order, int nthreads) -> py::object {
        int nmesh = mesh.shape(0);
        assert(mesh.ndim() == 3);
        return mesh_diff_impl<float>(mesh.data(), nmesh, lbox, order, nthreads);
      },
      py::arg("mesh"), py::arg("lbox"), py::arg("order") = 4, py::arg("nthreads") = 0);
}

PYBIND11_MODULE(_binding, m)
{
  m.doc() = "Single-process mass assignment (NGP/CIC/TSC/PCS) with OpenMP support and Tensor Norms";
  bind_dtype<float>(m);
  bind_dtype<double>(m);
}
