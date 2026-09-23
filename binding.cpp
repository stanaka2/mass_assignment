#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "domain.hpp"
#include "assignment_kernel.hpp"
#include "ptcl.hpp"
#include "mass_assign.hpp"
#include "reverse_mass_assign.hpp"
#include "mesh_finite_difference.hpp"

namespace py = pybind11;

// mass is optional. The numerical core treats a null pointer as unit mass, so mass=None does not
// allocate an O(N) array of ones. holder keeps the Python object alive for the pointer's lifetime.
template <typename T>
static const T *nullable_mass_ptr(const py::object &obj, const int64_t n, py::array_t<T, py::array::c_style> &holder)
{
  if(obj.is_none()) return nullptr;

  holder = py::cast<py::array_t<T, py::array::c_style>>(obj);
  if(holder.ndim() != 1 || holder.shape(0) != n) throw std::runtime_error("mass must have shape (N,) and match pos");

  return holder.data();
}

// --- Python Bindings ---

template <typename T>
static void bind_dtype(py::module_ &m)
{
  // --- 1. Density ---
  m.def(
      "dens",
      [](py::array_t<T, py::array::c_style> pos, py::object mass_obj, std::array<int, 3> nmesh,
         std::array<double, 3> lbox, int method, std::array<int, 3> bc, int nthreads,
         bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return density_impl<T, double>(p, grid, method, nthreads);
        return density_impl<T, float>(p, grid, method, nthreads);
      },
      py::arg("pos"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3, py::arg("bc"),
      py::arg("nthreads") = 0, py::arg("output_double") = false);

  // --- 1.1 Scalar ---
  m.def(
      "scalar",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> scalar, py::object mass_obj,
         std::array<int, 3> nmesh, std::array<double, 3> lbox, int method, std::array<int, 3> bc, int nthreads,
         bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return scalar_impl<T, double>(p, scalar.data(), grid, method, nthreads);
        return scalar_impl<T, float>(p, scalar.data(), grid, method, nthreads);
      },
      py::arg("pos"), py::arg("scalar"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3,
      py::arg("bc"), py::arg("nthreads") = 0, py::arg("output_double") = false);

  // --- 2. Velocity ---
  m.def(
      "velc",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, py::object mass_obj,
         std::array<int, 3> nmesh, std::array<double, 3> lbox, int method, std::array<int, 3> bc, int nthreads,
         bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.vel = vel.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return velocity_impl<T, double>(p, grid, method, nthreads);
        return velocity_impl<T, float>(p, grid, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3,
      py::arg("bc"), py::arg("nthreads") = 0, py::arg("output_double") = false);

  m.def(
      "velc_norm",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, py::object mass_obj,
         std::array<int, 3> nmesh, std::array<double, 3> lbox, int method, std::array<int, 3> bc, int nthreads,
         bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.vel = vel.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return velocity_norm_impl<T, double>(p, grid, method, nthreads);
        return velocity_norm_impl<T, float>(p, grid, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3,
      py::arg("bc"), py::arg("nthreads") = 0, py::arg("output_double") = false);

  // --- 3. Sigma ---
  m.def(
      "sigma",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, py::object mass_obj,
         std::array<int, 3> nmesh, std::array<double, 3> lbox, int method, std::array<int, 3> bc, int nthreads,
         bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.vel = vel.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return sigma_impl<T, double>(p, grid, method, nthreads);
        return sigma_impl<T, float>(p, grid, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3,
      py::arg("bc"), py::arg("nthreads") = 0, py::arg("output_double") = false);

  m.def(
      "sigma_norm",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, py::object mass_obj,
         std::array<int, 3> nmesh, std::array<double, 3> lbox, int method, int norm_mode, std::array<int, 3> bc,
         int nthreads, bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.vel = vel.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return sigma_norm_impl<T, double>(p, grid, method, nthreads, norm_mode);
        return sigma_norm_impl<T, float>(p, grid, method, nthreads, norm_mode);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3,
      py::arg("norm_mode") = 1, py::arg("bc"), py::arg("nthreads") = 0, py::arg("output_double") = false);

  // --- 4. Skewness ---
  m.def(
      "skewness",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, py::object mass_obj,
         std::array<int, 3> nmesh, std::array<double, 3> lbox, int method, std::array<int, 3> bc, int nthreads,
         bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.vel = vel.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return skewness_impl<T, double>(p, grid, method, nthreads);
        return skewness_impl<T, float>(p, grid, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3,
      py::arg("bc"), py::arg("nthreads") = 0, py::arg("output_double") = false);

  m.def(
      "skewness_norm",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, py::object mass_obj,
         std::array<int, 3> nmesh, std::array<double, 3> lbox, int method, int norm_mode, std::array<int, 3> bc,
         int nthreads, bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.vel = vel.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return skewness_norm_impl<T, double>(p, grid, method, nthreads, norm_mode);
        return skewness_norm_impl<T, float>(p, grid, method, nthreads, norm_mode);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3,
      py::arg("norm_mode") = 1, py::arg("bc"), py::arg("nthreads") = 0, py::arg("output_double") = false);

  // --- 5. Kurtosis ---
  m.def(
      "kurtosis",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, py::object mass_obj,
         std::array<int, 3> nmesh, std::array<double, 3> lbox, int method, std::array<int, 3> bc, int nthreads,
         bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.vel = vel.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return kurtosis_impl<T, double>(p, grid, method, nthreads);
        return kurtosis_impl<T, float>(p, grid, method, nthreads);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3,
      py::arg("bc"), py::arg("nthreads") = 0, py::arg("output_double") = false);

  m.def(
      "kurtosis_norm",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<T, py::array::c_style> vel, py::object mass_obj,
         std::array<int, 3> nmesh, std::array<double, 3> lbox, int method, int norm_mode, std::array<int, 3> bc,
         int nthreads, bool output_double) -> py::object {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        py::array_t<T, py::array::c_style> mass_holder;
        Ptcl<T> p;
        p.pos = pos.data();
        p.vel = vel.data();
        p.mass = nullable_mass_ptr<T>(mass_obj, n, mass_holder);
        p.n = n;

        if(output_double) return kurtosis_norm_impl<T, double>(p, grid, method, nthreads, norm_mode);
        return kurtosis_norm_impl<T, float>(p, grid, method, nthreads, norm_mode);
      },
      py::arg("pos"), py::arg("vel"), py::arg("mass"), py::arg("nmesh"), py::arg("lbox"), py::arg("method") = 3,
      py::arg("norm_mode") = 1, py::arg("bc"), py::arg("nthreads") = 0, py::arg("output_double") = false);

  // mesh_to_ptcl and mesh_diff take the mesh as input, so the grid shape comes from the array
  // itself and a cubic mesh is not assumed.
  m.def(
      "mesh_to_ptcl",
      [](py::array_t<T, py::array::c_style> pos, py::array_t<float, py::array::c_style> mesh,
         std::array<double, 3> lbox, int method, std::array<int, 3> bc, int nthreads) -> py::array_t<float> {
        const int64_t n = pos.shape(0);
        if(pos.ndim() != 2 || pos.shape(1) != 3) throw std::runtime_error("pos must have shape (N,3)");
        if(mesh.ndim() != 3) throw std::runtime_error("mesh must be a 3D array");

        const std::array<int, 3> nmesh = {(int)mesh.shape(0), (int)mesh.shape(1), (int)mesh.shape(2)};
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        Ptcl<T> p;
        p.pos = pos.data();
        p.n = n;

        return mesh_to_ptcl_impl<T, float>(p, mesh.data(), grid, method, nthreads);
      },
      py::arg("pos"), py::arg("mesh"), py::arg("lbox"), py::arg("method") = 3, py::arg("bc"), py::arg("nthreads") = 0);

  m.def(
      "mesh_diff",
      [](py::array_t<float, py::array::c_style> mesh, std::array<double, 3> lbox, int order, std::array<int, 3> bc,
         int nthreads) -> py::array_t<float> {
        if(mesh.ndim() != 3) throw std::runtime_error("mesh must be a 3D array");

        const std::array<int, 3> nmesh = {(int)mesh.shape(0), (int)mesh.shape(1), (int)mesh.shape(2)};
        const GridSpec grid = make_grid(nmesh, lbox, bc);

        return mesh_diff_impl<float>(mesh.data(), grid, order, nthreads);
      },
      py::arg("mesh"), py::arg("lbox"), py::arg("order") = 4, py::arg("bc"), py::arg("nthreads") = 0);
}

PYBIND11_MODULE(_binding, m)
{
  m.doc() = "Single-process mass assignment (NGP/CIC/TSC/PCS) with OpenMP support and Tensor Norms";
  bind_dtype<float>(m);
  bind_dtype<double>(m);
}
