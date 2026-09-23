#pragma once

#include <cstdint>
#include <array>
#include <stdexcept>

// Global physical boundary condition of the domain.
// This is not the edge of a local array: a future MPI rank boundary is not an open boundary.
enum class BoundaryType : uint8_t {
  Periodic = 0,
  Open = 1,
};

// Global physical domain. Holds no rank, halo or ownership information.
struct DomainSpec {
  std::array<double, 3> lbox{};
  std::array<BoundaryType, 3> bc{};
};

// Cartesian mesh laid over a DomainSpec. Backends that are not mesh based use DomainSpec alone.
struct GridSpec {
  DomainSpec domain{};
  std::array<int, 3> n{};

  inline int64_t size() const
  {
    return (int64_t)n[0] * (int64_t)n[1] * (int64_t)n[2];
  }

  inline double lbox(const int axis) const
  {
    return domain.lbox[axis];
  }

  inline double inv_dx(const int axis) const
  {
    return (double)n[axis] / domain.lbox[axis];
  }

  inline BoundaryType bc(const int axis) const
  {
    return domain.bc[axis];
  }
};

static inline int64_t idx3(const int ix, const int iy, const int iz, const std::array<int, 3> &n)
{
  return (int64_t)iz + (int64_t)n[2] * ((int64_t)iy + (int64_t)n[1] * (int64_t)ix);
}

static inline BoundaryType decode_bc(const int code)
{
  if(code == 0) return BoundaryType::Periodic;
  if(code == 1) return BoundaryType::Open;
  throw std::runtime_error("bc must be 0(periodic) or 1(open)");
}

static inline GridSpec make_grid(const std::array<int, 3> &nmesh, const std::array<double, 3> &lbox,
                                 const std::array<int, 3> &bc)
{
  for(int j = 0; j < 3; j++) {
    if(nmesh[j] <= 0) throw std::runtime_error("nmesh must be positive");
    if(lbox[j] <= 0.0) throw std::runtime_error("lbox must be positive");
  }

  GridSpec grid;
  grid.domain.lbox = lbox;
  grid.domain.bc = {decode_bc(bc[0]), decode_bc(bc[1]), decode_bc(bc[2])};
  grid.n = nmesh;
  return grid;
}
