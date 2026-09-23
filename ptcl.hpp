#pragma once

#include <cstdint>

// Non-owning view over a NumPy (N,3) C-contiguous buffer. The C++ side never owns particles and
// never copies them into a separate layout: it reads the caller's buffer as it is.
//
// vel and mass may be null. A null mass means unit mass, so mass=None costs no allocation.
//
// No runtime stride is kept here on purpose. The public API is (N,3) C contiguous, and a stride
// in the hot loop would cost more than it buys. If a benchmark ever shows SoA or AoSoA to be
// faster, add a separate view type and dispatch on it at compile time.
template <typename T>
struct Ptcl {
  const T *pos = nullptr;
  const T *vel = nullptr;
  const T *mass = nullptr;
  int64_t n = 0;

  inline double x(const int64_t i) const
  {
    return (double)pos[3 * i + 0];
  }
  inline double y(const int64_t i) const
  {
    return (double)pos[3 * i + 1];
  }
  inline double z(const int64_t i) const
  {
    return (double)pos[3 * i + 2];
  }

  inline double vx(const int64_t i) const
  {
    return (double)vel[3 * i + 0];
  }
  inline double vy(const int64_t i) const
  {
    return (double)vel[3 * i + 1];
  }
  inline double vz(const int64_t i) const
  {
    return (double)vel[3 * i + 2];
  }

  inline double pmass(const int64_t i) const
  {
    return mass ? (double)mass[i] : 1.0;
  }
};
