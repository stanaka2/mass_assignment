import numpy as np
from . import _binding


def method_type(method):
    method_map = {"NGP": 1, "CIC": 2, "TSC": 3, "PCS": 4}
    if isinstance(method, str):
        m = method.strip().upper()
        if m in method_map:
            return method_map[m]
    elif isinstance(method, int):
        if method in method_map.values():
            return method
    raise ValueError("method must be one of 'NGP', 'CIC', 'TSC', 'PCS' or their corresponding integer codes 1,2,3,4")


def _normalize_pv(a, dtype):
    a = np.asarray(a, dtype=dtype)
    if a.ndim != 2: raise ValueError("pos/vel must be 2D array")
    # allow (3,N) -> (N,3)
    if a.shape[0] == 3 and a.shape[1] != 3: a = a.T
    if a.shape[1] != 3: raise ValueError("pos/vel must have shape (N,3) or (3,N)")
    return np.require(a, requirements=["C_CONTIGUOUS"])


def _normalize_m(mass, n, dtype):
    if mass is None:
        return np.ones(n, dtype=dtype)
    m = np.asarray(mass, dtype=dtype)
    if m.ndim != 1 or m.shape[0] != n:
        raise ValueError("mass must have shape (N,) and match pos")
    return np.require(m, requirements=["C_CONTIGUOUS"])


def dens(pos, nmesh, lbox=1.0, method="TSC", mass=None, nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    return _binding.dens(pos, mass, lbox, nmesh, method, nthreads)


def velc(pos, vel, nmesh, lbox=1.0, method="TSC", mass=None, nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    return _binding.velc(pos, vel, mass, lbox, nmesh, method, nthreads)


def velc_norm(pos, vel, nmesh, lbox=1.0, method="TSC", mass=None, nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    return _binding.velc_norm(pos, vel, mass, lbox, nmesh, method, nthreads)


def sigma(pos, vel, nmesh, lbox=1.0, method="TSC", mass=None, nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    return _binding.sigma(pos, vel, mass, lbox, nmesh, method, nthreads)


def sigma_norm(pos, vel, nmesh, lbox=1.0, method="TSC", norm_mode="diag_norm", mass=None, nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    norm_mode = 1 if norm_mode == "diag_norm" else 0
    return _binding.sigma_norm(pos, vel, mass, lbox, nmesh, method, norm_mode, nthreads)


def skewness(pos, vel, nmesh, lbox=1.0, method="TSC", mass=None, nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    return _binding.skewness(pos, vel, mass, lbox, nmesh, method, nthreads)


def skewness_norm(pos, vel, nmesh, lbox=1.0, method="TSC", norm_mode="diag_norm", mass=None, nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    norm_mode = 1 if norm_mode == "diag_norm" else 0
    return _binding.skewness_norm(pos, vel, mass, lbox, nmesh, method, norm_mode, nthreads)


def kurtosis(pos, vel, nmesh, lbox=1.0, method="TSC", mass=None, nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    return _binding.kurtosis(pos, vel, mass, lbox, nmesh, method, nthreads)


def kurtosis_norm(pos, vel, nmesh, lbox=1.0, method="TSC", norm_mode="diag_norm", mass=None, nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    norm_mode = 1 if norm_mode == "diag_norm" else 0
    return _binding.kurtosis_norm(pos, vel, mass, lbox, nmesh, method, norm_mode, nthreads)


def mesh_to_ptcl(pos, mesh, lbox=1.0, method="TSC", nthreads=0):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    method = method_type(method)
    return _binding.mesh_to_ptcl(pos, mesh, lbox, method, nthreads)


def mesh_diff(mesh, lbox=1.0, order=4, nthreads=0):
    return _binding.mesh_diff(mesh, lbox, order, nthreads)
