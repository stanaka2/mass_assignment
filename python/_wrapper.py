import numpy as np
from . import _mass_assign_core as mac


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


def _parse_out_dtype(dtype):
    if isinstance(dtype, str):
        s = dtype.strip().lower()
        if s in ("f4", "float32"):
            return "f4"
        if s in ("f8", "float64"):
            return "f8"

    try:
        dt = np.dtype(dtype)
    except Exception as e:
        raise TypeError("dtype must be 'f4'/'float32' or 'f8'/'float64' or np.float32/np.float64") from e

    if dt == np.dtype(np.float32):
        return "f4"
    if dt == np.dtype(np.float64):
        return "f8"
    raise TypeError("dtype must be 'f4'/'float32' or 'f8'/'float64' or np.float32/np.float64")


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


def dens(pos, lbox, nmesh, method="TSC", mass=None, nthreads=0, dtype="f4"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    dtype = _parse_out_dtype(dtype)
    return mac.dens(pos, mass, lbox, nmesh, method, nthreads, dtype)


def velc(pos, vel, lbox, nmesh, method="TSC", mass=None, nthreads=0, dtype="f4"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    dtype = _parse_out_dtype(dtype)
    return mac.velc(pos, vel, mass, lbox, nmesh, method, nthreads, dtype)


def velc_norm(pos, vel, lbox, nmesh, method="TSC", mass=None, nthreads=0, dtype="f4"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    dtype = _parse_out_dtype(dtype)
    return mac.velc_norm(pos, vel, mass, lbox, nmesh, method, nthreads, dtype)


def sigma(pos, vel, lbox, nmesh, method="TSC", mass=None, nthreads=0, dtype="f4"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    dtype = _parse_out_dtype(dtype)
    return mac.sigma(pos, vel, mass, lbox, nmesh, method, nthreads, dtype)


def sigma_norm(pos, vel, lbox, nmesh, method="TSC", norm_mode="diag_norm", mass=None, nthreads=0, dtype="f4"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    dtype = _parse_out_dtype(dtype)
    norm_mode = 1 if norm_mode == "diag_norm" else 0
    return mac.sigma_norm(pos, vel, mass, lbox, nmesh, method, norm_mode, nthreads, dtype)


def skewness(pos, vel, lbox, nmesh, method="TSC", mass=None, nthreads=0, dtype="f4"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    dtype = _parse_out_dtype(dtype)
    return mac.skewness(pos, vel, mass, lbox, nmesh, method, nthreads, dtype)


def skewness_norm(pos, vel, lbox, nmesh, method="TSC", norm_mode="diag_norm", mass=None, nthreads=0, dtype="f4"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    dtype = _parse_out_dtype(dtype)
    norm_mode = 1 if norm_mode == "diag_norm" else 0
    return mac.skewness_norm(pos, vel, mass, lbox, nmesh, method, norm_mode, nthreads, dtype)


def kurtosis(pos, vel, lbox, nmesh, method="TSC", mass=None, nthreads=0, dtype="f4"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    dtype = _parse_out_dtype(dtype)
    return mac.kurtosis(pos, vel, mass, lbox, nmesh, method, nthreads, dtype)


def kurtosis_norm(pos, vel, lbox, nmesh, method="TSC", norm_mode="diag_norm", mass=None, nthreads=0, dtype="f4"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]: raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    dtype = _parse_out_dtype(dtype)
    norm_mode = 1 if norm_mode == "diag_norm" else 0
    return mac.kurtosis_norm(pos, vel, mass, lbox, nmesh, method, norm_mode, nthreads, dtype)
