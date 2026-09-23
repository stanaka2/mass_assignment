import numpy as np

from . import _binding


def method_type(method):
    method_map = {"NGP": 1, "CIC": 2, "TSC": 3, "PCS": 4}
    if isinstance(method, str):
        m = method.strip().upper()
        if m in method_map:
            return method_map[m]
    # np.integer as well as int, for an order read back from a file. bool is an int in Python,
    # which here would silently mean NGP
    elif isinstance(method, (int, np.integer)) and not isinstance(method, bool):
        if int(method) in method_map.values():
            return int(method)
    raise ValueError(
        "method must be one of 'NGP', 'CIC', 'TSC', 'PCS' or their corresponding integer codes 1,2,3,4"
    )


def _normalize_pv(a, dtype):
    a = np.asarray(a, dtype=dtype)
    if a.ndim != 2:
        raise ValueError("pos/vel must be 2D array")
    # allow (3,N) -> (N,3)
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T
    if a.shape[1] != 3:
        raise ValueError("pos/vel must have shape (N,3) or (3,N)")
    return np.require(a, requirements=["C_CONTIGUOUS"])


# None is passed straight through: the C++ side reads a null pointer as unit mass, so no O(N)
# array of ones is built
def _normalize_m(mass, n, dtype):
    if mass is None:
        return None
    m = np.asarray(mass, dtype=dtype)
    if m.ndim != 1 or m.shape[0] != n:
        raise ValueError("mass must have shape (N,) and match pos")
    return np.require(m, requirements=["C_CONTIGUOUS"])


def _normalize_s(scalar, n, dtype):
    s = np.asarray(scalar, dtype=dtype)
    if s.ndim != 1 or s.shape[0] != n:
        raise ValueError("scalar must have shape (N,) and match pos")
    return np.require(s, requirements=["C_CONTIGUOUS"])


# a scalar means the same value on all three axes
def _normalize_nmesh(nmesh):
    if np.ndim(nmesh) == 0:
        n = int(nmesh)
        if n <= 0:
            raise ValueError("nmesh must be positive")
        return (n, n, n)
    if len(nmesh) != 3:
        raise ValueError("nmesh must be a scalar or length-3 sequence")
    out = tuple(int(x) for x in nmesh)
    if any(x <= 0 for x in out):
        raise ValueError("all nmesh elements must be positive")
    return out


def _normalize_lbox(lbox):
    if np.ndim(lbox) == 0:
        L = float(lbox)
        if L <= 0.0:
            raise ValueError("lbox must be positive")
        return (L, L, L)
    if len(lbox) != 3:
        raise ValueError("lbox must be a scalar or length-3 sequence")
    out = tuple(float(x) for x in lbox)
    if any(x <= 0.0 for x in out):
        raise ValueError("all lbox elements must be positive")
    return out


# returns integer codes so that the C++ side never compares strings in a hot loop
def _normalize_bc(bc):
    if isinstance(bc, str):
        bc = (bc, bc, bc)
    if len(bc) != 3:
        raise ValueError("bc must be a string or length-3 sequence")
    out = tuple(str(x).strip().lower() for x in bc)
    if any(x not in ("periodic", "open") for x in out):
        raise ValueError("bc elements must be 'periodic' or 'open'")
    return tuple(0 if x == "periodic" else 1 for x in out)


def _output_double(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return False
    if dtype == np.dtype(np.float64):
        return True
    raise ValueError("dtype must be float32 or float64")


def dens(
    pos,
    nmesh,
    lbox=1.0,
    method="TSC",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.dens(
        pos, mass, nmesh, lbox, method, bc, nthreads, _output_double(dtype)
    )


# for non-positive scalar field
def scalar(
    pos,
    scalar,
    nmesh,
    lbox=1.0,
    method="TSC",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    n = pos.shape[0]
    scalar = _normalize_s(scalar, n, dtype=in_dtype)
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.scalar(
        pos, scalar, mass, nmesh, lbox, method, bc, nthreads, _output_double(dtype)
    )


# like a scalar[3]
def velc(
    pos,
    vel,
    nmesh,
    lbox=1.0,
    method="TSC",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]:
        raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.velc(
        pos, vel, mass, nmesh, lbox, method, bc, nthreads, _output_double(dtype)
    )


def velc_norm(
    pos,
    vel,
    nmesh,
    lbox=1.0,
    method="TSC",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]:
        raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.velc_norm(
        pos, vel, mass, nmesh, lbox, method, bc, nthreads, _output_double(dtype)
    )


def sigma(
    pos,
    vel,
    nmesh,
    lbox=1.0,
    method="TSC",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]:
        raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.sigma(
        pos, vel, mass, nmesh, lbox, method, bc, nthreads, _output_double(dtype)
    )


def sigma_norm(
    pos,
    vel,
    nmesh,
    lbox=1.0,
    method="TSC",
    norm_mode="diag_norm",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)

    if vel.shape[0] != pos.shape[0]:
        raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    norm_mode = 1 if norm_mode == "diag_norm" else 0
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.sigma_norm(
        pos,
        vel,
        mass,
        nmesh,
        lbox,
        method,
        norm_mode,
        bc,
        nthreads,
        _output_double(dtype),
    )


def skewness(
    pos,
    vel,
    nmesh,
    lbox=1.0,
    method="TSC",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]:
        raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.skewness(
        pos, vel, mass, nmesh, lbox, method, bc, nthreads, _output_double(dtype)
    )


def skewness_norm(
    pos,
    vel,
    nmesh,
    lbox=1.0,
    method="TSC",
    norm_mode="diag_norm",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]:
        raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)

    method = method_type(method)
    norm_mode = 1 if norm_mode == "diag_norm" else 0
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.skewness_norm(
        pos,
        vel,
        mass,
        nmesh,
        lbox,
        method,
        norm_mode,
        bc,
        nthreads,
        _output_double(dtype),
    )


def kurtosis(
    pos,
    vel,
    nmesh,
    lbox=1.0,
    method="TSC",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]:
        raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.kurtosis(
        pos, vel, mass, nmesh, lbox, method, bc, nthreads, _output_double(dtype)
    )


def kurtosis_norm(
    pos,
    vel,
    nmesh,
    lbox=1.0,
    method="TSC",
    norm_mode="diag_norm",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    vel = _normalize_pv(vel, dtype=in_dtype)
    if vel.shape[0] != pos.shape[0]:
        raise ValueError("vel must have shape (N,3) and match pos")

    n = pos.shape[0]
    mass = _normalize_m(mass, n, dtype=in_dtype)
    method = method_type(method)
    norm_mode = 1 if norm_mode == "diag_norm" else 0
    nmesh = _normalize_nmesh(nmesh)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.kurtosis_norm(
        pos,
        vel,
        mass,
        nmesh,
        lbox,
        method,
        norm_mode,
        bc,
        nthreads,
        _output_double(dtype),
    )


# the mesh shape gives the grid, so a cubic mesh is not assumed
def mesh_to_ptcl(pos, mesh, lbox=1.0, method="TSC", nthreads=0, *, bc="periodic"):
    in_dtype = np.asarray(pos).dtype
    pos = _normalize_pv(pos, dtype=in_dtype)
    method = method_type(method)
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.mesh_to_ptcl(pos, mesh, lbox, method, bc, nthreads)


def mesh_diff(mesh, lbox=1.0, order=4, nthreads=0, *, bc="periodic"):
    lbox = _normalize_lbox(lbox)
    bc = _normalize_bc(bc)
    return _binding.mesh_diff(mesh, lbox, order, bc, nthreads)


MOMENT_COMPONENTS = {
    2: ("xx", "xy", "xz", "yy", "yz", "zz"),
    3: ("xxx", "xxy", "xxz", "xyy", "xyz", "xzz", "yyy", "yyz", "yzz", "zzz"),
    4: (
        "xxxx",
        "xxxy",
        "xxxz",
        "xxyy",
        "xxyz",
        "xxzz",
        "xyyy",
        "xyyz",
        "xyzz",
        "xzzz",
        "yyyy",
        "yyyz",
        "yyzz",
        "yzzz",
        "zzzz",
    ),
}


def moment_components(order):
    """Return the Cartesian component labels for a central-moment order."""
    try:
        return MOMENT_COMPONENTS[int(order)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("order must be 2, 3, or 4") from exc


def central_moment(
    pos,
    vel,
    order,
    nmesh,
    lbox=1.0,
    method="TSC",
    mass=None,
    nthreads=0,
    *,
    bc="periodic",
    dtype=np.float32,
):
    """Compute all independent components of a normalized central moment."""
    functions = {2: sigma, 3: skewness, 4: kurtosis}
    try:
        function = functions[int(order)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("order must be 2, 3, or 4") from exc
    return function(
        pos,
        vel,
        nmesh,
        lbox=lbox,
        method=method,
        mass=mass,
        nthreads=nthreads,
        bc=bc,
        dtype=dtype,
    )


# Clear names matching snapshot datasets. The established public names remain
# exact aliases, so existing programs do not need to change.
moment2 = sigma
moment3 = skewness
moment4 = kurtosis


# Reverse of method_type, so a Grid can report back what it was given.
_METHOD_NAME = {1: "NGP", 2: "CIC", 3: "TSC", 4: "PCS"}


class Grid:
    """Mesh geometry and assignment settings, held once instead of repeated on every call.

    nmesh, lbox, method and bc do not usually change within one analysis, and repeating them on
    each call is a chance for them to drift apart without any error being raised. A Grid keeps them
    together and exposes them, so the same values can be handed to whatever consumes the mesh.

        g = ma.Grid(nmesh=512, lbox=1000.0, method="TSC")
        rho = g.dens(pos)
        vx, vy, vz = g.velc(pos, vel)

    The module level functions keep working unchanged; this only groups their settings.

    lbox is in the same units as the positions that will be deposited. It is a declaration: nothing
    about the particles can confirm it, and getting it wrong is not an error, only a different box.
    """

    __slots__ = ("bc", "dtype", "lbox", "method", "nmesh", "nthreads", "p")

    def __init__(
        self, nmesh, lbox=1.0, method="TSC", bc="periodic", nthreads=0, dtype=np.float32
    ):
        self.nmesh = _normalize_nmesh(nmesh)
        self.lbox = _normalize_lbox(lbox)
        self.p = method_type(method)
        self.method = _METHOD_NAME[self.p]
        codes = _normalize_bc(bc)
        self.bc = tuple("periodic" if c == 0 else "open" for c in codes)
        self.nthreads = int(nthreads)
        _output_double(dtype)
        self.dtype = np.dtype(dtype)

    def _kw(self):
        return {
            "nmesh": self.nmesh,
            "lbox": self.lbox,
            "method": self.p,
            "nthreads": self.nthreads,
            "bc": self.bc,
            "dtype": self.dtype,
        }

    def dens(self, pos, mass=None):
        return dens(pos, mass=mass, **self._kw())

    def scalar(self, pos, values, mass=None):
        return scalar(pos, values, mass=mass, **self._kw())

    def velc(self, pos, vel, mass=None):
        return velc(pos, vel, mass=mass, **self._kw())

    def velc_norm(self, pos, vel, mass=None):
        return velc_norm(pos, vel, mass=mass, **self._kw())

    def sigma(self, pos, vel, mass=None):
        return sigma(pos, vel, mass=mass, **self._kw())

    def sigma_norm(self, pos, vel, norm_mode="diag_norm", mass=None):
        return sigma_norm(pos, vel, norm_mode=norm_mode, mass=mass, **self._kw())

    def skewness(self, pos, vel, mass=None):
        return skewness(pos, vel, mass=mass, **self._kw())

    def skewness_norm(self, pos, vel, norm_mode="diag_norm", mass=None):
        return skewness_norm(pos, vel, norm_mode=norm_mode, mass=mass, **self._kw())

    def kurtosis(self, pos, vel, mass=None):
        return kurtosis(pos, vel, mass=mass, **self._kw())

    def kurtosis_norm(self, pos, vel, norm_mode="diag_norm", mass=None):
        return kurtosis_norm(pos, vel, norm_mode=norm_mode, mass=mass, **self._kw())

    def central_moment(self, pos, vel, order, mass=None):
        return central_moment(pos, vel, order, mass=mass, **self._kw())

    def moment2(self, pos, vel, mass=None):
        return self.sigma(pos, vel, mass=mass)

    def moment3(self, pos, vel, mass=None):
        return self.skewness(pos, vel, mass=mass)

    def moment4(self, pos, vel, mass=None):
        return self.kurtosis(pos, vel, mass=mass)

    def shotnoise(self, pos, mass=None, lbox=None):
        """Poisson term of the density contrast built from these particles.

            V * sum(w^2) / sum(w)^2

        which is V / N for equal weights. The rest comes from the particles that were deposited, so
        only the volume has to be agreed on.

        V is this Grid's box unless lbox says otherwise. The override is here because this lbox may
        be a normalization rather than a length: positions scaled to [0,1) are described by lbox=1,
        and then this returns the shot noise in those same units. Handing that number to a power
        spectrum measured in Mpc/h would be wrong by the volume ratio, and nothing would say so.

            g = ma.Grid(nmesh=512, lbox=1.0, method="TSC")   # positions in [0,1)
            sn = g.shotnoise(pos, lbox=1000.0)               # Mpc/h, to match the spectrum

        Safer still is not to carry the number across at all. field2pt takes npart= or weights= and
        works the volume out from the box it is measuring in, which cannot disagree with itself.

        This is the leading term. Close to the Nyquist frequency the aliased images make the true
        shot noise k-dependent, and interlacing is what removes that.
        """
        # through the same normalization the deposit uses, so that (3,N) counts N particles here
        # too rather than 3
        n = _normalize_pv(pos, np.float64).shape[0]
        box = self.lbox if lbox is None else _normalize_lbox(lbox)
        volume = box[0] * box[1] * box[2]
        if mass is None:
            if n == 0:
                raise ValueError("there are no particles to take a shot noise from")
            return volume / float(n)
        w = np.asarray(mass, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != n:
            raise ValueError("mass must have shape (N,) and match pos")
        total = w.sum()
        if total <= 0.0:
            raise ValueError("the total weight must be positive")
        return volume * float((w * w).sum()) / float(total * total)

    def mesh_to_ptcl(self, pos, mesh):
        return mesh_to_ptcl(
            pos, mesh, lbox=self.lbox, method=self.p, nthreads=self.nthreads, bc=self.bc
        )

    def mesh_diff(self, mesh, order=4):
        return mesh_diff(
            mesh, lbox=self.lbox, order=order, nthreads=self.nthreads, bc=self.bc
        )

    def __repr__(self):
        return (
            f"Grid(nmesh={self.nmesh}, lbox={self.lbox}, method='{self.method}' (p={self.p}), "
            f"bc={self.bc}, nthreads={self.nthreads}, dtype='{self.dtype.name}')"
        )
