try:
    from importlib.metadata import version
    __version__ = version("mass-assignment")
except Exception:
    __version__ = "unknown"

from ._wrapper import dens, scalar, velc, sigma, skewness, kurtosis
from ._wrapper import velc_norm, sigma_norm, skewness_norm, kurtosis_norm
from ._wrapper import mesh_to_ptcl, mesh_diff

__all__ = ["dens", "scalar", "velc", "sigma", "skewness", "kurtosis"]
__all__ += ["velc_norm", "sigma_norm", "skewness_norm", "kurtosis_norm"]
__all__ += ["mesh_to_ptcl", "mesh_diff"]
