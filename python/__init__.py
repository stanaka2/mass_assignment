try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("mass-assignment")
except PackageNotFoundError:
    __version__ = "unknown"

from ._wrapper import (
    MOMENT_COMPONENTS,
    Grid,
    central_moment,
    dens,
    kurtosis,
    kurtosis_norm,
    mesh_diff,
    mesh_to_ptcl,
    moment2,
    moment3,
    moment4,
    moment_components,
    scalar,
    sigma,
    sigma_norm,
    skewness,
    skewness_norm,
    velc,
    velc_norm,
)

__all__ = ["dens", "kurtosis", "scalar", "sigma", "skewness", "velc"]
__all__ += ["kurtosis_norm", "sigma_norm", "skewness_norm", "velc_norm"]
__all__ += ["mesh_diff", "mesh_to_ptcl"]
__all__ += ["central_moment", "moment2", "moment3", "moment4", "moment_components"]
__all__ += ["MOMENT_COMPONENTS", "Grid"]
