"""lenstorch — differentiable gravitational lensing in PyTorch, built for SBI."""

from .imaging import add_noise, gaussian_psf, pixel_grid, ray_trace_sie_shear, render
from .profiles import external_shear_deflection, sie_deflection, sis_deflection
from .simulator import PARAM_NAMES, LensSimulator, prior_bounds, sample_prior
from .sources import sersic

__version__ = "0.1.0"

__all__ = [
    "sie_deflection",
    "sis_deflection",
    "external_shear_deflection",
    "sersic",
    "pixel_grid",
    "ray_trace_sie_shear",
    "gaussian_psf",
    "render",
    "add_noise",
    "LensSimulator",
    "sample_prior",
    "prior_bounds",
    "PARAM_NAMES",
    "__version__",
]
