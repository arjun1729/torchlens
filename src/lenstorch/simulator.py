"""Batched lens simulator with priors — the bridge to simulation-based inference.

The parameter vector (11-D) is ordered as:

======  ===========  ==========================================
index   name         meaning
======  ===========  ==========================================
0       theta_e      Einstein radius [arcsec]
1       q_lens       lens axis ratio
2       phi_lens     lens position angle [rad]
3       gamma1       external shear component 1
4       gamma2       external shear component 2
5       src_x0       source x offset [arcsec]
6       src_y0       source y offset [arcsec]
7       src_r_eff    source effective radius [arcsec]
8       src_n        source Sersic index
9       src_q        source axis ratio
10      src_phi      source position angle [rad]
======  ===========  ==========================================

Lens centroid is fixed at the origin (standard for postage-stamp inference).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .imaging import add_noise, gaussian_psf, render

__all__ = ["PARAM_NAMES", "prior_bounds", "sample_prior", "LensSimulator"]

PARAM_NAMES = [
    "theta_e",
    "q_lens",
    "phi_lens",
    "gamma1",
    "gamma2",
    "src_x0",
    "src_y0",
    "src_r_eff",
    "src_n",
    "src_q",
    "src_phi",
]

_LOW = [0.5, 0.4, 0.0, -0.1, -0.1, -0.4, -0.4, 0.05, 0.8, 0.4, 0.0]
_HIGH = [2.0, 1.0, 3.14159265, 0.1, 0.1, 0.4, 0.4, 0.5, 4.0, 1.0, 3.14159265]


def prior_bounds(device: torch.device | str = "cpu") -> tuple[Tensor, Tensor]:
    """Uniform prior bounds (low, high), each shape (11,)."""
    return (
        torch.tensor(_LOW, dtype=torch.float32, device=device),
        torch.tensor(_HIGH, dtype=torch.float32, device=device),
    )


def sample_prior(n: int, device: torch.device | str = "cpu") -> Tensor:
    low, high = prior_bounds(device)
    return low + (high - low) * torch.rand(n, len(_LOW), device=device)


class LensSimulator:
    """Maps parameter batches (B, 11) -> noisy lensed images (B, npix, npix)."""

    def __init__(
        self,
        npix: int = 64,
        pixel_scale: float = 0.1,
        psf_fwhm_pix: float = 2.0,
        sigma_bkg: float = 0.02,
        exp_time: float = 1000.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.npix = npix
        self.pixel_scale = pixel_scale
        self.sigma_bkg = sigma_bkg
        self.exp_time = exp_time
        self.device = torch.device(device)
        self.psf = gaussian_psf(psf_fwhm_pix, device=self.device)

    def _split(self, theta: Tensor) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        t = theta.to(self.device).float()
        zeros = torch.zeros(t.shape[0], device=self.device)
        lens = {
            "theta_e": t[:, 0],
            "q": t[:, 1],
            "phi": t[:, 2],
            "x0": zeros,
            "y0": zeros,
            "gamma1": t[:, 3],
            "gamma2": t[:, 4],
        }
        src = {
            "x0": t[:, 5],
            "y0": t[:, 6],
            "r_eff": t[:, 7],
            "n": t[:, 8],
            "q": t[:, 9],
            "phi": t[:, 10],
            "amp": torch.ones_like(t[:, 0]),
        }
        return lens, src

    def render_clean(self, theta: Tensor) -> Tensor:
        """Noise-free images (differentiable w.r.t. theta)."""
        lens, src = self._split(torch.atleast_2d(theta))
        return render(
            lens, src, npix=self.npix, pixel_scale=self.pixel_scale, psf=self.psf, device=self.device
        )

    def __call__(self, theta: Tensor) -> Tensor:
        """Noisy images, flattened to (B, npix*npix) — the sbi-compatible signature."""
        img = self.render_clean(theta)
        img = add_noise(img, sigma_bkg=self.sigma_bkg, exp_time=self.exp_time)
        return img.reshape(img.shape[0], -1)
