"""Ray tracing and image rendering."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .profiles import external_shear_deflection, sie_deflection
from .sources import sersic

__all__ = ["pixel_grid", "ray_trace_sie_shear", "gaussian_psf", "render", "add_noise"]


def pixel_grid(
    npix: int, pixel_scale: float, device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Centered (H, W) coordinate grid in arcsec."""
    half = (npix - 1) / 2.0
    ax = (torch.arange(npix, device=device, dtype=torch.float32) - half) * pixel_scale
    y, x = torch.meshgrid(ax, ax, indexing="ij")
    return x, y


def ray_trace_sie_shear(x: Tensor, y: Tensor, params: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Lens equation beta = theta - alpha for SIE + external shear.

    ``params`` keys: theta_e, q, phi, x0, y0, gamma1, gamma2 — each shape (B,).
    """
    ax1, ay1 = sie_deflection(
        x, y, params["theta_e"], params["q"], params["phi"], params["x0"], params["y0"]
    )
    ax2, ay2 = external_shear_deflection(x, y, params["gamma1"], params["gamma2"])
    if x.dim() == 2:
        x = x[None]
        y = y[None]
    return x - ax1 - ax2, y - ay1 - ay2


def gaussian_psf(fwhm_pix: float, size: int = 11, device: torch.device | str = "cpu") -> Tensor:
    """Normalized Gaussian PSF kernel, shape (size, size)."""
    sigma = fwhm_pix / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    ax = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return k / k.sum()


def render(
    lens_params: dict[str, Tensor],
    src_params: dict[str, Tensor],
    npix: int = 64,
    pixel_scale: float = 0.1,
    psf: Tensor | None = None,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Render batched lensed images, shape (B, npix, npix). Fully differentiable."""
    x, y = pixel_grid(npix, pixel_scale, device=device)
    bx, by = ray_trace_sie_shear(x, y, lens_params)
    img = sersic(
        bx,
        by,
        src_params["amp"],
        src_params["r_eff"],
        src_params["n"],
        src_params["q"],
        src_params["phi"],
        src_params["x0"],
        src_params["y0"],
    )
    if psf is not None:
        pad = psf.shape[-1] // 2
        img = F.conv2d(img[:, None], psf[None, None], padding=pad)[:, 0]
    return img


def add_noise(img: Tensor, sigma_bkg: float = 0.05, exp_time: float = 1000.0) -> Tensor:
    """Gaussian background + Poisson-like shot noise (Gaussian approximation).

    Gaussian approximation keeps the operation differentiable-friendly and fast.
    """
    shot_var = img.clamp(min=0.0) / exp_time
    noise = torch.randn_like(img) * torch.sqrt(shot_var + sigma_bkg**2)
    return img + noise
