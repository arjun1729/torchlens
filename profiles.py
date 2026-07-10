"""Differentiable lens mass profiles.

All deflection functions are pure PyTorch, fully batched and autograd-friendly.

Conventions
-----------
* Angles in arcseconds on the image plane.
* Batched parameters: every lens parameter tensor has shape ``(B,)``.
* Coordinate grids ``x, y`` have shape ``(H, W)`` (shared across the batch)
  or ``(B, H, W)``.
* Returned deflections have shape ``(B, H, W)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["sie_deflection", "external_shear_deflection", "sis_deflection"]

_EPS = 1e-8
_Q_MAX = 0.9999  # SIE formulas are singular at q=1; clamp just below.


def _rotate(x: Tensor, y: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate coordinates by -phi (into the lens major-axis frame)."""
    c, s = torch.cos(phi), torch.sin(phi)
    c = c[..., None, None]
    s = s[..., None, None]
    return x * c + y * s, -x * s + y * c


def _rotate_back(ax: Tensor, ay: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
    c, s = torch.cos(phi), torch.sin(phi)
    c = c[..., None, None]
    s = s[..., None, None]
    return ax * c - ay * s, ax * s + ay * c


def sie_deflection(
    x: Tensor,
    y: Tensor,
    theta_e: Tensor,
    q: Tensor,
    phi: Tensor,
    x0: Tensor,
    y0: Tensor,
) -> tuple[Tensor, Tensor]:
    """Deflection field of a Singular Isothermal Ellipsoid (Kormann et al. 1994).

    Parameters
    ----------
    x, y : (H, W) or (B, H, W)
        Image-plane coordinates [arcsec].
    theta_e : (B,)
        Einstein radius [arcsec].
    q : (B,)
        Projected axis ratio, 0 < q <= 1.
    phi : (B,)
        Major-axis position angle [rad].
    x0, y0 : (B,)
        Lens centroid [arcsec].

    Returns
    -------
    (alpha_x, alpha_y) : each (B, H, W)
    """
    if x.dim() == 2:
        x = x[None]
        y = y[None]
    dx = x - x0[..., None, None]
    dy = y - y0[..., None, None]
    xr, yr = _rotate(dx, dy, phi)

    q = q.clamp(max=_Q_MAX)
    qp = torch.sqrt(1.0 - q * q)[..., None, None]
    te = theta_e[..., None, None]
    qb = q[..., None, None]

    psi = torch.sqrt(qb * qb * xr * xr + yr * yr + _EPS)
    pref = te * torch.sqrt(qb) / qp  # sqrt(q) normalization: theta_E = intermediate-axis
    ax = pref * torch.atan(qp * xr / (psi + _EPS))
    ay = pref * torch.atanh((qp * yr / (psi + _EPS)).clamp(-1 + 1e-7, 1 - 1e-7))

    return _rotate_back(ax, ay, phi)


def sis_deflection(
    x: Tensor, y: Tensor, theta_e: Tensor, x0: Tensor, y0: Tensor
) -> tuple[Tensor, Tensor]:
    """Singular Isothermal Sphere deflection: alpha = theta_E * r_hat."""
    if x.dim() == 2:
        x = x[None]
        y = y[None]
    dx = x - x0[..., None, None]
    dy = y - y0[..., None, None]
    r = torch.sqrt(dx * dx + dy * dy + _EPS)
    te = theta_e[..., None, None]
    return te * dx / r, te * dy / r


def external_shear_deflection(
    x: Tensor, y: Tensor, gamma1: Tensor, gamma2: Tensor
) -> tuple[Tensor, Tensor]:
    """External shear deflection about the origin."""
    if x.dim() == 2:
        x = x[None]
        y = y[None]
    g1 = gamma1[..., None, None]
    g2 = gamma2[..., None, None]
    return g1 * x + g2 * y, g2 * x - g1 * y
