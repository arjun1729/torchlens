"""Differentiable source light profiles."""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["sersic"]

_EPS = 1e-8


def sersic(
    bx: Tensor,
    by: Tensor,
    amp: Tensor,
    r_eff: Tensor,
    n: Tensor,
    q: Tensor,
    phi: Tensor,
    x0: Tensor,
    y0: Tensor,
) -> Tensor:
    """Elliptical Sersic surface brightness, evaluated at source-plane coords.

    Parameters
    ----------
    bx, by : (B, H, W)
        Source-plane coordinates [arcsec] (i.e. ray-traced beta).
    amp : (B,)
        Amplitude at the effective radius.
    r_eff : (B,)
        Effective (half-light) radius [arcsec].
    n : (B,)
        Sersic index.
    q, phi : (B,)
        Axis ratio and position angle [rad].
    x0, y0 : (B,)
        Source centroid [arcsec].

    Returns
    -------
    (B, H, W) surface brightness.
    """
    c, s = torch.cos(phi), torch.sin(phi)
    c = c[..., None, None]
    s = s[..., None, None]
    dx = bx - x0[..., None, None]
    dy = by - y0[..., None, None]
    xr = dx * c + dy * s
    yr = -dx * s + dy * c

    qb = q[..., None, None]
    r = torch.sqrt(qb * xr * xr + yr * yr / qb + _EPS)

    nb = n[..., None, None]
    # Ciotti & Bertin (1999) approximation for b_n.
    bn = 2.0 * nb - 1.0 / 3.0 + 4.0 / (405.0 * nb) + 46.0 / (25515.0 * nb * nb)
    re = r_eff[..., None, None].clamp(min=1e-4)
    return amp[..., None, None] * torch.exp(-bn * ((r / re) ** (1.0 / nb) - 1.0))
