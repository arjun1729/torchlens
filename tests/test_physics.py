import torch

from lenstorch import (
    LensSimulator,
    pixel_grid,
    render,
    sample_prior,
    sie_deflection,
    sis_deflection,
)


def _grid():
    return pixel_grid(32, 0.1)


def test_sie_reduces_to_sis_at_q1():
    """As q -> 1, SIE deflection must approach the SIS solution."""
    x, y = _grid()
    te = torch.tensor([1.2])
    zero = torch.zeros(1)
    ax_sie, ay_sie = sie_deflection(x, y, te, torch.tensor([0.9999]), zero, zero, zero)
    ax_sis, ay_sis = sis_deflection(x, y, te, zero, zero)
    assert torch.allclose(ax_sie, ax_sis, atol=5e-3)
    assert torch.allclose(ay_sie, ay_sis, atol=5e-3)


def test_sis_deflection_magnitude_is_theta_e():
    """|alpha| = theta_E everywhere for an SIS."""
    x, y = _grid()
    te = torch.tensor([1.5])
    zero = torch.zeros(1)
    ax, ay = sis_deflection(x, y, te, zero, zero)
    mag = torch.sqrt(ax**2 + ay**2)
    # Exclude the central pixel region where the profile is singular.
    r = torch.sqrt(x**2 + y**2)
    mask = r > 0.3
    assert torch.allclose(mag[0][mask], torch.full_like(mag[0][mask], 1.5), atol=1e-2)


def test_sie_deflection_antisymmetry():
    """alpha(-theta) = -alpha(theta) for a centered lens."""
    x, y = _grid()
    te, q = torch.tensor([1.0]), torch.tensor([0.7])
    zero = torch.zeros(1)
    ax, ay = sie_deflection(x, y, te, q, zero, zero, zero)
    ax_f, ay_f = sie_deflection(-x, -y, te, q, zero, zero, zero)
    assert torch.allclose(ax, -ax_f, atol=1e-6)
    assert torch.allclose(ay, -ay_f, atol=1e-6)


def test_render_shapes_and_batching():
    sim = LensSimulator(npix=48)
    theta = sample_prior(4)
    imgs = sim.render_clean(theta)
    assert imgs.shape == (4, 48, 48)
    flat = sim(theta)
    assert flat.shape == (4, 48 * 48)
    assert torch.isfinite(flat).all()


def test_gradients_flow_to_parameters():
    """End-to-end differentiability: dL/dtheta_E exists and is nonzero."""
    x, y = pixel_grid(32, 0.1)
    theta_e = torch.tensor([1.0], requires_grad=True)
    lens = {
        "theta_e": theta_e,
        "q": torch.tensor([0.8]),
        "phi": torch.zeros(1),
        "x0": torch.zeros(1),
        "y0": torch.zeros(1),
        "gamma1": torch.zeros(1),
        "gamma2": torch.zeros(1),
    }
    src = {
        "amp": torch.ones(1),
        "r_eff": torch.tensor([0.2]),
        "n": torch.tensor([1.5]),
        "q": torch.tensor([0.9]),
        "phi": torch.zeros(1),
        "x0": torch.tensor([0.1]),
        "y0": torch.zeros(1),
    }
    img = render(lens, src, npix=32)
    img.sum().backward()
    assert theta_e.grad is not None
    assert torch.isfinite(theta_e.grad).all()
    assert theta_e.grad.abs().item() > 0


def test_gradient_based_recovery_of_theta_e():
    """A few Adam steps on the clean-image chi^2 should move theta_E toward truth."""
    sim = LensSimulator(npix=48, sigma_bkg=0.0)
    true = sample_prior(1)
    true[0, 0] = 1.4
    target = sim.render_clean(true).detach()

    guess = true.clone()
    guess[0, 0] = 1.0
    guess.requires_grad_(True)
    opt = torch.optim.Adam([guess], lr=5e-2)
    start_err = abs(guess[0, 0].item() - 1.4)
    for _ in range(30):
        opt.zero_grad()
        loss = ((sim.render_clean(guess) - target) ** 2).mean()
        loss.backward()
        # Only optimize theta_E in this test.
        guess.grad[0, 1:] = 0.0
        opt.step()
    assert abs(guess[0, 0].item() - 1.4) < start_err
