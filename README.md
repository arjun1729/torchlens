# lenstorch

**Differentiable gravitational lensing in PyTorch, built for simulation-based inference.**

![Simulated lenses](assets/gallery.png)

`lenstorch` is a minimal, fully-batched, autograd-compatible strong-lensing forward model.
It renders SIE + external-shear lenses with Sérsic sources — with PSF convolution and
realistic noise — as a single differentiable PyTorch pipeline, and ships a ready-made
bridge to neural posterior estimation via the [`sbi`](https://sbi-dev.github.io/sbi/) package.

**Why differentiable?**
- Fit lens models with **gradient descent / Adam / L-BFGS** instead of derivative-free samplers.
- Drop the forward model **inside a neural network** and train end-to-end.
- Simulate **millions of lenses on GPU** for amortized inference (train once, infer any lens in milliseconds).

## Install

```bash
git clone https://github.com/YOURUSERNAME/lenstorch
cd lenstorch
pip install -e ".[sbi,dev]"
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0. GPU optional but recommended for large simulation budgets.

## Quickstart

Render a batch of lenses in four lines:

```python
import torch
from lenstorch import LensSimulator, sample_prior

sim = LensSimulator(npix=64, pixel_scale=0.1, device="cuda" if torch.cuda.is_available() else "cpu")
theta = sample_prior(256)          # (256, 11) parameters from the prior
images = sim(theta)                # (256, 4096) noisy lensed images
```

Fit a lens with gradients (the differentiable payoff):

```python
theta = theta_init.clone().requires_grad_(True)
opt = torch.optim.Adam([theta], lr=2e-2)
for _ in range(400):
    opt.zero_grad()
    loss = ((sim.render_clean(theta) - observed) ** 2).sum()
    loss.backward()
    opt.step()
```

Full runnable versions: [`examples/fit_with_gradients.py`](examples/fit_with_gradients.py) and
[`examples/train_npe.py`](examples/train_npe.py) (amortized neural posterior with `sbi`).

## Parameterization

11 parameters per lens (see `lenstorch.PARAM_NAMES`): Einstein radius, lens axis ratio and
position angle, external shear (γ₁, γ₂), source position, effective radius, Sérsic index,
source axis ratio and position angle. Uniform priors defined in `lenstorch/simulator.py`.

## Generate a dataset

```bash
python scripts/generate_dataset.py --n 50000 --npix 64 --out data/lenses_50k.pt
```

A small pre-generated demo set (2,000 lenses, 64×64) ships in `data/lenses_2k_demo.pt`.
For a *useful* NPE posterior you want ≥ 50k simulations; the demo set is for pipeline testing only.

## Train an amortized posterior

```bash
python examples/train_npe.py --n-sims 50000
```

After training, posterior inference on any new lens image takes milliseconds — no MCMC.

## Tests

```bash
pytest tests/
```

Physics sanity checks included: SIE → SIS limit, |α| = θ_E for SIS, deflection antisymmetry,
end-to-end gradient flow, and gradient-based parameter recovery.

## Relation to existing packages

`lenstorch` does **not** replace [`lenstronomy`](https://github.com/lenstronomy/lenstronomy),
[`PyAutoLens`](https://github.com/Jammy2211/PyAutoLens), or
[`caustics`](https://github.com/Ciela-Institute/caustics) — those are far more complete.
The niche here is *minimalism*: a single readable forward model (~300 lines) that you can
fully understand, modify, and embed in ML pipelines without a framework around it.
Think "the `micrograd` of strong lensing." If you need multi-plane lensing, pixelated
sources, or instrument-grade noise models, use the packages above.

## Roadmap

- [ ] NFW + elliptical power-law profiles
- [ ] Subhalo perturbers (dark-matter substructure inference)
- [ ] Pixelated / neural (diffusion-prior) source models
- [ ] Real-survey noise + PSF loading (HST / Euclid / LSST cutouts)
- [ ] JOSS paper

Contributions welcome — open an issue first for anything non-trivial.

## Citation

If you use `lenstorch` in your research, please cite it (see [`CITATION.cff`](CITATION.cff)).
A Zenodo DOI will be minted at the first tagged release.

## License

MIT
