# ESN MMD Generator

Echo State Network (ESN) **generative models** trained by Maximum Mean Discrepancy (MMD)
to match the *distribution* of a target stochastic process from sample paths.

The ESN has a fixed random reservoir and a trainable readout; training minimizes the MMD
(static RBF or signature kernels) between generated paths and target paths. The design is
built around **Theorem A** — ESN universality for compactly-supported `k`-Markov processes
(Grigoryeva–Ortega, *Echo state networks are universal*, 2018) — and the repo is the
empirical study of how closely a random-reservoir + MMD-trained ESN realizes that guarantee.

## Layout

```
generators/
  ESN.py                 ESNGenerator (fixed A,C; trainable readout W) + activation/rescale helpers
  TheoremAESN.py         Theorem-A model: cold-start, output feedback, k-lag state augmentation,
                         configurable noise, optional Gaussian skip, linear or shallow-NN readout
  synthetic_generators.py  Target processes: ARMA, GARCH11, MarkovTiltedUniform
  noise.py               Noise: normal / t / gamma / uniform / truncated_normal (GPU-safe)
loss/
  loss.py                MMD loss (mmd_loss / compute_mmd_loss), optional lead-lag transform
models/
  trainer_ESN_LBFGS.py   Frozen-state LBFGS MMD trainer (readout-only; exact when no feedback)
sigkernel_/              Discretized / truncated signature kernels
utils/                   Median-heuristic bandwidth, kernel tuning, ACF tests, data helpers
train_ESN_vs_*.ipynb     Experiment notebooks (AR2, ARMA11, GARCH11, MA2, Theorem-A target tests)
```

## Key pieces

- **`TheoremAESN`** — the autoregressive generator. Recursion (k = `feedback_lags`):
  `h_t = σ(A h_{t-1} + C[x̂_{t-1..t-k}, u_t] + ζ)`, `x̂_t = readout(h_t) + skip_scale·v·ξ_t`.
  Knobs (all notebook-driven): `noise` (PIT law), `feedback_lags`, `skip_scale`,
  `readout="linear"|"nonlinear"`. For symmetric targets keep `zeta_scale=0`, symmetric
  noise, and an **odd** `readout_activation` (`tanh`/`asinh`/`softsign`).
- **Targets** — `ARMA`, `GARCH11`, and `MarkovTiltedUniform`: a compact-support, density-
  bounded-below `k`-Markov process that *satisfies Theorem A's hypotheses* (the honest test
  target). All share `generate(N, T=...)`.
- **Training** — minimize MMD between target and ESN path samples. `models/` holds the
  readout-only LBFGS trainer; the notebooks use a full-BPTT trainer (differentiates through
  the recurrence, required when output feedback / nonlinear readout is on).

## Quick start

```python
import torch
from generators.TheoremAESN import TheoremAESN
from generators.synthetic_generators import ARMA
from generators.noise import Noise

torch.set_default_dtype(torch.float64)

target = ARMA(T=100, p=2, q=0, phi=[0.7, -0.2], burnin=50,
              noise=Noise("truncated_normal", {"c": 2.5}))   # compact-support AR2
Z_target = target.generate(N=100, T=100)

esn = TheoremAESN(h_dim=500, out_dim=1, target_spectral_norm=0.9, activation="tanh",
                  C_scale=0.7, sparsity=1.0, noise=Noise("uniform"), seed=0)
# ... fit the readout by minimizing MMD(Z_target, esn(T=100, N=...)), then evaluate
```

See the `train_ESN_vs_*.ipynb` notebooks for full training + diagnostics (ACF / two-sample
ACF tests, marginal & tail-reach tables).

## Requirements

Python 3.11, `torch`, `numpy`, `matplotlib`, `pandas`, `sigkernel` (PDE signature kernel),
`tensorboard`. A CUDA GPU is recommended for the PDE signature-kernel experiments.

## License

See `LICENSE`.
