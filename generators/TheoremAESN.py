from __future__ import annotations
from typing import Callable, Optional, Union, Tuple

import torch
import torch.nn as nn

from .ESN import _get_activation, rescale_spectral_norm
from .noise import Noise

Tensor = torch.Tensor

# A "noise spec" is anything we can turn into samples of a given shape:
#   - None                                  -> per-channel default Noise (uniform / normal)
#   - a callable (shape, device, dtype) -> Tensor
#   - any object exposing .sample(shape, device=..., dtype=..., generator=...)
#     (e.g. generators.noise.Noise, which supports normal / t / gamma / uniform)
NoiseSpec = Union[None, Callable[..., Tensor], object]


class TheoremAESN(nn.Module):
    """
    Autoregressive ESN generative model satisfying Theorem A
    (Grigoryeva-Ortega 2018, cold-start single-readout variant), with three
    experiment knobs that can all be driven from the notebook:

        (1) configurable reservoir-input noise   -> `noise`
        (2) k-Markov state augmentation          -> `feedback_lags`
        (3) scaled Gaussian output skip path     -> `skip_scale`, `v`
        (4) linear or shallow-NN readout         -> `readout`

    Recursion (feedback_lags = k):
        s_j    = [x_hat_{j-1}, ..., x_hat_{j-k}]            # explicit Markov state
        h_j    = sigma( A h_{j-1} + C [s_j, u_j]^T + zeta )
        r_j    = W h_j                       (readout="linear")
                 W phi(W_sigma h_j)          (readout="nonlinear", shallow NN)
        x_hat_j = r_j  +  skip_scale * v * xi_skip_j        # skip term (off if skip_scale=0)

    where u_j ~ `noise` (default Noise("uniform") = Unif[-1,1]) is the PIT channel
    and xi_skip_j ~ `skip_noise` (default Noise("normal") = N(0,1), unbounded -> tails).

    --- (1) noise -----------------------------------------------------------
    Any fixed continuous law is admissible: an affine/monotone reparametrization
    of the Unif[0,1] PIT variable is absorbed into C and zeta (Theorem A Rmk 5),
    so universality is preserved. Both channels go through generators.noise.Noise.
    Pass `noise` as a Noise, a callable (shape, device, dtype)->Tensor, or leave
    None for the symmetric Unif[-1,1] default. It is a plain attribute, so you can
    also swap it live from the notebook:
        esn.noise = Noise("normal"); esn.noise = Noise("t", {"df": 4})
        esn.noise = Noise("uniform", {"scale": 0.5})   # Unif[-0.5, 0.5]
    With zero-mean noise + zeta=0 + cold start, tanh (odd) gives E[h]=0 so the
    generator is exactly centered (good for symmetric targets; use zeta_scale>0
    or a skip/bias to model skew).

    --- (2) feedback_lags (k-Markov augmentation) ---------------------------
    A k-Markov target's conditional law depends on the window x_{j-k:j-1}. Rather
    than rely on the random reservoir's fading memory to reconstruct that window,
    feed it back explicitly: `feedback_lags=k` feeds [x_hat_{j-1},...,x_hat_{j-k}]
    into the reservoir input. For AR2 set feedback_lags=2 (the default 1 already
    feeds x_hat_{j-1}, so you only add one more lag). The fed-back value INCLUDES
    the skip innovation, so it propagates like an AR innovation.

    --- (3) skip_scale / v --------------------------------------------------
    `v` (shape (d,), trainable) is the output skip weight; the contribution is
    `skip_scale * v * xi_skip`. skip_scale is a fixed gain:
        skip_scale = 0  -> skip OFF; the term is identically 0, so v.grad = 0 and
                           v does not learn (lets you isolate the no-skip model).
        skip_scale > 0  -> skip ON; train v (include esn.v in your optimizer).
    The unbounded Gaussian skip is the cheap fix for the bounded-tanh tail/kurtosis
    deficit.

    --- (4) readout ---------------------------------------------------------
    readout="linear"    -> x_hat = W h            (W: d x h)
    readout="nonlinear" -> x_hat = W phi(W_sigma h)   (shallow NN: h -> readout_hidden
                           -> phi=readout_activation -> d), trainable W and W_sigma.
    The nonlinear readout (cf. ESN_updated) adds capacity for the steep quantile
    tails a linear readout off a fixed reservoir cannot reach. Note: phi (e.g.
    gelu) is not odd, so the symmetric-noise centering guarantee no longer holds
    exactly under a nonlinear readout -- centering becomes something training (or
    a skew term) must provide.

    Stability: ||A||_2 * L_sigma < 1  (spectral NORM, not radius). target_spectral_norm < 1 for tanh.

    Trainable:  W, v (and W_sigma if readout="nonlinear")     Fixed: A, C, zeta.
    Cold start: h_0 = 0, lag buffer = x_init (default 0); washout_len absorbs the transient.

    Trainer note:
        Use a full-BPTT trainer (run forward inside the loss). To use these knobs:
          - optimize [esn.W, esn.v]  (v only matters when skip_scale>0);
          - for a deterministic objective, sample BOTH drives once and pass them:
                u  = esn.sample_noise(N, T, generator=g1)
                us = esn.sample_skip_noise(N, T, generator=g2)   # only if skip_scale>0
                esn(T=T, N=N, xi=u, xi_skip=us)
        The frozen-Xhist trainer (fit_ESN_MMD_LBFGS) is NOT valid here: it assumes
        Z = Xhist @ W.T and ignores v, the skip, and the lag feedback.
    """

    def __init__(
        self,
        h_dim: int,
        out_dim: int = 1,
        *,
        target_spectral_norm: float = 0.9,
        activation: Union[str, Callable[[Tensor], Tensor]] = "tanh",
        readout: str = "linear",                  # "linear" or "nonlinear" (shallow NN readout)
        readout_hidden: int = 64,                 # hidden width when readout="nonlinear"
        readout_activation: Union[str, Callable[[Tensor], Tensor]] = "gelu",
        sparsity: float = 0.1,
        C_scale: float = 0.1,
        zeta_scale: float = 0.0,        # 0 keeps the generator centered for symmetric targets
        W_init_std: float = 0.01,
        noise_dim: int = 1,
        noise: NoiseSpec = None,        # reservoir-input noise; None -> Noise("uniform") = Unif[-1,1]
        use_output_feedback: Optional[bool] = True,
        feedback_lags: Optional[int] = 1,         # k: # past outputs fed back (k-Markov state augmentation)
        skip_scale: Optional[float] = 0.0,        # gain on output skip innovation; 0 -> OFF (v does not learn)
        v_init_std: Optional[float] = 0.1,        # init scale of trainable skip weight v
        skip_noise: NoiseSpec = None,   # skip innovation; None -> N(0,1)
        washout_len: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        if not (0.0 < target_spectral_norm < 1.0):
            raise ValueError(
                "target_spectral_norm must be in (0,1) to guarantee ||A||_2 * L_sigma < 1 "
                "for squashing activations (L_sigma <= 1)."
            )
        if int(feedback_lags) < 1:
            raise ValueError("feedback_lags must be >= 1")

        self.h = h_dim
        self.d = out_dim
        self.noise_dim = noise_dim
        # Per-channel default noise laws (all routed through generators.noise.Noise):
        #   reservoir input -> Unif[-1,1] (symmetric PIT) ; output skip -> N(0,1) (tails)
        self.noise = noise if noise is not None else Noise("uniform")
        self.skip_noise = skip_noise if skip_noise is not None else Noise("normal")
        self.use_output_feedback = use_output_feedback
        self.feedback_lags = int(feedback_lags)
        self.skip_scale = float(skip_scale)
        self.washout_len = washout_len if washout_len is not None else h_dim
        self.target_spectral_norm = target_spectral_norm

        fb_dim = self.feedback_lags * out_dim if use_output_feedback else 0
        self.input_dim = fb_dim + noise_dim

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)

        # Sparse random reservoir, normalised to ||A||_2 = target_spectral_norm
        mask  = (torch.rand(h_dim, h_dim, generator=gen) < sparsity).float()
        A_raw = torch.randn(h_dim, h_dim, generator=gen) * mask
        A     = rescale_spectral_norm(A_raw, target_spectral_norm)
        self.register_buffer("A", A)

        C    = torch.randn(h_dim, self.input_dim, generator=gen) * C_scale
        zeta = torch.randn(h_dim, generator=gen) * zeta_scale
        self.register_buffer("C", C)
        self.register_buffer("zeta", zeta)

        # Readout: linear (W: d x h) or shallow NN (h -> readout_hidden -> d).
        self.nonlinear_readout = str(readout).lower() in ("nonlinear", "nn", "shallow", "mlp")
        if self.nonlinear_readout:
            self.readout_hidden = int(readout_hidden)
            self.W_sigma = nn.Parameter(torch.randn(self.readout_hidden, h_dim, generator=gen) * W_init_std)
            W0 = torch.randn(out_dim, self.readout_hidden, generator=gen) * W_init_std
            self.readout_activation = _get_activation(readout_activation)
            self.readout_activation_name = (
                readout_activation if isinstance(readout_activation, str)
                else getattr(readout_activation, "__name__", "custom")
            )
        else:
            W0 = torch.randn(out_dim, h_dim, generator=gen) * W_init_std
        self.W = nn.Parameter(W0)

        # Trainable output skip weight (per output dim). Always present; only
        # contributes when skip_scale != 0 (so v.grad = 0 when skip is OFF).
        v0 = torch.randn(out_dim, generator=gen) * v_init_std
        self.v = nn.Parameter(v0)

        self.activation = _get_activation(activation)
        self.activation_name = (
            activation if isinstance(activation, str)
            else getattr(activation, "__name__", "custom")
        )

    # ------------------------------------------------------------------ noise
    def _draw(self, spec: NoiseSpec, shape, *, generator: Optional[torch.Generator]) -> Tensor:
        device, dtype = self.A.device, self.A.dtype
        if callable(spec):                       # plain callable (shape, device, dtype) -> Tensor
            return spec(shape, device=device, dtype=dtype)
        # Noise-like: generator honored for normal/uniform, ignored for t/gamma
        return spec.sample(shape, device=device, dtype=dtype, generator=generator)

    def sample_noise(self, N: int, T: int, *, generator: Optional[torch.Generator] = None) -> Tensor:
        """Reservoir-input drive, shape (N, washout_len+T, noise_dim). Call this from the trainer
        so the optimizer's drive matches forward()'s. Default noise = Noise('uniform') = Unif[-1,1]."""
        total_T = self.washout_len + T
        return self._draw(self.noise, (N, total_T, self.noise_dim), generator=generator)

    def sample_skip_noise(self, N: int, T: int, *, generator: Optional[torch.Generator] = None) -> Tensor:
        """Output skip innovation, shape (N, washout_len+T, d). Default noise = Noise('normal') = N(0,1)."""
        total_T = self.washout_len + T
        return self._draw(self.skip_noise, (N, total_T, self.d), generator=generator)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        T: int,
        *,
        N: int = 1,
        x_init: Optional[Tensor] = None,   # (N,d) or (d,) or None -> zeros
        xi: Optional[Tensor] = None,       # reservoir drive: (N, total_T, noise_dim)
        xi_skip: Optional[Tensor] = None,  # skip innovation: (N, total_T, d) (used iff skip_scale!=0)
        eta: Optional[Tensor] = None,      # accepted for trainer-signature compatibility; ignored
        return_states: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Returns Z of shape (N, T, d). Discards the first washout_len steps."""
        device, dtype = self.A.device, self.A.dtype
        total_T = self.washout_len + T

        h = torch.zeros(N, self.h, device=device, dtype=dtype)

        # initial output (seeds the lag buffer)
        if x_init is None:
            x0 = torch.zeros(N, self.d, device=device, dtype=dtype)
        else:
            x_init = torch.as_tensor(x_init, device=device, dtype=dtype)
            if x_init.shape == (self.d,):
                x0 = x_init.unsqueeze(0).expand(N, -1).clone()
            elif x_init.shape == (N, self.d):
                x0 = x_init
            else:
                raise ValueError("x_init must be (d,) or (N,d)")

        # lag buffer: most-recent first, length feedback_lags
        lag_buf = [x0.clone() for _ in range(self.feedback_lags)] if self.use_output_feedback else None

        # reservoir noise
        if xi is None:
            u_all = self.sample_noise(N, T)
        else:
            u_all = torch.as_tensor(xi, device=device, dtype=dtype)
            if u_all.shape != (N, total_T, self.noise_dim):
                raise ValueError(f"xi must have shape ({N},{total_T},{self.noise_dim}), got {tuple(u_all.shape)}")

        # skip noise (only when the skip path is active)
        skip_on = self.skip_scale != 0.0
        if skip_on:
            if xi_skip is None:
                s_all = self.sample_skip_noise(N, T)
            else:
                s_all = torch.as_tensor(xi_skip, device=device, dtype=dtype)
                if s_all.shape != (N, total_T, self.d):
                    raise ValueError(f"xi_skip must have shape ({N},{total_T},{self.d}), got {tuple(s_all.shape)}")

        Z_full = torch.empty(N, total_T, self.d, device=device, dtype=dtype)
        X_full = torch.empty(N, total_T, self.h, device=device, dtype=dtype) if return_states else None

        A, C, zeta, W, act = self.A, self.C, self.zeta, self.W, self.activation
        nonlinear = self.nonlinear_readout
        if nonlinear:
            W_sigma, ro_act = self.W_sigma, self.readout_activation

        for t in range(total_T):
            u_t = u_all[:, t, :]
            if self.use_output_feedback:
                inp = torch.cat(lag_buf + [u_t], dim=-1)   # [x_{t-1},...,x_{t-k}, u_t]
            else:
                inp = u_t
            h = act(h @ A.T + inp @ C.T + zeta)
            if nonlinear:
                x_hat = ro_act(h @ W_sigma.T) @ W.T        # shallow NN readout
            else:
                x_hat = h @ W.T                            # linear readout
            if skip_on:
                x_hat = x_hat + self.skip_scale * self.v * s_all[:, t, :]
            Z_full[:, t, :] = x_hat
            if return_states:
                X_full[:, t, :] = h
            if self.use_output_feedback:
                lag_buf = [x_hat] + lag_buf[:-1]           # shift in newest, drop oldest

        Z = Z_full[:, self.washout_len:, :]
        X = X_full[:, self.washout_len:, :] if return_states else None
        return (Z, X) if return_states else Z
