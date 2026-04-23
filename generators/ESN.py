from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Union, Literal, Dict

import torch
import torch.nn as nn

Tensor = torch.Tensor

def _get_activation(name_or_fn: Union[str, Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
    # Returns a PyTorch activation function based on a string or callable input
    if callable(name_or_fn):
        return name_or_fn
    name = str(name_or_fn).lower()
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "gelu":
        return torch.nn.functional.gelu
    raise ValueError(f"Unknown activation: {name_or_fn}")

def _ma_filter(e: Tensor, theta: Tensor) -> Tensor:
    # Applies a moving average (MA) filter to input noise using coefficients theta
    """
    e: (N,T,m) iid innovations
    theta: (q,) MA coefficients
    y_t = e_t + sum_{j=1}^q theta[j-1] e_{t-j}
    """
    theta = theta.reshape(-1)
    q = int(theta.numel())
    if q == 0:
        return e

    N, T, m = e.shape
    y = e.clone()
    for j in range(1, q + 1):
        y[:, j:, :] = y[:, j:, :] + theta[j - 1] * e[:, : T - j, :]
    return y


def rescale_spectral_radius(A: Tensor, target_rho: float) -> Tensor:
    # Rescales matrix A so its spectral radius matches target_rho
    """
    Rescale A so that spectral radius max |lambda_i| equals target_rho.
    One-time constraint at init, A remains fixed afterwards.
    """
    if not (0.0 < target_rho < 1.0):
        raise ValueError("target_rho must be in (0,1)")
    with torch.no_grad():
        eigvals = torch.linalg.eigvals(A)
        rho = eigvals.abs().max().real
        if rho <= 0:
            return A
        return A * (target_rho / rho)


class ESNGenerator(nn.Module):
    """
    ESN generator (A, C fixed; only W trainable):

      X_t = sigma( A X_{t-1} + C xi_t )
      Z_t = W X_t + eta_t + t_tilt

    xi_t ~ IID Normal(0, xi_scale^2 I_m)
    eta_t ~ IID Normal(0, eta_scale^2 I_d)

    Shapes:
      A: (h, h), spectral radius constrained to target_rho < 1 at init
      C: (h, m)
      W: (d, h)  (trainable)
      X_t: (N, h)
      Z_t: (N, d)

    forward returns Z of shape (N, T, d).
    """

    def __init__(
        # Initializes the ESN generator, sets up matrices, parameters, and feedback
        self,
        A: Tensor,
        C: Tensor,
        out_dim: int,
        *,
        activation: Union[str, Callable[[Tensor], Tensor]] = "tanh",
        xi_scale: float = 1.0,
        eta_scale: float = 1.0,
        target_rho: float = 0.9,
        xi_ma_theta: Optional[Tensor] = None,
        t_tilt: Optional[Tensor] = None,
        W_init_std: float = 0.1,
        quad_feedback: bool = False,  # Enable quadratic feedback in the reservoir
        quad_gain: float = 0.1,       # Scaling factor for quadratic feedback matrix
        train_quad: bool = False,     # If True, make quadratic feedback trainable
        W_init: Optional[Tensor] = None,
        washout_len: Optional[int] = None,  # Optional washout length; if None, defaults to h
    ):
        super().__init__()
        A = torch.as_tensor(A)
        C = torch.as_tensor(C)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square (h,h)")
        if C.ndim != 2 or C.shape[0] != A.shape[0]:
            raise ValueError("C must have shape (h,m) with same h as A")

        self.h = int(A.shape[0])
        self.m = int(C.shape[1])
        self.d = int(out_dim)

        self.washout_len = washout_len if not None else self.h # Use reservoir size as washout length if not specified

        # Quadratic feedback in the reservoir
        self.quad_feedback = bool(quad_feedback)
        if self.quad_feedback:
            self.quad_gain = float(quad_gain)
            G0 = float(quad_gain) * torch.randn(self.h, self.h, device=A.device, dtype=A.dtype) / (self.h ** 0.5)
            if train_quad:
                self.G_quad = nn.Parameter(G0)
            else:
                self.register_buffer("G_quad", G0)
        else:
            self.register_buffer("G_quad", None, persistent=False)

        # Rescale A to have spectral radius target_rho < 1 for echo state property, and register buffers/parameters
        A = rescale_spectral_radius(A, float(target_rho))
        self.register_buffer("A", A)
        self.register_buffer("C", C)

        if W_init is None:
            W0 = torch.randn(self.d, self.h, device=A.device, dtype=A.dtype) * float(W_init_std)
        else:
            W0 = torch.as_tensor(W_init, device=A.device, dtype=A.dtype)
        self.W = nn.Parameter(W0)

        self.activation = _get_activation(activation)
        self.activation_name = activation if isinstance(activation, str) else getattr(activation, "__name__", "custom")
        self.xi_scale = float(xi_scale)
        self.eta_scale = float(eta_scale)

        if t_tilt is None:
            self.register_buffer("t_tilt", None, persistent=False)
        else:
            self.register_buffer("t_tilt", torch.as_tensor(t_tilt, device=A.device, dtype=A.dtype), persistent=False)

        if xi_ma_theta is None:
            self.register_buffer("xi_ma_theta", None, persistent=False)
        else:
            self.register_buffer("xi_ma_theta", torch.as_tensor(xi_ma_theta, device=A.device, dtype=A.dtype), persistent=False)
        if self.xi_ma_theta is not None and self.xi_ma_theta.ndim != 1:
            raise ValueError("xi_ma_theta must be 1D (q,).")

    @torch.no_grad()
    def sample_noise(self, N: int, T: int) -> tuple[Tensor, Tensor]:
        # Generates input and output noise samples, applies MA filtering if needed
        device, dtype = self.A.device, self.A.dtype
        xi_ma_theta = self.xi_ma_theta

        # iid innovations
        xi_e = torch.randn(N, T, self.m, device=device, dtype=dtype) * self.xi_scale
        eta = torch.randn(N, T, self.d, device=device, dtype=dtype) * self.eta_scale

        # colored input noise
        if xi_ma_theta is not None:
            th = torch.as_tensor(xi_ma_theta, device=device, dtype=dtype)
            xi = _ma_filter(xi_e, th)
        else:
            xi = xi_e

        return xi, eta
    
    def forward(
        # Runs the ESN for T time steps, generating output sequences (and optionally hidden states)
        self,
        T: int,
        *,
        N: int = 1,
        x0: Optional[Tensor] = None,     # (N,h) or (h,) or None
        xi: Optional[Tensor] = None,     # (N,T,m) or None
        eta: Optional[Tensor] = None,    # (N,T,d) or None
        return_states: bool = False,
    ):
        device, dtype = self.A.device, self.A.dtype

        # --- Washout setup ---
        total_T = self.washout_len + T

        if x0 is None:
            x = torch.zeros(N, self.h, device=device, dtype=dtype)
        else:
            x0 = torch.as_tensor(x0, device=device, dtype=dtype)
            if x0.shape == (self.h,):
                x = x0.view(1, self.h).repeat(N, 1)
            elif x0.shape == (N, self.h):
                x = x0
            else:
                raise ValueError("x0 must be (h,) or (N,h)")

        # Generate or validate noise for total_T steps
        if xi is None or eta is None:
            xi_s, eta_s = self.sample_noise(N, total_T)
            if xi is None:
                xi = xi_s
            if eta is None:
                eta = eta_s
        else:
            xi = torch.as_tensor(xi, device=device, dtype=dtype)
            eta = torch.as_tensor(eta, device=device, dtype=dtype)
            if xi.shape[1] != total_T or eta.shape[1] != total_T:
                raise ValueError(f"xi and eta must have shape (N, {total_T}, m/d)")

        if xi.shape != (N, total_T, self.m):
            raise ValueError(f"xi must have shape (N,total_T,m)=({N},{total_T},{self.m})")
        if eta.shape != (N, total_T, self.d):
            raise ValueError(f"eta must have shape (N,total_T,d)=({N},{total_T},{self.d})")

        # Handle t_tilt for total_T
        if self.t_tilt is None:
            tilt = None
        else:
            tt = self.t_tilt
            if tt.ndim == 1 and tt.shape == (self.d,):
                tilt = tt.view(1, 1, self.d)
            elif tt.ndim == 2 and tt.shape == (total_T, self.d):
                tilt = tt.view(1, total_T, self.d)
            elif tt.ndim == 3 and tt.shape[1:] == (total_T, self.d):
                tilt = tt
            else:
                raise ValueError("t_tilt must be (d,) or (T,d) or (1,T,d) or broadcastable to (N,T,d)")

        Z_full = torch.empty(N, total_T, self.d, device=device, dtype=dtype)
        Xhist_full = torch.empty(N, total_T, self.h, device=device, dtype=dtype) if return_states else None

        A, C, W = self.A, self.C, self.W
        act = self.activation

        for t in range(total_T):
            pre = x @ A.T + xi[:, t, :] @ C.T
            if self.quad_feedback:
                pre = pre + (x * x) @ self.G_quad.T
            x = act(pre)
            z = x @ W.T + eta[:, t, :]
            if tilt is not None:
                z = z + tilt[:, t, :]
            Z_full[:, t, :] = z
            if return_states:
                Xhist_full[:, t, :] = x

        # Only return the last T points after washout
        Z = Z_full[:, self.washout_len:, :]
        Xhist = Xhist_full[:, self.washout_len:, :] if return_states else None

        return (Z, Xhist) if return_states else Z


# For convenience, a wrapper that treats the ESN as a generator of targets only (no input noise, no feedback, no state return).
class ESNAsTarget(nn.Module):
    def __init__(self, esn: nn.Module, T_default: int):
        # Wraps an ESN to provide a simple target generator interface
        super().__init__()
        self.esn = esn
        self.T_default = int(T_default)

    @torch.no_grad()
    def generate(self, *, N: int, T: int | None = None, noise=None):
        # Calls the ESN to generate output sequences for a given number of samples and time steps
        T_use = self.T_default if T is None else int(T)
        return self.esn(T=T_use, N=N)