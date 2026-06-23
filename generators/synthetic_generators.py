from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from .noise import Noise

Tensor = torch.Tensor

# -------------------------
# base
# -------------------------

class Proc(nn.Module):
    def __init__(self, T: int, d: int = 1):
        super().__init__()
        self.T, self.d = int(T), int(d)
        self.register_buffer("_dummy", torch.empty(()), persistent=False)

    @property
    def device(self): return self._dummy.device
    @property
    def dtype(self): return self._dummy.dtype

    @torch.no_grad()
    def generate(
        self,
        N: int = 1,
        *,
        T: Optional[int] = None,
        noise: Optional[Noise] = None,
        eps: Optional[Tensor] = None
    ) -> Tensor:
        """
        Returns (N, T, d). If T is None, uses self.T.
        If eps is provided, T is inferred from eps.shape[1].
        """
        if eps is None:
            T_eff = self.T if T is None else int(T)
            noise = noise or Noise("normal")
            eps = noise.sample((N, T_eff, self.d), device=self.device, dtype=self.dtype)
        else:
            eps = eps.to(device=self.device, dtype=self.dtype)
            if eps.ndim != 3 or eps.shape[0] != N or eps.shape[2] != self.d:
                raise ValueError(f"eps must have shape (N,T,d)=({N},T,{self.d}), got {tuple(eps.shape)}")

        return self._gen(N, eps)


# -------------------------
# ARMA (AR if q=0, MA if p=0)
# -------------------------

class ARMA(Proc):
    # x_t = sum_i phi_i x_{t-i} + eps_t + sum_j theta_j eps_{t-j}
    def __init__(self, T: int, p: int, q: int, phi=None, theta=None, d: int = 1, burnin: Optional[int] = None, noise: Optional[Noise] = None):
        super().__init__(T, d)
        self.p, self.q = int(p), int(q)
        self.burnin = int(burnin) if burnin is not None else None
        self.noise = noise

        if self.p:
            phi = torch.zeros(self.p) if phi is None else torch.as_tensor(phi)
            if phi.numel() != self.p: raise ValueError("phi size mismatch")
            self.register_buffer("phi", phi.to(self.dtype))
        else:
            self.register_buffer("phi", torch.empty(0, dtype=self.dtype))

        if self.q:
            theta = torch.zeros(self.q) if theta is None else torch.as_tensor(theta)
            if theta.numel() != self.q: raise ValueError("theta size mismatch")
            self.register_buffer("theta", theta.to(self.dtype))
        else:
            self.register_buffer("theta", torch.empty(0, dtype=self.dtype))

    def _gen(self, N: int, eps: Tensor) -> Tensor:
        burnin = self.burnin if self.burnin is not None else 0
        total_T = eps.shape[1]
        d = self.d

        x = torch.zeros((N, total_T, d), device=self.device, dtype=self.dtype)

        for t in range(total_T):

            ar = 0.0
            if self.p:
                for i in range(1, self.p + 1):
                    if t - i >= 0:
                        ar = ar + self.phi[i - 1] * x[:, t - i, :]

            ma = 0.0
            if self.q:
                for j in range(1, self.q + 1):
                    if t - j >= 0:
                        ma = ma + self.theta[j - 1] * eps[:, t - j, :]

            x[:, t, :] = ar + eps[:, t, :] + ma

        return x[:, burnin:, :]
    
    def generate(self, N: int, T: int | None = None, eps: Tensor | None = None):

        if T is None:
            T = self.T

        burnin = self.burnin if self.burnin is not None else 0
        total_T = T + burnin

        if eps is None:
            noise = self.noise or Noise("normal")
            eps = noise.sample((N, total_T, self.d),
                            device=self.device,
                            dtype=self.dtype)
        else:
            eps = eps.to(device=self.device, dtype=self.dtype)

        return self._gen(N, eps)

    def spec(self) -> dict:
        return {
            "name": "ARMA",
            "T": int(self.T),
            "p": int(self.p),
            "q": int(self.q),
            "phi": None if self.phi is None else [float(x) for x in self.phi],
            "theta": None if self.theta is None else [float(x) for x in self.theta],
            "burnin": int(getattr(self, "burnin", 0)),
            "mean": float(getattr(self, "mean", 0.0)),
        }


# -------------------------
# GARCH(1,1) only, simplest useful one
# -------------------------

class GARCH11(Proc):
    # sigma^2_t = omega + alpha * eps^2_{t-1} + beta * sigma^2_{t-1}
    # x_t = sigma_t * z_t    where z_t is the provided noise eps
    def __init__(self, T: int, omega: float, alpha: float, beta: float, d: int = 1, sigma2_0: float = 1.0, burnin: Optional[int] = None, noise: Optional[Noise] = None):
        super().__init__(T, d)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma2_0 = float(sigma2_0)
        self.burnin = int(burnin) if burnin is not None else None
        self.noise = noise
        if self.omega <= 0:
            raise ValueError("omega must be > 0")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha, beta must be >= 0")
        if self.alpha + self.beta >= 1:
            raise ValueError("Need alpha + beta < 1 for (variance) stationarity")

    def _gen(self, N: int, z: Optional[Tensor] = None) -> Tensor:
        d = self.d
        burnin = self.burnin if self.burnin is not None else 0
        # When z is provided, it already includes the burnin steps (added by generate()).
        # Only add burnin when sampling internally (z is None).
        total_T = z.shape[1] if z is not None else self.T + burnin

        device, dtype = self.device, self.dtype

        x = torch.zeros((N, total_T, d), device=device, dtype=dtype)

        sigma2 = torch.full((N, d), self.sigma2_0, device=device, dtype=dtype)
        eps_prev = torch.zeros((N, d), device=device, dtype=dtype)

        # ---- sample noise if not provided ----
        if z is None:
            noise = self.noise or Noise("normal")
            z = noise.sample((N, total_T, d), device=device, dtype=dtype)

        for t in range(total_T):

            sigma2 = self.omega + self.alpha * (eps_prev ** 2) + self.beta * sigma2
            sigma2 = torch.clamp(sigma2, min=1e-12)

            if self.burnin is not None and t < burnin:
                z_t = torch.zeros((N, d), device=device, dtype=dtype)
            else:
                z_t = z[:, t - burnin, :] if self.burnin is not None else z[:, t, :]

            eps_t = torch.sqrt(sigma2) * z_t
            x[:, t, :] = eps_t
            eps_prev = eps_t

        return x[:, burnin:, :] if self.burnin is not None else x
    
    def get_sigma2(self, N: int, z: Tensor) -> Tensor:
        T = z.shape[1]
        d = self.d
        sigma2 = torch.full((N, d), self.sigma2_0, device=self.device, dtype=self.dtype)
        eps_prev = torch.zeros((N, d), device=self.device, dtype=self.dtype)
        sigma2_list = [sigma2]

        for t in range(T):
            sigma2 = self.omega + self.alpha * (eps_prev ** 2) + self.beta * sigma2
            sigma2 = torch.clamp(sigma2, min=1e-12)
            eps_t = torch.sqrt(sigma2) * z[:, t, :]
            eps_prev = eps_t
            sigma2_list.append(sigma2)

        return torch.stack(sigma2_list[1:], dim=1)

    def spec(self) -> dict:
        return {
            "name": "GARCH11",
            "T": int(self.T),
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "sigma2_0": self.sigma2_0,
        }


# -------------------------
# Markov tilted-uniform: a k-Markov process that SATISFIES Theorem A's
# standing assumptions (compact support + density bounded below + Lipschitz),
# unlike AR/GARCH whose Gaussian innovations give unbounded support.
# -------------------------

class MarkovTiltedUniform(Proc):
    """
    k-Markov process on the COMPACT interval [-M, M] with a tilted-uniform
    conditional law:

        p(x | x_{t-1}, ..., x_{t-k}) = (1/(2M)) * (1 + beta * x/M),   x in [-M, M]
        beta = b * tanh( sum_{j=1}^k w_j * x_{t-j}/M ),   with 0 < b < 1  =>  |beta| < 1

    Satisfies Theorem A's standing assumptions:
      (A1) compact support [-M, M];
      (A2) density in [(1-b)/(2M), (1+b)/(2M)] -> bounded above AND below, Lipschitz;
           Lipschitz in the conditioning window (tanh tilt);
      k-Markov by construction; transition density bounded below => Doeblin =>
      uniformly ergodic => unique strictly-stationary law (reached after burn-in).

    Sampled EXACTLY by inverse-CDF. With s = x/M and beta the tilt:
        F(s) = (s+1)/2 + (beta/4)(s^2 - 1)
        s = ( -1 + sqrt((1-beta)^2 + 4*beta*u) ) / beta     (u ~ Unif[0,1]);  s = 2u-1 if beta=0
    The quantile q(u; past) is exactly the Knothe-Rosenblatt map Theorem A's
    proof constructs -- so an ESN matching this is the most honest test of the theory.

    Args:
      w: length-k tilt weights (defines the Markov order and lag dependence).
      b: tilt strength, 0 < b < 1 (keeps the density strictly positive).
      M: support half-width; output lives in [-M, M].
      burnin: steps discarded so the returned block is (approximately) stationary.
    """

    def __init__(
        self,
        T: int,
        w=(0.6, -0.3),
        b: float = 0.8,
        M: float = 1.0,
        d: int = 1,
        burnin: Optional[int] = 200,
    ):
        super().__init__(T, d)
        if not (0.0 < float(b) < 1.0):
            raise ValueError("b must be in (0,1) to keep |beta|<1 (density strictly positive)")
        if float(M) <= 0:
            raise ValueError("M must be > 0")
        w = torch.as_tensor(w, dtype=self.dtype).reshape(-1)
        if w.numel() < 1:
            raise ValueError("w must have at least one weight (Markov order k = len(w))")
        self.k = int(w.numel())
        self.b = float(b)
        self.M = float(M)
        self.burnin = int(burnin) if burnin is not None else 0
        self.register_buffer("w", w)

    def _quantile(self, u: Tensor, beta: Tensor) -> Tensor:
        # tilted-uniform inverse-CDF on s in [-1,1]; u, beta: (N, d)
        disc = ((1.0 - beta) ** 2 + 4.0 * beta * u).clamp_min(0.0)
        safe_beta = torch.where(beta.abs() > 1e-12, beta, torch.ones_like(beta))
        s_tilt = (-1.0 + torch.sqrt(disc)) / safe_beta
        s_lin = 2.0 * u - 1.0                       # beta -> 0 limit (plain uniform)
        return torch.where(beta.abs() > 1e-12, s_tilt, s_lin)

    def _gen(self, N: int, u: Tensor) -> Tensor:
        # u: (N, total_T, d) ~ Unif[0,1] (PIT draws). Returns (N, total_T - burnin, d).
        total_T = u.shape[1]
        d = self.d
        M, w, k, b = self.M, self.w, self.k, self.b
        x = torch.zeros((N, total_T, d), device=self.device, dtype=self.dtype)
        for t in range(total_T):
            arg = torch.zeros((N, d), device=self.device, dtype=self.dtype)
            for j in range(1, k + 1):
                if t - j >= 0:
                    arg = arg + w[j - 1] * (x[:, t - j, :] / M)
            beta = b * torch.tanh(arg)              # |beta| < b < 1
            s = self._quantile(u[:, t, :], beta)    # in [-1, 1]
            x[:, t, :] = M * s
        return x[:, self.burnin:, :]

    @torch.no_grad()
    def generate(self, N: int, T: Optional[int] = None, eps: Optional[Tensor] = None) -> Tensor:
        """
        Returns (N, T, d). `eps`, if given, are the Unif[0,1] PIT draws of shape
        (N, T+burnin, d); otherwise they are sampled internally with torch.rand.
        """
        if T is None:
            T = self.T
        total_T = int(T) + self.burnin
        if eps is None:
            u = torch.rand((N, total_T, self.d), device=self.device, dtype=self.dtype)
        else:
            u = eps.to(device=self.device, dtype=self.dtype)
            if u.shape != (N, total_T, self.d):
                raise ValueError(
                    f"eps (Unif[0,1] PIT draws) must have shape ({N},{total_T},{self.d}), got {tuple(u.shape)}"
                )
        return self._gen(N, u)

    def spec(self) -> dict:
        return {
            "name": "MarkovTiltedUniform",
            "T": int(self.T),
            "k": int(self.k),
            "w": [float(x) for x in self.w],
            "b": self.b,
            "M": self.M,
            "burnin": int(self.burnin),
        }