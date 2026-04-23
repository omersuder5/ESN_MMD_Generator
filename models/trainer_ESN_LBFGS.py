from __future__ import annotations

from typing import Any, Optional, Literal, Dict, Tuple

import torch
from loss import loss
from loss.loss import compute_mmd_loss

Tensor = torch.Tensor

@torch.no_grad()
# Fixes the noise and reservoir states for the ESN, so that all optimization steps use the same (deterministic) data.
# This eliminates stochasticity during optimization and ensures reproducibility.
def _sample_states_once(
    esn,
    *,
    T: int,
    N_model: int,
    dtype: torch.dtype,
    device: torch.device,
    xi: Optional[Tensor] = None,
    eta: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
      Z0: (N_model, T, d)  (ignored for optimization, but sometimes useful)
      Xhist: (N_model, T, h)  reservoir states, fixed during optimization
    """
    esn = esn.to(device=device, dtype=dtype).eval()
    total_T = (esn.washout_len + T)

    if eta is None:
        eta = torch.zeros(N_model, total_T, esn.d, device=device, dtype=dtype)

    out = esn(T=T, N=N_model, xi=xi, eta=eta, return_states=True)
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        raise ValueError("esn(T=..., N=..., return_states=True) must return (Z, Xhist)")
    Z0, Xhist = out

    if Z0.shape[:2] != (N_model, T):
        raise ValueError(f"Z0 shape {tuple(Z0.shape)} incompatible with (N_model,T)=({N_model},{T})")
    if Xhist.shape[:2] != (N_model, T):
        raise ValueError(f"Xhist shape {tuple(Xhist.shape)} incompatible with (N_model,T)=({N_model},{T})")

    return Z0, Xhist


def fit_ESN_MMD_LBFGS(
    *,
    esn,
    Z_target: Tensor,                              # (N_target, T, d), fixed
    kernel: Any,
    kernel_mode: Literal["static", "sequential"],
    N_model: int,                                  # number of ESN paths used inside MMD
    lead_lag: bool = False,
    lags: int = 1,
    max_iter: int = 200,
    lr: float = 1.0,
    history_size: int = 20,
    tol_grad: float = 1e-12,
    tol_change: float = 1e-12,
    force_float64: bool = True,
    xi: Optional[Tensor] = None,                    # optional fixed drive for model sampling (N_model,T,m)
    eta: Optional[Tensor] = None,                   # optional fixed output noise (N_model,T,d); default 0
    verbose: bool = True,
    target_type: Optional[str] = None,              # if specified, must be "returns", "log_returns", or "sqrd_log_returns"; applies transformation to Z_target for MMD comparison only (not changing the ESN output or W)
) -> Dict[str, Any]:
    """
    Optimizes W only:
        min_W MMD( Z_target , Z_model(W) )
    where Z_model(W) = Xhist @ W^T (+ tilt if your esn includes it in forward; here we ignore tilt).

    Notes:
    - Z_target is fixed (no resampling).
    - Model states Xhist are sampled once (using xi, eta), then held fixed.
    - If kernel is sigkernel-based, force_float64 should stay True.
    """
    if Z_target.ndim != 3:
        raise ValueError("Z_target must have shape (N_target,T,d)")
    N_target, T, d = Z_target.shape

    device = esn.A.device
    dtype = torch.float64 if force_float64 else esn.A.dtype

    esn = esn.to(device=device, dtype=dtype).eval()
    Z_target = Z_target.to(device=device, dtype=dtype)

    if int(esn.d) != int(d):
        raise ValueError(f"esn.d={int(esn.d)} must match Z_target last dim d={int(d)}")

    # Precompute model reservoir states once (fixes the noise and states for all optimization steps).
    # This ensures the loss landscape is deterministic and reproducible, not affected by random sampling each step.
    _, Xhist = _sample_states_once(
        esn,
        T=T,
        N_model=N_model,
        dtype=dtype,
        device=device,
        xi=xi,
        eta=eta,
    )  # Xhist: (N_model,T,h)

    # Optimize W (copy so we don't mutate esn until the end)
    W = torch.nn.Parameter(esn.W.detach().clone().to(device=device, dtype=dtype))

    opt = torch.optim.LBFGS(
        [W],
        lr=lr,
        max_iter=max_iter,
        history_size=history_size,
        line_search_fn="strong_wolfe",
        tolerance_grad=tol_grad,
        tolerance_change=tol_change,
    )

    def mmd_loss_from_W() -> Tensor:
        Zm_raw = Xhist @ W.T  # (N_model,T,d)

        eps = 1e-8
        if target_type is None:
            Zm = Zm_raw
        elif target_type == "log_returns":
            Zm = torch.log1p(Zm_raw)
        elif target_type == "sqrd_log_returns":
            Zm = torch.log1p(Zm_raw) ** 2
        elif target_type == "log_sqrd_returns":
            Zm = torch.log(Zm_raw ** 2 + eps)

        if kernel_mode == "static":
            Xk = Z_target.reshape(N_target, -1)
            Yk = Zm.reshape(N_model, -1)
        else:
            Xk, Yk = Z_target, Zm

        # compute_mmd_loss is assumed to handle different batch sizes
        return compute_mmd_loss(kernel, Xk, Yk, lead_lag, lags) + 1e-3 * float(torch.linalg.norm(W))
    
    loss_history = []

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = mmd_loss_from_W()
        loss.backward()

        loss_history.append(loss.item())
        if len(loss_history) % 100 == 0:
            print(f"Eval {len(loss_history)}: Loss {loss.item():.6f}")
        
        return loss

    m0 = float(mmd_loss_from_W().detach().cpu())
    if verbose:
        print("MMD initial:", m0)

    opt.step(closure)

    m1 = float(mmd_loss_from_W().detach().cpu())
    if verbose:
        print("MMD final:", m1)

    # Write W back into esn
    with torch.no_grad():
        esn.W.copy_(W)

    return {
        "W_fit": W.detach().cpu(),
        "mmd_initial": m0,
        "mmd_final": m1,
        "T": int(T),
        "N_target": int(N_target),
        "N_model": int(N_model),
        "dtype": str(dtype),
        "loss_history": loss_history,
    }
