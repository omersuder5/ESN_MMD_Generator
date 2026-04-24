
import numpy as np
import torch
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.arima.model import ARIMA

from tqdm import tqdm
from utils.data import *
from loss.loss import compute_mmd_loss

# ================================================================================================================
# Computes the baseline MMD distribution by comparing two independent samples from the target generator.
def compute_baseline_mmd(target_generator, kernel, noise, iterations, N_small, N_large, kernel_mode, dtype, device, target_type):

    baseline_mmd = torch.empty(iterations, dtype=dtype)

    with torch.no_grad():
        for i in tqdm(range(iterations)):
            X_small_raw = target_generator.generate(N=N_small, noise=noise).to(device=device, dtype=dtype)
            X_large_raw = target_generator.generate(N=N_large, noise=noise).to(device=device, dtype=dtype)

            eps = 1e-8  # small constant to avoid log(0)
            if target_type == "returns":
                X_small = X_small_raw
                X_large = X_large_raw
            elif target_type == "log_returns":
                X_small = torch.log1p(X_small_raw)
                X_large = torch.log1p(X_large_raw)
            elif target_type == "sqrd_log_returns":
                log_r = torch.log1p(X_small_raw)
                X_small = log_r ** 2
                X_large = torch.log1p(X_large_raw) ** 2
            elif target_type == "log_sqrd_returns":
                X_small = torch.log(X_small_raw ** 2 + eps)
                X_large = torch.log(X_large_raw ** 2 + eps)

            if kernel_mode == "static":
                mmd_val = compute_mmd_loss(
                    kernel,
                    X_small.reshape(N_small, -1),
                    X_large.reshape(N_large, -1),
                )
            else:
                mmd_val = compute_mmd_loss(kernel, X_small, X_large)

            baseline_mmd[i] = mmd_val.detach().cpu()

    fmt = lambda x: f"{float(x):.6f}"

    baseline_stats = {
        "iterations": iterations,
        "N_small": N_small,
        "N_large": N_large,
        "mean": fmt(baseline_mmd.mean()),
        "std": fmt(baseline_mmd.std(unbiased=True)),
        "median": fmt(baseline_mmd.median()),
        "q05": fmt(torch.quantile(baseline_mmd, 0.05)),
        "q95": fmt(torch.quantile(baseline_mmd, 0.95)),
        "min": fmt(baseline_mmd.min()),
        "max": fmt(baseline_mmd.max()),
    }
    return baseline_stats

# ================================================================================================================

# ============================================================
# ACF utilities
# ============================================================

def _acf_vectors(X, lag, component=0, square=False):
    """
    X: (N, T, d)
    returns: (N, K)
    """
    X = X.detach()[..., component].cpu().numpy()
    N, T = X.shape
    lag_eff = min(lag, T - 1)

    out = np.empty((N, lag_eff))

    for i in range(N):
        x = X[i]
        if square:
            x = x * x
        a = sm_acf(x, nlags=lag_eff, fft=True)
        out[i] = a[1:]  # drop lag 0

    return out


def _acf_mean_std(X, lag, component=0, square=False):
    A = _acf_vectors(X, lag, component, square)
    return A.mean(axis=0), A.std(axis=0)


# ============================================================
# Plotting (overlapping bars)
# ============================================================

def _plot_acf_compare(ax, lags, m1, m2, s1=None, s2=None, title=""):
    width = 0.4

    ax.bar(lags - width/2, m1, width=width, alpha=0.7, label="target")
    ax.bar(lags + width/2, m2, width=width, alpha=0.7, label="esn")

    if s1 is not None:
        ax.errorbar(lags - width/2, m1, yerr=s1, fmt="none", capsize=2)
    if s2 is not None:
        ax.errorbar(lags + width/2, m2, yerr=s2, fmt="none", capsize=2)

    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.legend()


# ============================================================
# Main ACF analysis
# ============================================================

def acf_analysis(
    esn,
    *,
    Z_target=None,
    target_generator=None,
    N,
    T,
    lag=40,
    component=0,
    device=None,
    dtype=torch.float64,
    show_paths=True,
):
    if (Z_target is None) == (target_generator is None):
        raise ValueError("Provide exactly one of Z_target or target_generator")
    
    
    with torch.no_grad():

        # ---- target resolution (NO resampling, NO noise here) ----
        if target_generator is not None:
            Z_target = target_generator.generate(N=N, T=T)

        if Z_target.ndim != 3:
            raise ValueError("Z_target must be (N,T,d)")

        Z_target = Z_target.to(device=device, dtype=dtype)

        # ---- ESN generation (self-contained stochasticity) ----
        Z_esn = esn(T=T, N=N).to(device=device, dtype=dtype)

    N = min(N, Z_target.shape[0])
    T = min(T, Z_target.shape[1])

    Z_target = Z_target[:N, :T].to(device=device, dtype=dtype)
    Z_esn = Z_esn[:N, :T]

    lag_eff = min(lag, T - 1)
    lags = np.arange(1, lag_eff + 1)

    # ACF(x)
    acf_tgt, acf_tgt_std = _acf_mean_std(Z_target, lag, component, False)
    acf_esn, acf_esn_std = _acf_mean_std(Z_esn, lag, component, False)

    # ACF(x^2)
    acf2_tgt, acf2_tgt_std = _acf_mean_std(Z_target, lag, component, True)
    acf2_esn, acf2_esn_std = _acf_mean_std(Z_esn, lag, component, True)

    # drop lag 0 already done → vectors align with lags
    # plotting
    if show_paths:
        fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
        ax[0].plot(Z_target[0, :, component].cpu().numpy())
        ax[0].set_title("Target path")
        ax[1].plot(Z_esn[0, :, component].cpu().numpy())
        ax[1].set_title("ESN path")
        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    _plot_acf_compare(
        ax[0], lags,
        acf_tgt, acf_esn,
        acf_tgt_std, acf_esn_std,
        title="ACF of x_t"
    )

    _plot_acf_compare(
        ax[1], lags,
        acf2_tgt, acf2_esn,
        acf2_tgt_std, acf2_esn_std,
        title="ACF of x_t^2"
    )

    ax[1].set_xlabel("lag")
    for a in ax:
        a.set_ylabel("acf")

    plt.tight_layout()
    plt.show()

    return {
        "lags": lags,
        "acf_target": acf_tgt,
        "acf_esn": acf_esn,
        "acf2_target": acf2_tgt,
        "acf2_esn": acf2_esn,
    }


# ============================================================
# Two-sample permutation test on ACF
# ============================================================

def acf_two_sample_test(
    Z_target,
    Z_esn,
    *,
    lag=40,
    component=0,
    n_perm=1000,
    use_squared=False,
    joint=False,
    normalize=True,
    seed=None,
):
    if seed is not None:
        np.random.seed(seed)

    if joint:
        A1 = _acf_vectors(Z_target, lag, component, False)
        B1 = _acf_vectors(Z_esn, lag, component, False)

        A2 = _acf_vectors(Z_target, lag, component, True)
        B2 = _acf_vectors(Z_esn, lag, component, True)

        A = np.concatenate([A1, A2], axis=1)
        B = np.concatenate([B1, B2], axis=1)

    else:
        A = _acf_vectors(Z_target, lag, component, use_squared)
        B = _acf_vectors(Z_esn, lag, component, use_squared)

    if normalize:
        pooled = np.vstack([A, B])
        std = pooled.std(axis=0) + 1e-8
        A = A / std
        B = B / std

    mean_A = A.mean(axis=0)
    mean_B = B.mean(axis=0)

    T_obs = np.sum((mean_A - mean_B) ** 2)

    pooled = np.vstack([A, B])
    N = A.shape[0]

    T_perm = np.empty(n_perm)

    for b in range(n_perm):
        idx = np.random.permutation(2 * N)
        A_b = pooled[idx[:N]]
        B_b = pooled[idx[N:]]
        T_perm[b] = np.sum((A_b.mean(0) - B_b.mean(0)) ** 2)

    p_value = np.mean(T_perm >= T_obs)

    return {
        "T_obs": float(T_obs),
        "p_value": float(p_value),
        "T_perm": T_perm,
    }

# ================================================================================================================

# =============================================================================
# ARMA fitting utilities (for analyzing the fitted ARMA parameters on ESN vs target)
# =============================================================================

def fit_arma_mle(x, p, q):
    # ensure numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    model = ARIMA(x, order=(p, 0, q), trend="n")
    # model = ARIMA(x, order=(p, 0, q), trend="n", enforce_stationarity=False, enforce_invertibility=False) (use if uncertain of the orders)
    res = model.fit()

    phi = res.arparams if p > 0 else np.array([])
    theta = res.maparams if q > 0 else np.array([])

    return {
        "phi": phi,
        "theta": theta,
        "params": res.params,
        "cov": res.cov_params(),
    }

def fit_arma_on_paths(paths, p, q):
    if isinstance(paths, torch.Tensor):
        paths = paths.detach().cpu().numpy()

    phis, thetas = [], []

    for i in range(paths.shape[0]):
        res = fit_arma_mle(paths[i], p, q)
        phis.append(res["phi"])
        thetas.append(res["theta"])

    return np.array(phis), np.array(thetas)

def summarize_params(phis, thetas):
    summary = {}

    if phis.size > 0:
        summary["phi_mean"] = phis.mean(axis=0)
        summary["phi_std"] = phis.std(axis=0)

    if thetas.size > 0:
        summary["theta_mean"] = thetas.mean(axis=0)
        summary["theta_std"] = thetas.std(axis=0)

    return summary