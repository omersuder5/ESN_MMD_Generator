import torch
import numpy as np
import matplotlib.pyplot as plt

def compare_Ws(
    esn,
    W_fixed,
    *,
    title: str = "W_fit vs W_fixed",
    scatter: bool = True,
) -> dict:
    """
    Compares the W matrix of a fitted ESN to a reference ESN.
    Works for any (d,h) shape.
    """
    W_fit = esn.W.detach().cpu()
    W_fixed = W_fixed

    if W_fit.shape != W_fixed.shape:
        raise ValueError(f"Shape mismatch: W_fit {tuple(W_fit.shape)} vs W_fixed {tuple(W_fixed.shape)}")

    diff = W_fit - W_fixed

    mse = float((diff ** 2).mean())
    mse_ref = float((W_fixed ** 2).mean())
    rel_mse = float(mse / (mse_ref + 1e-12))

    fro = float(torch.linalg.norm(diff))
    fro_ref = float(torch.linalg.norm(W_fixed))
    rel_fro = float(fro / (fro_ref + 1e-12))

    a = W_fit.reshape(-1).numpy()
    b = W_fixed.reshape(-1).numpy()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")

    print(f"E[W_fixed^2]          = {mse_ref:.6g}")
    print(f"MSE(W_fit, W_fixed)   = {mse:.6g}")
    print(f"relative MSE          = {rel_mse:.6g}")
    print(f"||W_fixed||_F         = {fro_ref:.6g}")
    print(f"||W_fit||_F           = {float(torch.linalg.norm(W_fit)):.6g}")
    print(f"||W_fit-W_fixed||_F   = {fro:.6g}")
    print(f"relative Frobenius    = {rel_fro:.6g}")
    print(f"Corr(flattened)       = {corr:.6g}")
    print(f"W_fit[:5]             = {W_fit.flatten()[:5].numpy()}")
    print(f"W_fixed[:5]           = {W_fixed.flatten()[:5].numpy()}")

    if scatter:
        plt.figure(figsize=(5, 5))
        plt.scatter(b, a, s=10)
        plt.xlabel("W_fixed entries")
        plt.ylabel("W_fit entries")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    return {
        "mse": mse,
        "mse_ref": mse_ref,
        "rel_mse": rel_mse,
        "fro": fro,
        "fro_ref": fro_ref,
        "rel_fro": rel_fro,
        "corr": corr,
    }
