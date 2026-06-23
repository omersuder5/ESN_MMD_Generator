from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict
import math
import torch

Tensor = torch.Tensor

# -------------------------
# noise
# -------------------------

@dataclass
class Noise:
    kind: Literal["normal", "t", "gamma", "uniform"] = "normal"
    params: Dict = None

    def sample(self, shape, device=None, dtype=None, generator=None) -> Tensor:
        # generator is honored for normal/uniform/truncated_normal (torch RNG); ignored for
        # t/gamma (torch.distributions.sample has no generator hook) -- seed globally for those.
        # All kinds sample natively on the requested device (GPU-safe).
        params = {} if self.params is None else self.params
        device = device or torch.device("cpu")
        dtype = dtype or torch.get_default_dtype()

        if self.kind == "normal":
            loc = float(params.get("loc", 0.0))
            scale = float(params.get("scale", 1.0))
            return loc + scale * torch.randn(*shape, device=device, dtype=dtype, generator=generator)

        if self.kind == "t":
            df = float(params.get("df", 5.0))
            loc = float(params.get("loc", 0.0))
            scale = float(params.get("scale", 1.0))
            # build params on the target device/dtype so sampling is device-native
            df_t = torch.tensor(df, device=device, dtype=dtype)
            return torch.distributions.StudentT(df_t, loc=loc, scale=scale).sample(shape)

        if self.kind == "gamma":
            # Gamma(concentration, rate). Default: mean=2 if conc=2, rate=1
            conc = float(params.get("concentration", 2.0))
            rate = float(params.get("rate", 1.0))
            conc_t = torch.tensor(conc, device=device, dtype=dtype)
            rate_t = torch.tensor(rate, device=device, dtype=dtype)
            return torch.distributions.Gamma(conc_t, rate_t).sample(shape)

        if self.kind == "uniform":
            # Uniform(-scale, scale); default scale=1 gives U[-1,1], zero-mean
            scale = float(params.get("scale", 1.0))
            return scale * (2.0 * torch.rand(*shape, device=device, dtype=dtype, generator=generator) - 1.0)
        
        if self.kind == "truncated_normal":
            c   = float(params.get("c", 2.5))      # support [-c, c] (in std units)
            loc = float(params.get("loc", 0.0)); scale = float(params.get("scale", 1.0))
            # inverse-CDF via erf/erfinv: lo,hi are python floats (no device), u is on-device,
            # torch.erfinv is device-native -> fully GPU-safe, no CPU<->device tensor mixing.
            INV_SQRT2, SQRT2 = 0.7071067811865476, 1.4142135623730951
            lo = 0.5 * (1.0 + math.erf(-c * INV_SQRT2))   # Phi(-c)
            hi = 0.5 * (1.0 + math.erf( c * INV_SQRT2))   # Phi(c)
            u = torch.rand(*shape, device=device, dtype=dtype, generator=generator)
            z = SQRT2 * torch.erfinv(2.0 * (lo + u * (hi - lo)) - 1.0)   # truncated standard normal
            return loc + scale * z

        raise ValueError(f"unknown noise kind: {self.kind}")

    def spec(self) -> dict:
        return {
            "name": "Noise",
            "kind": str(self.kind),
            "params": {} if self.params is None else {k: float(v) for k, v in self.params.items()},
        }
