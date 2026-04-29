from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict
import torch

Tensor = torch.Tensor

# -------------------------
# noise
# -------------------------

@dataclass
class Noise:
    kind: Literal["normal", "t", "gamma"] = "normal"
    params: Dict = None

    def sample(self, shape, device=None, dtype=None) -> Tensor:
        params = {} if self.params is None else self.params
        device = device or torch.device("cpu")
        dtype = dtype or torch.get_default_dtype()

        if self.kind == "normal":
            loc = float(params.get("loc", 0.0))
            scale = float(params.get("scale", 1.0))
            return loc + scale * torch.randn(*shape, device=device, dtype=dtype)

        if self.kind == "t":
            df = float(params.get("df", 5.0))
            loc = float(params.get("loc", 0.0))
            scale = float(params.get("scale", 1.0))
            return torch.distributions.StudentT(df, loc=loc, scale=scale).sample(shape).to(device=device, dtype=dtype)

        if self.kind == "gamma":
            # Gamma(concentration, rate). Default: mean=2 if conc=2, rate=1
            conc = float(params.get("concentration", 2.0))
            rate = float(params.get("rate", 1.0))
            return torch.distributions.Gamma(conc, rate=rate).sample(shape).to(device=device, dtype=dtype)

        raise ValueError(f"unknown noise kind: {self.kind}")

    def spec(self) -> dict:
        return {
            "name": "Noise",
            "kind": str(self.kind),
            "params": {} if self.params is None else {k: float(v) for k, v in self.params.items()},
        }
