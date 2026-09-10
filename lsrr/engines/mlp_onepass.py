import torch
import torch.nn as nn
from lsrr.interfaces import BaseRefinementEngine
from lsrr.registry import ENGINE_REGISTRY

@ENGINE_REGISTRY.register("mlp_onepass")
class MLPOnePassEngine(BaseRefinementEngine):
    """1-pass MLP baseline (ablation for recurrence, used with M=1)."""
    def __init__(
        self,
        d_model: int = 768,
        expand: int = 4,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        d_ffn = expand * d_model
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model)
        )

    def forward_step(self, R_m: torch.Tensor, R0: torch.Tensor, m: int) -> torch.Tensor:
        # Residual MLP pass
        return R_m + self.net(R_m)
