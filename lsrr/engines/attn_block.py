import math
from typing import Optional
import torch
import torch.nn as nn
from lsrr.interfaces import BaseRefinementEngine
from lsrr.registry import ENGINE_REGISTRY
from lsrr.engines.wrapper import EngineWrapper

class AttentionBlockCore(nn.Module):
    """Transformer Multi-Head Self-Attention Block with configurable FFN dimension
    to precisely match the parameter budget of HydraQS.
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ffn: Optional[int] = None,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # Default d_ffn to match ~5.8M params if d_model=512
        if d_ffn is None:
            # For d_model=512, hydra_qs core has ~5.87M params.
            # QKVO = 4 * 512^2 = 1,048,576. Remaining ~4.82M / (2 * 512) ≈ 4700.
            self.d_ffn = 4710
        else:
            self.d_ffn = d_ffn

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ffn, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention + residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x

@ENGINE_REGISTRY.register("attn_block")
class AttentionBlockEngine(BaseRefinementEngine):
    """Parameter-matched Attention Block Engine."""
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ffn: Optional[int] = None,
        n_blocks: int = 2,
        damping_alpha: float = 0.5,
        cycle_embedding: bool = True,
        reinject_r0: str = "gate",
        match_hydra_params: bool = True,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks

        # If matching hydra_qs, compute exact d_ffn needed
        if match_hydra_params and d_ffn is None:
            # Target per-block core params of HydraQSCore
            from lsrr.engines.hydra_qs import HydraQSCore
            dummy_hydra = HydraQSCore(d_model=d_model, **kwargs)
            target_params = sum(p.numel() for p in dummy_hydra.parameters())
            # Attn has QKVO + norms = 4 * d_model^2 + 4 * d_model
            base_attn_params = 4 * (d_model ** 2) + 4 * d_model
            # FFN params = 2 * d_model * d_ffn + d_ffn + d_model
            # Solve for d_ffn: target_params = base_attn_params + (2 * d_model + 1) * d_ffn + d_model
            d_ffn = int((target_params - base_attn_params - d_model) / (2 * d_model + 1))

        layers = [
            AttentionBlockCore(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn)
            for _ in range(n_blocks)
        ]
        self.core = nn.Sequential(*layers) if n_blocks > 1 else layers[0]

        self.wrapper = EngineWrapper(
            core_engine=self.core,
            d_model=d_model,
            damping_alpha=damping_alpha,
            cycle_embedding=cycle_embedding,
            reinject_r0=reinject_r0
        )

    def forward_step(self, R_m: torch.Tensor, R0: torch.Tensor, m: int) -> torch.Tensor:
        return self.wrapper(R_m, R0, m)
