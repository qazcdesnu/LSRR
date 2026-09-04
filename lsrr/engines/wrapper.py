from typing import Optional, Literal
import torch
import torch.nn as nn
from lsrr.interfaces import BaseRefinementEngine

class EngineWrapper(nn.Module):
    """Universal wrapper providing:
    1. Damped update: R' = (1 - alpha) * R + alpha * f(R)
    2. Cycle embedding: m-conditioning (learnable embedding for step m)
    3. Input reinjection: R0 reinjected via add / concat / gate
    4. LayerNorm / RMSNorm pre/post-conditioning
    """
    def __init__(
        self,
        core_engine: nn.Module,
        d_model: int = 512,
        max_cycles: int = 32,
        damping_alpha: float = 0.5,
        cycle_embedding: bool = True,
        reinject_r0: Optional[Literal["none", "add", "concat", "gate"]] = "gate",
        norm_type: str = "rmsnorm"
    ):
        super().__init__()
        self.core_engine = core_engine
        self.d_model = d_model
        self.max_cycles = max_cycles
        self.damping_alpha = damping_alpha
        self.cycle_embedding = cycle_embedding
        self.reinject_r0 = reinject_r0

        if cycle_embedding:
            self.cycle_emb = nn.Embedding(max_cycles, d_model)
        else:
            self.cycle_emb = None

        if reinject_r0 == "gate":
            # Gating mechanism: g = sigmoid(W_g [R_m, R_0])
            self.gate_proj = nn.Linear(2 * d_model, d_model)
        elif reinject_r0 == "concat":
            self.concat_proj = nn.Linear(2 * d_model, d_model)

        self.norm = nn.LayerNorm(d_model) if norm_type == "layernorm" else nn.RMSNorm(d_model)

    def forward(
        self,
        R_m: torch.Tensor,
        R0: torch.Tensor,
        m: int
    ) -> torch.Tensor:
        """Args:
            R_m: [B, L, d_model]
            R0: [B, L, d_model]
            m: current cycle index
        Returns:
            R_{m+1}: [B, L, d_model]
        """
        B, L, d = R_m.shape
        x = R_m

        # 1. Cycle embedding
        if self.cycle_emb is not None:
            m_clamped = min(m, self.max_cycles - 1)
            m_tensor = torch.tensor(m_clamped, device=R_m.device, dtype=torch.long)
            c_emb = self.cycle_emb(m_tensor).view(1, 1, d)  # [1, 1, d]
            x = x + c_emb

        # 2. R0 reinjection before engine step
        if self.reinject_r0 == "add":
            x = x + R0
        elif self.reinject_r0 == "concat":
            x = self.concat_proj(torch.cat([x, R0], dim=-1))
        elif self.reinject_r0 == "gate":
            g = torch.sigmoid(self.gate_proj(torch.cat([x, R0], dim=-1)))
            x = (1.0 - g) * x + g * R0

        # Pre-norm
        x_normed = self.norm(x)

        # 3. Core engine forward step
        f_R = self.core_engine(x_normed)

        # 4. Damped update: (1 - alpha) * R_m + alpha * f(R)
        R_next = (1.0 - self.damping_alpha) * R_m + self.damping_alpha * f_R
        return R_next
