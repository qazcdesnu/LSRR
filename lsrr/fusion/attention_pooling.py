from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from lsrr.interfaces import BaseFusionHead
from lsrr.registry import FUSION_REGISTRY

@FUSION_REGISTRY.register("attention_pooling")
class AttentionPoolingFusionHead(BaseFusionHead):
    """Attention pooling over layers:
    alpha_l = softmax(w^T r_l^*)
    h_ssm = sum_l alpha_l * r_l^*
    h_fusion = h^L + W_r * h_ssm (or gate / concat)
    """
    def __init__(
        self,
        d_model: int = 512,
        d_out: Optional[int] = None,
        fusion_type: Literal["residual", "gate", "concat"] = "residual",
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out or d_model
        self.fusion_type = fusion_type

        # Learnable attention vector w
        self.score_proj = nn.Linear(d_model, 1, bias=False)

        # Reasoning projection W_r
        self.w_r = nn.Linear(d_model, self.d_out)

        if fusion_type == "gate":
            self.gate_proj = nn.Linear(self.d_out + d_model, self.d_out)
        elif fusion_type == "concat":
            self.combine_proj = nn.Linear(self.d_out + d_model, self.d_out)

    def forward(
        self,
        R_star: torch.Tensor,
        h_orig_L: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            R_star: [B, L, d_model]
            h_orig_L: [B, d_ctx] or None
        Returns:
            h_fusion: [B, d_out]
            alpha: [B, L]
        """
        B, L, d = R_star.shape
        # Compute alpha_l = softmax(w^T r_l^*)
        scores = self.score_proj(R_star).squeeze(-1)  # [B, L]
        alpha = F.softmax(scores, dim=-1)  # [B, L]

        # h_ssm = sum_l alpha_l * r_l^*
        h_ssm = torch.einsum("bl,bld->bd", alpha, R_star)  # [B, d_model]
        projected_ssm = self.w_r(h_ssm)  # [B, d_out]

        if h_orig_L is None:
            # If no backbone context vector given, use top layer R_star[:, -1, :]
            h_orig_L = R_star[:, -1, :]

        if self.fusion_type == "residual":
            h_fusion = h_orig_L + projected_ssm
        elif self.fusion_type == "gate":
            g = torch.sigmoid(self.gate_proj(torch.cat([h_orig_L, projected_ssm], dim=-1)))
            h_fusion = (1.0 - g) * h_orig_L + g * projected_ssm
        elif self.fusion_type == "concat":
            h_fusion = self.combine_proj(torch.cat([h_orig_L, projected_ssm], dim=-1))
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        return h_fusion, alpha
