from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from lsrr.interfaces import BaseFusionHead
from lsrr.registry import FUSION_REGISTRY

@FUSION_REGISTRY.register("attention_pooling")
class AttentionPoolingFusionHead(BaseFusionHead):
    """Attention pooling over layers (proposal 3.4):
    alpha_l = softmax(w^T r_l^*)
    h_ssm = sum_l alpha_l * r_l^*
    h_fusion = h_orig_L + W_r * h_ssm (or gate / concat)

    h_orig_L is the backbone's own top-layer context representation, supplied by the
    caller as R0[:, -1, :]. It is required: substituting a slice of R_star would make
    h_fusion depend only on the SSM output and drop the residual bypass.
    """
    def __init__(
        self,
        d_model: int = 768,
        d_out: Optional[int] = None,
        fusion_type: Literal["residual", "gate", "concat"] = "residual",
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out or d_model
        self.fusion_type = fusion_type

        if self.fusion_type == "residual" and self.d_out != self.d_model:
            raise ValueError(
                f"residual fusion adds h_orig_L (d_model={self.d_model}) to W_r * h_ssm "
                f"(d_out={self.d_out}); the two must share a dimension. Set d_out=d_model."
            )

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
            R_star: [B, L, d_model] final refined state
            h_orig_L: [B, d_model] adapted backbone context representation, i.e. R0[:, -1, :]
        Returns:
            h_fusion: [B, d_out]
            alpha: [B, L]
        """
        B, L, d = R_star.shape

        if h_orig_L is None:
            raise ValueError(
                "h_orig_L is required: proposal 3.4 fuses the refined reasoning state onto "
                "the backbone context representation h^(L). Pass R0[:, -1, :] (the adapter "
                "output, before refinement), not a slice of R_star."
            )
        if h_orig_L.shape != (B, self.d_model):
            raise ValueError(
                f"h_orig_L must be [B, d_model] = [{B}, {self.d_model}], got {tuple(h_orig_L.shape)}."
            )

        # Compute alpha_l = softmax(w^T r_l^*)
        scores = self.score_proj(R_star).squeeze(-1)  # [B, L]
        alpha = F.softmax(scores, dim=-1)  # [B, L]

        # h_ssm = sum_l alpha_l * r_l^*
        h_ssm = torch.einsum("bl,bld->bd", alpha, R_star)  # [B, d_model]
        projected_ssm = self.w_r(h_ssm)  # [B, d_out]

        if self.fusion_type == "residual":
            # h_fusion = h^(L) + W_r * h_ssm
            h_fusion = h_orig_L + projected_ssm
        elif self.fusion_type == "gate":
            g = torch.sigmoid(self.gate_proj(torch.cat([h_orig_L, projected_ssm], dim=-1)))
            h_fusion = (1.0 - g) * h_orig_L + g * projected_ssm
        elif self.fusion_type == "concat":
            h_fusion = self.combine_proj(torch.cat([h_orig_L, projected_ssm], dim=-1))
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        return h_fusion, alpha
