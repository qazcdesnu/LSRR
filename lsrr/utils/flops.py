import torch
import torch.nn as nn
from typing import Dict, Any

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }

def estimate_engine_flops_per_step(engine_name: str, d_model: int, num_layers: int, L: int) -> float:
    """Analytical FLOPs estimation for 1 refinement step over L layers with dimension d_model.
    For SSM: O(L * d_model) state transitions.
    For Attention: 4 * L * d_model^2 (projections) + 2 * L^2 * d_model (QK^T and AV) + 4 * L * d_model^2 (FFN)
    """
    if "attn" in engine_name:
        # Standard self-attention block with 4x FFN
        # Projections: Q, K, V, O = 4 * L * d_model^2 * 2
        # Attention matrix: 2 * L^2 * d_model
        # FFN: 2 * (L * d_model * 4d_model * 2) = 16 * L * d_model^2
        flops = num_layers * (8 * L * (d_model ** 2) + 2 * (L ** 2) * d_model + 16 * L * (d_model ** 2))
    elif "mamba" in engine_name or "hydra" in engine_name:
        # SSM block with expand factor (typically 2x)
        d_inner = 2 * d_model
        # in_proj: 2 * L * d_model * (2 * d_inner)
        # scan: ~ 6 * L * d_inner (for state dimension d_state=16)
        # out_proj: 2 * L * d_inner * d_model
        flops = num_layers * (4 * L * d_model * d_inner + 12 * L * d_inner + 2 * L * d_inner * d_model)
    else:
        # MLP baseline
        flops = num_layers * (2 * L * d_model * (4 * d_model) + 2 * L * (4 * d_model) * d_model)
    return float(flops)
