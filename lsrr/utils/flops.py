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


def estimate_backbone_flops_per_forward(
    num_layers: int,
    d_model: int,
    seq_len: int
) -> float:
    """Analytical FLOPs for ONE frozen-transformer forward pass over `seq_len` tokens.

    Per layer, per token:
      QKVO projections    2 * 4 * d^2      = 8 * d^2
      attention scores+AV 2 * 2 * seq * d  = 4 * seq * d
      FFN (4d expansion)  2 * 8 * d^2      = 16 * d^2
    The LM head is excluded: LSRR only reads hidden states from this pass.

    This is the cost the proposal's efficiency claim turns on. Coconut-style rollouts pay
    it once per latent step; LSRR pays it exactly once, so it must be counted, not omitted.
    """
    per_layer_per_token = 24.0 * (d_model ** 2) + 4.0 * seq_len * d_model
    return float(num_layers) * float(seq_len) * per_layer_per_token


def estimate_decoder_flops_per_token(
    d_model: int,
    n_layers: int,
    vocab_size: int,
    seq_len: int = 1,
    memory_len: int = 1
) -> float:
    """Analytical FLOPs to emit ONE answer token from the light decoder.

    Per layer: self-attention (QKVO + scores) + cross-attention over the h_fusion memory
    + a 4d FFN. Plus the tied LM head projection over the vocabulary.
    """
    self_attn = 8.0 * (d_model ** 2) + 4.0 * seq_len * d_model
    cross_attn = 8.0 * (d_model ** 2) + 4.0 * memory_len * d_model
    ffn = 16.0 * (d_model ** 2)
    lm_head = 2.0 * d_model * vocab_size
    return float(n_layers) * (self_attn + cross_attn + ffn) + lm_head


def estimate_adapter_flops(d_in: int, d_model: int, num_layers: int) -> float:
    """Per-layer affine projection of H into the reasoning space."""
    return 2.0 * float(num_layers) * float(d_in) * float(d_model)
