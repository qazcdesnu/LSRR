import math
from typing import Optional
import torch
import torch.nn as nn
from lsrr.interfaces import BaseLayerAdapter
from lsrr.registry import ADAPTER_REGISTRY

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

@ADAPTER_REGISTRY.register("identity")
class IdentityAdapter(BaseLayerAdapter):
    def __init__(self, d_in: int, d_model: Optional[int] = None, **kwargs):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model or d_in
        if self.d_in != self.d_model:
            self.proj = nn.Linear(self.d_in, self.d_model, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.proj(H)

@ADAPTER_REGISTRY.register("per_layer_affine")
@ADAPTER_REGISTRY.register("per_layer_affine+rmsnorm")
class PerLayerAffineAdapter(BaseLayerAdapter):
    """Layer-wise affine transformation with optional RMSNorm and layer position embeddings."""
    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        num_layers: int = 12,
        layer_pos_emb: bool = True,
        use_rmsnorm: bool = True,
        **kwargs
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_layers = num_layers
        self.layer_pos_emb = layer_pos_emb
        self.use_rmsnorm = use_rmsnorm

        # Separate linear projection for each layer l ∈ [0..L-1]
        # Weight shape: [num_layers, d_in, d_model], Bias shape: [num_layers, d_model]
        self.weight = nn.Parameter(torch.empty(num_layers, d_in, d_model))
        self.bias = nn.Parameter(torch.zeros(num_layers, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_rmsnorm:
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.Identity()

        if self.layer_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, num_layers, d_model) * 0.02)
        else:
            self.pos_emb = None

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Args:
            H: [B, L, d_in]
        Returns:
            R0: [B, L, d_model]
        """
        B, L, _ = H.shape
        # Batched matrix multiplication: H[:, l, :] @ weight[l, :, :] + bias[l, :]
        # torch.einsum: b l i, l i o -> b l o
        out = torch.einsum("bli,lio->blo", H, self.weight) + self.bias
        out = self.norm(out)

        if self.pos_emb is not None:
            out = out + self.pos_emb[:, :L, :]

        return out

@ADAPTER_REGISTRY.register("shared_affine")
class SharedAffineAdapter(BaseLayerAdapter):
    """Shared affine projection across all layers."""
    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        num_layers: int = 12,
        layer_pos_emb: bool = True,
        **kwargs
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.layer_pos_emb = layer_pos_emb
        if layer_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, num_layers, d_model) * 0.02)
        else:
            self.pos_emb = None

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        out = self.proj(H)
        if self.pos_emb is not None:
            out = out + self.pos_emb[:, :H.size(1), :]
        return out
