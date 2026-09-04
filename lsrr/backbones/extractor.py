import hashlib
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from lsrr.interfaces import BaseBackboneExtractor
from lsrr.registry import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register("hf_causal")
@BACKBONE_REGISTRY.register("gpt2")
class HFCausalBackboneExtractor(BaseBackboneExtractor):
    """Extracts layer-wise hidden states H ∈ R^{L x d} from frozen HuggingFace causal LM."""
    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        torch_dtype: str = "float32",
        device: Optional[str] = None,
        include_embedding: bool = False,
        **kwargs
    ):
        self.model_name = model_name_or_path
        self.include_embedding = include_embedding

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        dtype = torch.bfloat16 if torch_dtype == "bfloat16" else (torch.float16 if torch_dtype == "float16" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype
        ).to(self.device)

        # Strictly freeze backbone: [결정 D-1]
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self._num_layers = self.model.config.num_hidden_layers
        self._hidden_dim = self.model.config.hidden_size

    @property
    def num_layers(self) -> int:
        return self._num_layers + (1 if self.include_embedding else 0)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def get_tokenizer_hash(self) -> str:
        vocab_keys = list(self.tokenizer.get_vocab().keys())[:100]
        return hashlib.sha256("".join(vocab_keys).encode("utf-8")).hexdigest()[:12]

    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_rule: str = "last_token"
    ) -> torch.Tensor:
        """Extract H ∈ R^{B x L x d} at specified token position (default: last token)."""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors:
        # index 0: embedding output
        # index 1..L: output of each layer
        all_hidden = outputs.hidden_states
        if not self.include_embedding:
            all_hidden = all_hidden[1:]

        # Stack over layer dimension: [L, B, T, d] -> [B, L, T, d]
        stacked = torch.stack(all_hidden, dim=1)

        if position_rule == "last_token":
            if attention_mask is not None:
                # Find last active token index for each sequence in batch
                last_indices = attention_mask.sum(dim=1) - 1  # [B]
                B, L, T, d = stacked.shape
                batch_idx = torch.arange(B, device=self.device)
                H = stacked[batch_idx, :, last_indices, :]  # [B, L, d]
            else:
                H = stacked[:, :, -1, :]  # [B, L, d]
        elif position_rule == "mean_pooling":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, T, 1]
                H = (stacked * mask_expanded).sum(dim=2) / mask_expanded.sum(dim=2).clamp(min=1.0)
            else:
                H = stacked.mean(dim=2)
        else:
            raise ValueError(f"Unknown position_rule: {position_rule}")

        return H
