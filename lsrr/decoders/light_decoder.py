from typing import Optional
import torch
import torch.nn as nn
from lsrr.interfaces import BaseAnswerDecoder
from lsrr.registry import DECODER_REGISTRY

@DECODER_REGISTRY.register("trained_light_decoder")
@DECODER_REGISTRY.register("light")
class TrainedLightDecoder(BaseAnswerDecoder):
    """Small trainable transformer decoder operating directly on h_fusion.
    Allows running the entire training loop without loading the frozen backbone.
    """
    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 50257,  # Default GPT-2 vocab size
        n_layers: int = 2,
        n_heads: int = 8,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between embedding and lm_head for language modeling efficiency
        self.lm_head.weight = self.tok_embeddings.weight

    def forward(
        self,
        h_fusion: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.
        h_fusion: [B, d_model]
        target_ids: [B, seq_len]
        Returns: logits [B, seq_len, vocab_size]
        """
        B = h_fusion.shape[0]
        # Treat h_fusion as memory: [B, 1, d_model]
        memory = h_fusion.unsqueeze(1)

        if target_ids is None or target_ids.size(1) == 0:
            # Single step prediction directly from h_fusion
            logits = self.lm_head(h_fusion).unsqueeze(1)  # [B, 1, vocab_size]
            return logits

        seq_len = target_ids.size(1)
        # Shift targets: inputs are target_ids, autoregressive mask applied
        positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0)
        tgt_emb = self.tok_embeddings(target_ids) + self.pos_embeddings(positions)

        # Causal mask for autoregressive teacher forcing
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=target_ids.device)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=causal_mask
        )  # [B, seq_len, d_model]

        logits = self.lm_head(out)  # [B, seq_len, vocab_size]
        return logits

    def generate(
        self,
        h_fusion: torch.Tensor,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = 50256,
        bos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Autoregressive generation."""
        B = h_fusion.shape[0]
        device = h_fusion.device
        memory = h_fusion.unsqueeze(1)

        if bos_token_id is not None:
            cur_ids = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        else:
            # First token predicted directly from memory
            first_logits = self.lm_head(h_fusion)
            first_token = first_logits.argmax(dim=-1, keepdim=True)
            cur_ids = first_token

        for _ in range(max_new_tokens - 1):
            seq_len = cur_ids.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt_emb = self.tok_embeddings(cur_ids) + self.pos_embeddings(positions)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
            next_logits = self.lm_head(out[:, -1, :])
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return cur_ids
