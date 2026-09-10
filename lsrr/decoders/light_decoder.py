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
        d_model: int = 768,
        vocab_size: int = 50257,  # Default GPT-2 vocab size
        n_layers: int = 2,
        n_heads: int = 8,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        bos_token_id: Optional[int] = 50256,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        if bos_token_id is None or bos_token_id >= vocab_size:
            self.bos_token_id = 0
        else:
            self.bos_token_id = bos_token_id

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
        """Teacher-forcing forward pass with proper causal right-shift.
        At position t, the decoder only sees target_ids[:, <t], eliminating self-copy target leakage.
        h_fusion: [B, d_model]
        target_ids: [B, seq_len]
        Returns: logits [B, seq_len, vocab_size]
        """
        B = h_fusion.shape[0]
        # Treat h_fusion as memory: [B, 1, d_model]
        memory = h_fusion.unsqueeze(1)

        if target_ids is None or target_ids.size(1) == 0:
            # Single step prediction directly from BOS token
            bos = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=h_fusion.device)
            positions = torch.zeros((1, 1), dtype=torch.long, device=h_fusion.device)
            tgt_emb = self.tok_embeddings(bos) + self.pos_embeddings(positions)
            out = self.decoder(tgt=tgt_emb, memory=memory)
            logits = self.lm_head(out)  # [B, 1, vocab_size]
            return logits

        seq_len = target_ids.size(1)
        # Proper autoregressive right shift:
        # Decoder input is [BOS, target_ids[:, 0], ..., target_ids[:, seq_len - 2]]
        # so position t only sees target_ids[:, <t], eliminating self-copy target leakage!
        bos = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=target_ids.device)
        decoder_input_ids = torch.cat([bos, target_ids[:, :-1]], dim=1)  # [B, seq_len]

        positions = torch.arange(seq_len, device=target_ids.device).unsqueeze(0)
        tgt_emb = self.tok_embeddings(decoder_input_ids) + self.pos_embeddings(positions)

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
        """Autoregressive generation without target_ids."""
        B = h_fusion.shape[0]
        device = h_fusion.device
        memory = h_fusion.unsqueeze(1)

        start_bos = self.bos_token_id if bos_token_id is None else bos_token_id
        if start_bos >= self.vocab_size:
            start_bos = 0

        # An eos id outside this decoder's vocabulary can never be produced; treating it
        # as "no eos" keeps small test vocabularies from silently disabling truncation.
        eos = eos_token_id if (eos_token_id is not None and eos_token_id < self.vocab_size) else None

        cur_ids = torch.full((B, 1), start_bos, dtype=torch.long, device=device)
        gen_tokens_list = []
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            seq_len = cur_ids.size(1)
            if seq_len > self.max_seq_len:
                break
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            tgt_emb = self.tok_embeddings(cur_ids) + self.pos_embeddings(positions)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
            next_logits = self.lm_head(out[:, -1, :])  # [B, vocab_size]
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            if eos is not None:
                # Per-sequence truncation: once a sequence has emitted EOS, every later
                # slot is filled with EOS so decoding with skip_special_tokens drops the
                # tail. Without this, post-EOS text reaches the answer scorer.
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos),
                    next_token
                )

            gen_tokens_list.append(next_token)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)

            if eos is not None:
                finished = finished | (next_token.squeeze(1) == eos)
                if bool(finished.all()):
                    break

        if gen_tokens_list:
            return torch.cat(gen_tokens_list, dim=1)
        else:
            return torch.empty((B, 0), dtype=torch.long, device=device)

