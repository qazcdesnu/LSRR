"""Canonical collate for cached hidden-state batches.

Shared by training and evaluation so the two cannot drift apart. It produces two
distinct target tensors:

    target_ids  decoder *input* ids, padded with a real token id
    labels      loss targets, padded with IGNORE_INDEX so padding is never supervised

Keeping them separate matters: feeding IGNORE_INDEX into an embedding would crash, and
supervising the padding is what made 84% of the GSM8K training signal "emit '!'".
"""
from typing import Any, Dict, List, Optional

import torch

IGNORE_INDEX = -100


def _ensure_eos(ids: torch.Tensor, eos_token_id: Optional[int]) -> torch.Tensor:
    """Append EOS unless it is already the final token.

    Caches written before the extractor added EOS carry bare answer tokens; this keeps
    them usable without re-extracting the hidden states.
    """
    if eos_token_id is None:
        return ids
    if ids.numel() > 0 and int(ids[-1]) == eos_token_id:
        return ids
    return torch.cat([ids, torch.tensor([eos_token_id], dtype=ids.dtype)])


def collate_h_cache(
    batch: List[Dict[str, Any]],
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Collate cached samples into a padded batch.

    Args:
        batch: items from ShardedHCacheDataset, whose target_ids are already trimmed.
        eos_token_id: when given, every target is made to end with EOS.
        pad_token_id: filler for decoder input padding; defaults to eos_token_id, else 0.
    """
    H = torch.stack([item["H"] for item in batch], dim=0)

    targets = [_ensure_eos(item["target_ids"], eos_token_id) for item in batch]
    target_lens = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    max_len = int(target_lens.max()) if len(targets) else 0

    pad_id = pad_token_id if pad_token_id is not None else (eos_token_id or 0)
    target_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, t in enumerate(targets):
        n = t.size(0)
        if n > 0:
            target_ids[i, :n] = t
            labels[i, :n] = t

    return {
        "H": H,
        "target_ids": target_ids,
        "labels": labels,
        "target_lens": target_lens,
        "meta": [item["meta"] for item in batch],
    }
