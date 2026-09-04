#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from lsrr.registry import BACKBONE_REGISTRY, DATA_REGISTRY
from lsrr.data.cache import ShardedHCacheWriter, compute_cache_key
from lsrr.config import load_config_with_cli
# Import data & backbone implementations to trigger registration
import lsrr.data
import lsrr.backbones

def extract_h():
    cfg = load_config_with_cli()

    backbone_cfg = cfg.get("backbone", {"type": "gpt2", "model_name_or_path": "gpt2"})
    data_cfg = cfg.get("data", {"type": "multiplication", "digits": 4})
    extract_cfg = cfg.get("extract", {})

    split = extract_cfg.get("split", "train")
    batch_size = extract_cfg.get("batch_size", 16)
    precision = extract_cfg.get("precision", "float32")
    position_rule = extract_cfg.get("position_rule", "last_token")
    shard_size = extract_cfg.get("shard_size", 2000)
    device = extract_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Extract] Loading backbone: {backbone_cfg}")
    extractor = BACKBONE_REGISTRY.build(backbone_cfg, device=device)

    print(f"[Extract] Loading data: {data_cfg}")
    dataset_mod = DATA_REGISTRY.build(data_cfg)
    samples = dataset_mod.get_split(split)
    print(f"[Extract] Extracted split '{split}' with {len(samples)} samples.")

    # Compute cache key and directory
    b_id = str(backbone_cfg.get("model_name_or_path", backbone_cfg.get("type"))).replace("/", "_")
    d_id = str(data_cfg.get("type", "data"))
    tok_hash = extractor.get_tokenizer_hash()
    cache_key = compute_cache_key(b_id, d_id, split, tok_hash, position_rule, precision)

    base_cache_dir = Path(extract_cfg.get("cache_dir", "caches"))
    out_dir = base_cache_dir / b_id / f"{d_id}_{split}_{cache_key}"
    print(f"[Extract] Output directory: {out_dir}")

    writer = ShardedHCacheWriter(out_dir, shard_size=shard_size, precision=precision)

    start_time = time.time()
    for i in tqdm(range(0, len(samples), batch_size), desc=f"Extracting H ({split})"):
        batch_samples = samples[i:i+batch_size]
        questions = [s.question for s in batch_samples]
        answers = [s.answer for s in batch_samples]

        enc_q = extractor.tokenizer(
            questions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        enc_ans = extractor.tokenizer(
            answers,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            H_batch = extractor.extract_hidden_states(
                input_ids=enc_q["input_ids"],
                attention_mask=enc_q["attention_mask"],
                position_rule=position_rule
            )  # [B, L, d]

        for b in range(len(batch_samples)):
            h_sample = H_batch[b]  # [L, d]
            ans_ids = enc_ans["input_ids"][b]
            # Strip padding from target_ids
            ans_mask = enc_ans["attention_mask"][b]
            clean_ans_ids = ans_ids[ans_mask == 1]
            writer.add_sample(h_sample, target_ids=clean_ans_ids, meta=batch_samples[b].meta)

    manifest_path = writer.finalize(extra_meta={
        "backbone": OmegaConf.to_container(backbone_cfg, resolve=True),
        "data": OmegaConf.to_container(data_cfg, resolve=True),
        "position_rule": position_rule,
        "num_layers": extractor.num_layers,
        "hidden_dim": extractor.hidden_dim,
        "cache_key": cache_key
    })

    elapsed = time.time() - start_time
    print(f"[Extract] Done! Wrote {writer.total_samples} samples across {writer.current_shard_idx} shards in {elapsed:.2f}s.")
    print(f"[Extract] Manifest saved to: {manifest_path}")
    return out_dir

if __name__ == "__main__":
    extract_h()
