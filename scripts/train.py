#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from lsrr.config import load_config_with_cli
from lsrr.utils.seed import set_seed
from lsrr.utils.logging import ExperimentTracker
from lsrr.data.cache import ShardedHCacheDataset, compute_cache_key
from lsrr.data.schema import StandardDataBatch
from lsrr.losses.composite import CompositeLoss
from lsrr.model import LSRRModel
from lsrr.training.trainer import Trainer
from lsrr.registry import BACKBONE_REGISTRY, DATA_REGISTRY
import lsrr.data
import lsrr.backbones

def collate_h_cache(batch):
    H = torch.stack([item["H"] for item in batch], dim=0)
    
    # Pad target_ids to maximum length in batch
    max_target_len = max((item["target_ids"].size(0) for item in batch), default=0)
    target_ids = torch.full((len(batch), max_target_len), fill_value=0, dtype=torch.long)
    for i, item in enumerate(batch):
        t_len = item["target_ids"].size(0)
        if t_len > 0:
            target_ids[i, :t_len] = item["target_ids"]
            
    metas = [item["meta"] for item in batch]
    return {
        "H": H,
        "target_ids": target_ids,
        "meta": metas
    }

def ensure_cached_data(cfg, split="train", device="cpu"):
    """Check if cache exists, if not extract on the fly."""
    backbone_cfg = cfg.get("backbone", {"type": "gpt2", "model_name_or_path": "gpt2"})
    data_cfg = cfg.get("data", {"type": "multiplication", "digits": 4})
    extract_cfg = cfg.get("extract", {})

    b_id = str(backbone_cfg.get("model_name_or_path", backbone_cfg.get("type"))).replace("/", "_")
    d_id = str(data_cfg.get("type", "data"))
    precision = extract_cfg.get("precision", "float32")
    pos_rule = extract_cfg.get("position_rule", "last_token")
    base_cache_dir = Path(extract_cfg.get("cache_dir", "caches"))

    # Need tokenizer hash
    extractor = BACKBONE_REGISTRY.build(backbone_cfg, device=device)
    tok_hash = extractor.get_tokenizer_hash()
    cache_key = compute_cache_key(b_id, d_id, split, tok_hash, pos_rule, precision)

    cache_dir = base_cache_dir / b_id / f"{d_id}_{split}_{cache_key}"
    if not (cache_dir / "manifest.json").exists():
        print(f"[Train] Cache not found at {cache_dir}. Extracting H for split '{split}'...")
        from scripts.extract_h import extract_h
        extract_cfg_copy = OmegaConf.create(dict(cfg))
        extract_cfg_copy.extract.split = split
        extract_cfg_copy.extract.device = str(device)
        # Call extraction directly
        from lsrr.data.cache import ShardedHCacheWriter
        dataset_mod = DATA_REGISTRY.build(data_cfg)
        samples = dataset_mod.get_split(split)
        writer = ShardedHCacheWriter(cache_dir, shard_size=extract_cfg.get("shard_size", 5000), precision=precision)

        from lsrr.utils.oom import process_batch_with_oom_recovery
        bs = extract_cfg.get("batch_size", 16)
        for i in range(0, len(samples), bs):
            chunk = samples[i:i+bs]
            process_batch_with_oom_recovery(extractor, writer, chunk, position_rule=pos_rule)
        writer.finalize()

    return cache_dir, extractor.num_layers, extractor.hidden_dim

def run_single_seed(cfg, seed: int, exp_name: str, device: torch.device):
    set_seed(seed, cfg.get("deterministic", True))
    tracker = ExperimentTracker(exp_name=exp_name, cfg=cfg, seed=seed)

    # 1. Prepare data loaders from cache
    train_cache_dir, num_layers, hidden_dim = ensure_cached_data(cfg, split="train", device=device)
    val_cache_dir, _, _ = ensure_cached_data(cfg, split="val", device=device)

    train_ds = ShardedHCacheDataset(train_cache_dir)
    val_ds = ShardedHCacheDataset(val_cache_dir)

    batch_size = cfg.get("train", {}).get("bs", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_h_cache)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_h_cache)

    # 2. Build LSRR Model
    model = LSRRModel(
        adapter_cfg=cfg.get("adapter", {"type": "per_layer_affine+rmsnorm"}),
        engine_cfg=cfg.get("engine", {"type": "mamba_up"}),
        fusion_cfg=cfg.get("fusion", {"type": "attention_pooling"}),
        decoder_cfg=cfg.get("decoder", {"type": "trained_light_decoder"}),
        iteration_cfg=cfg.get("iteration", {}),
        termination_cfg=cfg.get("termination", {}),
        d_in=hidden_dim,
        num_layers=num_layers
    )

    # 3. Build Loss
    losses_cfg = cfg.get("losses", [{"type": "answer_nll", "w": 1.0}])
    loss_fn = CompositeLoss(losses_cfg)

    # 4. Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        cfg=cfg,
        tracker=tracker,
        device=device
    )
    summary = trainer.fit()
    return summary

def main():
    cfg = load_config_with_cli()

    # Determine device
    device_cfg = cfg.get("device", "auto")
    if device_cfg == "auto":
        # Check if CUDA works without error
        if torch.cuda.is_available():
            try:
                t = torch.zeros(1, device="cuda")
                device = torch.device("cuda")
            except Exception:
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_cfg)

    print(f"[Train] Using device: {device}")

    # Extract seeds
    train_cfg = cfg.get("train", {})
    seeds = train_cfg.get("seeds", [cfg.get("seed", 42)])
    if isinstance(seeds, int):
        seeds = [seeds]

    exp_name = cfg.get("exp_name", "lsrr_experiment")
    # If cli had exp=..., name from that
    for arg in sys.argv:
        if arg.startswith("exp="):
            exp_name = arg.split("=")[1]

    results = []
    for s in seeds:
        print(f"\n===== Running Seed {s} =====")
        res = run_single_seed(cfg, seed=s, exp_name=exp_name, device=device)
        results.append(res)

    print("\n[Train] All seeds finished successfully!")
    for s, r in zip(seeds, results):
        print(f"  Seed {s} -> Best Val Loss: {r.get('best_val_loss', 'N/A')}")

if __name__ == "__main__":
    main()
