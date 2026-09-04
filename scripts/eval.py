#!/usr/bin/env python3
import sys
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from lsrr.config import load_config_with_cli, resolve_hierarchical_config
from lsrr.model import LSRRModel
from lsrr.training.evaluator import Evaluator
from lsrr.data.cache import ShardedHCacheDataset
from lsrr.registry import BACKBONE_REGISTRY, DATA_REGISTRY
import lsrr.data
import lsrr.backbones

def collate_h_cache(batch):
    H = torch.stack([item["H"] for item in batch], dim=0)
    metas = [item["meta"] for item in batch]
    return {"H": H, "meta": metas}

def main():
    cfg = load_config_with_cli()
    run_dir = Path(cfg.get("run_dir", "runs/latest"))

    # Load config snapshot from run_dir if available
    snap_cfg_path = run_dir / "config.yaml"
    if snap_cfg_path.exists():
        run_cfg = OmegaConf.load(snap_cfg_path)
    else:
        run_cfg = cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    # Load backbone extractor for tokenizer
    backbone_cfg = run_cfg.get("backbone", {"type": "gpt2", "model_name_or_path": "gpt2"})
    extractor = BACKBONE_REGISTRY.build(backbone_cfg, device=device)

    # Load dataset module
    data_cfg = run_cfg.get("data", {"type": "multiplication", "digits": 4})
    dataset_mod = DATA_REGISTRY.build(data_cfg)

    # Build model
    d_model = run_cfg.get("engine", {}).get("d_model", 512)
    model = LSRRModel(
        adapter_cfg=run_cfg.get("adapter", {"type": "per_layer_affine+rmsnorm"}),
        engine_cfg=run_cfg.get("engine", {"type": "mamba_up"}),
        fusion_cfg=run_cfg.get("fusion", {"type": "attention_pooling"}),
        decoder_cfg=run_cfg.get("decoder", {"type": "trained_light_decoder"}),
        iteration_cfg=run_cfg.get("iteration", {}),
        termination_cfg=run_cfg.get("termination", {}),
        d_in=extractor.hidden_dim,
        num_layers=extractor.num_layers
    )

    ckpt_path = run_dir / "best_model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[Eval] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"[Eval] Warning: Checkpoint not found at {ckpt_path}. Evaluating uninitialized model.")

    # Prepare evaluation data
    from scripts.train import ensure_cached_data
    val_cache_dir, _, _ = ensure_cached_data(run_cfg, split="val", device=device)
    val_ds = ShardedHCacheDataset(val_cache_dir)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_h_cache)

    evaluator = Evaluator(
        model=model,
        dataset_mod=dataset_mod,
        tokenizer=extractor.tokenizer,
        device=device,
        engine_name=run_cfg.get("engine", {}).get("type", "mamba_up"),
        d_model=d_model,
        num_engine_layers=run_cfg.get("engine", {}).get("n_blocks", 2),
        num_model_layers=extractor.num_layers
    )

    # 1. Base evaluation
    base_res = evaluator.evaluate_rule(val_loader)
    print("\n--- Base Evaluation Result ---")
    for k, v in base_res.items():
        print(f"{k}: {v}")

    # 2. Sweep termination rules for Pareto curve
    sweep_rules = [
        {"type": "delta_state", "eps": 1e-2, "m_max": 32},
        {"type": "delta_state", "eps": 5e-3, "m_max": 32},
        {"type": "delta_state", "eps": 1e-3, "m_max": 32},
        {"type": "delta_state", "eps": 5e-4, "m_max": 32},
        {"type": "delta_state", "eps": 1e-4, "m_max": 32},
        {"type": "fixed_m", "m": 2, "m_max": 32},
        {"type": "fixed_m", "m": 4, "m_max": 32},
        {"type": "fixed_m", "m": 8, "m_max": 32},
    ]
    pareto_data = evaluator.sweep_termination_rules(val_loader, sweep_rules)

    # Save Pareto results
    out_pareto = run_dir / "pareto_sweep.json"
    with open(out_pareto, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2)
    print(f"\n[Eval] Pareto sweep saved to {out_pareto}")

if __name__ == "__main__":
    main()
