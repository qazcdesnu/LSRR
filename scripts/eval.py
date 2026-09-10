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
    """Keep target_ids: the evaluator needs the gold answer for every sample.

    Older caches predate `meta["answer"]`, so target_ids is the fallback the scorer
    falls back to. Dropping it here is what made evaluation score against "".
    """
    H = torch.stack([item["H"] for item in batch], dim=0)

    target_lens = torch.tensor([item["target_ids"].size(0) for item in batch], dtype=torch.long)
    max_target_len = max((item["target_ids"].size(0) for item in batch), default=0)
    target_ids = torch.zeros((len(batch), max_target_len), dtype=torch.long)
    for i, item in enumerate(batch):
        t_len = item["target_ids"].size(0)
        if t_len > 0:
            target_ids[i, :t_len] = item["target_ids"]

    metas = [item["meta"] for item in batch]
    return {"H": H, "target_ids": target_ids, "target_lens": target_lens, "meta": metas}


def main():
    cfg = load_config_with_cli()
    run_dir = Path(cfg.get("run_dir", "runs/latest"))

    # Load config snapshot from run_dir if available
    snap_cfg_path = run_dir / "config.yaml"
    if snap_cfg_path.exists():
        run_cfg = OmegaConf.load(snap_cfg_path)
    else:
        run_cfg = cfg

    # Held-out split. The baselines this work compares against report test-set accuracy,
    # so `test` is the default; override with `eval.split=val` for development runs.
    split = cfg.get("eval", {}).get("split", None) or run_cfg.get("eval", {}).get("split", "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device} | split: {split}")

    # Load backbone extractor for tokenizer
    backbone_cfg = run_cfg.get("backbone", {"type": "gpt2", "model_name_or_path": "gpt2"})
    extractor = BACKBONE_REGISTRY.build(backbone_cfg, device=device)

    # Load dataset module
    data_cfg = run_cfg.get("data", {"type": "multiplication", "digits": 4})
    dataset_mod = DATA_REGISTRY.build(data_cfg)
    eval_samples = dataset_mod.get_split(split)
    print(f"[Eval] Split '{split}' has {len(eval_samples)} samples.")

    # Build model. The reasoning width is resolved inside LSRRModel from the backbone
    # hidden size, so read it back off the model rather than guessing from the config.
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
    eval_cache_dir, _, _ = ensure_cached_data(run_cfg, split=split, device=device)
    eval_ds = ShardedHCacheDataset(eval_cache_dir)
    eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=False, collate_fn=collate_h_cache)

    evaluator = Evaluator(
        model=model,
        dataset_mod=dataset_mod,
        tokenizer=extractor.tokenizer,
        device=device,
        engine_name=run_cfg.get("engine", {}).get("type", "mamba_up"),
        d_model=model.d_model,
        num_engine_layers=run_cfg.get("engine", {}).get("n_blocks", 2),
        num_model_layers=extractor.num_layers,
        # Supplying these lets the evaluator measure and charge the one backbone forward
        # pass, which is the cost the efficiency claim is about.
        extractor=extractor,
        eval_samples=eval_samples,
    )

    print("\n[Eval] Measuring frozen-backbone forward cost...")
    backbone_cost = evaluator.measure_backbone_cost()
    for k, v in backbone_cost.items():
        print(f"  {k}: {v}")

    # 1. Base evaluation
    base_res = evaluator.evaluate_rule(eval_loader)
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
    pareto_data = evaluator.sweep_termination_rules(eval_loader, sweep_rules)

    # Save Pareto results
    out_pareto = run_dir / "pareto_sweep.json"
    with open(out_pareto, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2)
    print(f"\n[Eval] Pareto sweep saved to {out_pareto}")

    # Save headline results so make_tables.py can report accuracy, not just val loss.
    out_results = run_dir / "eval_results.json"
    with open(out_results, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_dir.name,
            "split": split,
            "num_samples": len(eval_samples),
            "dataset": run_cfg.get("data", {}).get("name", "unknown"),
            "engine": run_cfg.get("engine", {}).get("type", "unknown"),
            "backbone": run_cfg.get("backbone", {}).get("name", "unknown"),
            "backbone_cost": backbone_cost,
            "base": base_res,
        }, f, indent=2)
    print(f"[Eval] Results saved to {out_results}")


if __name__ == "__main__":
    main()
