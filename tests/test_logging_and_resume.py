import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import pytest
from omegaconf import OmegaConf

from lsrr.model import LSRRModel
from lsrr.losses.composite import CompositeLoss
from lsrr.training.trainer import Trainer, compute_batch_accuracy
from lsrr.utils.logging import ExperimentTracker
from scripts.train import find_resume_checkpoint

def test_compute_batch_accuracy():
    # Batch size 2, seq_len 4
    preds = torch.tensor([
        [10, 20, 30, 40],
        [10, 25, 30, 40]
    ])
    targets = torch.tensor([
        [10, 20, 30, 0],
        [10, 20, 30, 40]
    ])
    target_lens = torch.tensor([3, 4])

    e_corr, b_tot, t_corr, t_tot = compute_batch_accuracy(preds, targets, target_lens)
    assert b_tot == 2
    # Sample 0: preds[:3] == [10, 20, 30] matches targets[:3] == [10, 20, 30] -> exact match!
    # Sample 1: preds[:4] has 25 vs 20 -> not exact match.
    assert e_corr == 1
    # Token matches: sample 0 has 3/3, sample 1 has 3/4 -> total 6/7
    assert t_corr == 6
    assert t_tot == 7

def test_epoch_logging_and_checkpoint_resume():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        seed = 42
        torch.manual_seed(seed)

        # Create synthetic data
        num_samples = 16
        H_data = torch.randn(num_samples, 6, 32)
        target_data = torch.randint(0, 50, (num_samples, 3))
        dataset = TensorDataset(H_data, target_data)

        def collate(batch):
            target_ids = torch.stack([x[1] for x in batch])
            target_lens = torch.tensor([x[1].size(0) for x in batch], dtype=torch.long)
            return {
                "H": torch.stack([x[0] for x in batch]),
                "target_ids": target_ids,
                "target_lens": target_lens,
                "meta": [{"id": i} for i in range(len(batch))]
            }

        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)

        # Stage 1: Train for 2 epochs
        cfg_stage1 = OmegaConf.create({
            "train": {"epochs": 2, "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0, "amp": False},
            "engine": {"type": "mamba_up"},
            "backbone": {"name": "test"},
            "data": {"name": "test"}
        })

        tracker1 = ExperimentTracker(exp_name="test_exp", cfg=cfg_stage1, base_runs_dir=tmp_path, seed=seed)
        model1 = LSRRModel(
            adapter_cfg={"type": "per_layer_affine+rmsnorm"},
            engine_cfg={"type": "mamba_up", "n_blocks": 1},
            fusion_cfg={"type": "attention_pooling"},
            decoder_cfg={"type": "trained_light_decoder", "vocab_size": 100, "n_layers": 1, "n_heads": 2},
            d_in=32,
            num_layers=6
        )
        loss_fn = CompositeLoss([{"type": "answer_nll", "w": 1.0}])

        trainer1 = Trainer(
            model=model1,
            train_loader=loader,
            val_loader=loader,
            loss_fn=loss_fn,
            cfg=cfg_stage1,
            tracker=tracker1,
            device=torch.device("cpu")
        )
        summary1 = trainer1.fit()

        run_dir = tracker1.run_dir
        # Verify history.csv and history.json
        csv_path = run_dir / "history.csv"
        json_path = run_dir / "history.json"
        assert csv_path.exists(), "history.csv was not created!"
        assert json_path.exists(), "history.json was not created!"

        df = pd.read_csv(csv_path)
        assert len(df) == 2, f"Expected 2 epochs in history.csv, got {len(df)}"
        for col in ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]:
            assert col in df.columns, f"Missing column '{col}' in history.csv"

        # Verify checkpoints
        assert (run_dir / "checkpoint_last.pt").exists(), "checkpoint_last.pt not found!"
        assert (run_dir / "best_model.pt").exists(), "best_model.pt not found!"

        # Stage 2: Resume training to 4 epochs
        cfg_stage2 = OmegaConf.create({
            "train": {"epochs": 4, "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0, "amp": False},
            "engine": {"type": "mamba_up"},
            "backbone": {"name": "test"},
            "data": {"name": "test"},
            "resume_from": str(run_dir / "checkpoint_last.pt")
        })

        ckpt_file, run_id = find_resume_checkpoint(cfg_stage2, "test_exp", base_runs_dir=tmp_path)
        assert ckpt_file == run_dir / "checkpoint_last.pt"
        assert run_id == tracker1.run_id

        resume_checkpoint = torch.load(ckpt_file, map_location="cpu")
        tracker2 = ExperimentTracker(exp_name="test_exp", cfg=cfg_stage2, base_runs_dir=tmp_path, seed=seed, run_id=run_id)

        model2 = LSRRModel(
            adapter_cfg={"type": "per_layer_affine+rmsnorm"},
            engine_cfg={"type": "mamba_up", "n_blocks": 1},
            fusion_cfg={"type": "attention_pooling"},
            decoder_cfg={"type": "trained_light_decoder", "vocab_size": 100, "n_layers": 1, "n_heads": 2},
            d_in=32,
            num_layers=6
        )

        trainer2 = Trainer(
            model=model2,
            train_loader=loader,
            val_loader=loader,
            loss_fn=loss_fn,
            cfg=cfg_stage2,
            tracker=tracker2,
            device=torch.device("cpu"),
            resume_checkpoint=resume_checkpoint
        )
        summary2 = trainer2.fit()

        # Check updated history
        df2 = pd.read_csv(csv_path)
        assert len(df2) == 4, f"Expected 4 epochs after resume, got {len(df2)}"
        assert list(df2["epoch"]) == [1, 2, 3, 4]
        print("\n[Test Passed] Accuracy logging and checkpoint resume verified successfully!")

def test_cosine_scheduler_with_warmup():
    from lsrr.training.trainer import get_cosine_schedule_with_warmup
    param = torch.nn.Parameter(torch.zeros(1))
    base_lr = 3e-4
    min_lr = 1e-5
    optimizer = torch.optim.AdamW([param], lr=base_lr)
    
    total_steps = 100
    warmup_steps = 10  # 1/10 of total steps (0-10%)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=min_lr
    )

    lrs = []
    for step in range(total_steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # Step 0: lr starts at 0
    assert lrs[0] == 0.0
    # Warmup is strictly increasing up to warmup_steps
    for s in range(1, warmup_steps):
        assert lrs[s] > lrs[s-1], f"Expected LR to increase during warmup at step {s}: {lrs[s]} vs {lrs[s-1]}"
    # At warmup_steps (step 10), reaches base_lr
    assert abs(lrs[warmup_steps] - base_lr) < 1e-9
    # Cosine decay phase is strictly decreasing
    for s in range(warmup_steps + 1, total_steps):
        assert lrs[s] < lrs[s-1], f"Expected LR to decrease during cosine decay at step {s}: {lrs[s]} vs {lrs[s-1]}"
    # At final step (last iteration of last epoch, index 99 of 100 steps), LR is EXACTLY min_lr (1e-5)
    assert abs(lrs[-1] - min_lr) < 1e-9, f"Expected final step LR to be {min_lr}, got {lrs[-1]}"
    print("\n[Test Passed] Mamba-2 recipe: Warmup (1/10) + half-cosine decay to 1e-5 at last step verified!")

