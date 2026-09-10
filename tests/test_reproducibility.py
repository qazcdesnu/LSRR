import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest
from lsrr.utils.seed import set_seed
from lsrr.model import LSRRModel
from lsrr.losses.composite import CompositeLoss
from lsrr.training.trainer import Trainer
from lsrr.utils.logging import ExperimentTracker
from omegaconf import OmegaConf

def run_experiment(seed: int, tmp_path: Path):
    set_seed(seed, deterministic=True)

    # Synthetic dataset
    torch.manual_seed(seed)
    H_data = torch.randn(20, 8, 32)
    target_data = torch.randint(0, 100, (20, 4))
    dataset = TensorDataset(H_data, target_data)

    def collate(batch):
        return {
            "H": torch.stack([x[0] for x in batch]),
            "target_ids": torch.stack([x[1] for x in batch])
        }

    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)

    cfg = OmegaConf.create({
        "train": {"epochs": 2, "lr": 1e-3, "weight_decay": 0.0, "grad_clip": 1.0, "amp": False},
        "engine": {"type": "mamba_up"},
        "backbone": {"name": "test"},
        "data": {"name": "test"}
    })

    tracker = ExperimentTracker(exp_name="repro_test", cfg=cfg, base_runs_dir=tmp_path, seed=seed)

    model = LSRRModel(
        adapter_cfg={"type": "per_layer_affine+rmsnorm"},
        engine_cfg={"type": "mamba_up", "n_blocks": 1},
        fusion_cfg={"type": "attention_pooling"},
        decoder_cfg={"type": "trained_light_decoder", "vocab_size": 100, "n_layers": 1, "n_heads": 2},
        d_in=32,
        num_layers=8
    )

    loss_fn = CompositeLoss([{"type": "answer_nll", "w": 1.0}])

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        loss_fn=loss_fn,
        cfg=cfg,
        tracker=tracker,
        device=torch.device("cpu")
    )
    summary = trainer.fit()
    return summary, model.state_dict()

def test_reproducibility():
    """Test 5: Two runs with identical seed and deterministic settings yield exact matching losses."""
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        sum1, state1 = run_experiment(seed=123, tmp_path=Path(tmp1))
        sum2, state2 = run_experiment(seed=123, tmp_path=Path(tmp2))

        loss1 = sum1["final_epoch_metrics"]["train_loss"]
        loss2 = sum2["final_epoch_metrics"]["train_loss"]

        print(f"\n[Test 5] Run 1 Train Loss: {loss1:.6f}, Run 2 Train Loss: {loss2:.6f}")
        assert abs(loss1 - loss2) < 1e-6, f"Losses diverged! {loss1} vs {loss2}"

        # Check parameter state equality
        for k in state1:
            diff = (state1[k] - state2[k]).abs().max().item()
            assert diff < 1e-6, f"Parameter {k} diverged: {diff}"

        print("[Test 5 Passed] Deterministic reproducibility verified bit-exact.")
