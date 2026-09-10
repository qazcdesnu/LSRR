"""M3 완료 조건 — 최소 학습 루프가 끝까지 돈다.

실제 GPT-2를 쓴다. 조립 → 인코딩 → 메모리 → 정제 → 판독 → 손실 → 역전파의
전 경로가 실제로 이어지는지는 더미로는 확인할 수 없기 때문이다.
"""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

from lsrr.builder import build_slots
from lsrr.core.invariants import assert_no_grad
from lsrr.data import PromptEncoder, PromptSpec
from lsrr.data.collate import make_loader
from lsrr.model import LSRRModel
from lsrr.runtime import Trainer, load_checkpoint, parameter_summary, set_seed
from lsrr.telemetry import ExperimentTracker


@pytest.fixture(scope="module")
def tiny_cfg():
    """가장 작은 실행 가능 설정 — CPU에서 몇 초에 끝난다."""
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "backbone": {
                "type": "hf_frozen_causal",
                "model_name_or_path": "gpt2",
                "pooler": {"type": "attn_pool"},
            },
            "memory": {
                "composer": {"type": "gate"},
                "scope": {"type": "late_band"},
                "adapter": {"type": "per_layer_affine", "d_model": 32},
            },
            "engine": {"type": "mlp_onepass", "expansion": 1, "n_blocks": 1},
            "recurrence": {
                "schedule": {"type": "fixed", "k": 2},
                "tbptt_k": 2,
                "hooks": {"readout_per_cycle": False},
            },
            "termination": {"type": "fixed_m", "m": 2, "m_max": 4},
            "readout": {
                "fusion": {"type": "attention_pooling", "fusion_type": "residual"},
                "path": {"type": "backbone_continuation"},
            },
            "objective": {"answer_nll": {"enabled": True, "w": 1.0}},
            "data": {
                "type": "multiplication",
                "digits": 2,
                "sizes": {"train": 16, "val": 8, "test": 8},
            },
            "prompt": {"max_question_tokens": 16, "max_answer_tokens": 8},
            "train": {"epochs": 1, "bs": 4, "lr": 1e-3, "log_every": 1},
        }
    )


@pytest.fixture(scope="module")
def trained(tiny_cfg, tmp_path_factory):
    set_seed(0, deterministic=False)
    bundle = build_slots(tiny_cfg)
    model = LSRRModel(bundle=bundle, cfg=tiny_cfg, runner=bundle.runner)
    encoder = PromptEncoder(
        bundle.encoder.tokenizer, PromptSpec(max_question_tokens=16, max_answer_tokens=8)
    )
    loader = make_loader(bundle.data.get_split("train"), encoder, batch_size=4)

    root = tmp_path_factory.mktemp("runs")
    tracker = ExperimentTracker(tiny_cfg, exp_name="tiny", seed=0, root=root)
    trainer = Trainer(model, bundle.objective, tracker, tiny_cfg, device=torch.device("cpu"))
    result = trainer.fit(loader)
    tracker.close()
    return {"bundle": bundle, "model": model, "tracker": tracker, "result": result,
            "cfg": tiny_cfg}


def test_training_completes(trained):
    assert trained["result"]["steps"] == 4  # 16 샘플 / bs 4


def test_loss_is_finite_and_recorded(trained):
    rows = [
        json.loads(line)
        for line in (trained["tracker"].dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert rows, "지표가 기록되지 않았다"
    assert all(torch.isfinite(torch.tensor(r["loss/total"])) for r in rows)


def test_run_directory_layout(trained):
    d = trained["tracker"].dir
    for name in ("config.yaml", "meta.json", "metrics.jsonl", "diagnostics.jsonl"):
        assert (d / name).exists(), f"{name}이 없다"
    assert (d / "checkpoints" / "epoch_0.pt").exists()


def test_config_snapshot_round_trips(trained):
    """스냅샷 해시가 런 신원이다 — 다시 읽어 같은 모델을 조립할 수 있어야 한다."""
    from lsrr.config import config_hash, load_snapshot

    snap = load_snapshot(trained["tracker"].dir / "config.yaml")
    assert config_hash(snap) == config_hash(trained["cfg"])


def test_diagnostics_capture_delta_trajectory(trained):
    """게이트 ④(Δ 궤적)의 원재료가 남아야 한다."""
    rows = [
        json.loads(line)
        for line in (trained["tracker"].dir / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert rows and len(rows[-1]["deltas"]) == 2
    assert "state_norm" in rows[-1]


def test_trainable_budget_within_proposal_claim(trained):
    """제안서 §5: 학습 대상은 백본 대비 약 3% 이내."""
    params = parameter_summary(trained["model"])
    ratio = params["total"] / trained["bundle"].encoder.num_parameters()
    assert ratio < 0.03, f"학습 파라미터 비율 {ratio:.1%}가 3%를 넘는다"


def test_backbone_untouched_after_training(trained):
    """I1·I4 — 실제 학습 루프를 돈 뒤에도 백본은 그대로다."""
    encoder = trained["bundle"].encoder
    encoder.verify_frozen()
    assert_no_grad(encoder.model, what="backbone")


def test_checkpoint_excludes_backbone(trained):
    payload = torch.load(
        trained["tracker"].dir / "checkpoints" / "epoch_0.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert not any(k.startswith("_encoder") for k in payload["state_dict"])
    assert payload["backbone"]["id"] == "gpt2"
    assert payload["backbone"]["weight_hash"] is not None


def test_checkpoint_round_trip(trained):
    """재개: 저장한 파라미터를 새 모델에 적재해 같은 출력을 낸다."""
    cfg = trained["cfg"]
    bundle = build_slots(cfg)
    fresh = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)
    load_checkpoint(fresh, trained["tracker"].dir / "checkpoints" / "epoch_0.pt")

    for (n1, p1), (n2, p2) in zip(
        trained["model"].named_parameters(), fresh.named_parameters()
    ):
        assert n1 == n2 and torch.allclose(p1, p2)


def test_scope_narrows_engine_input(trained):
    """late_band 설정이 실제로 레이어 축을 줄였다."""
    assert trained["bundle"].widths["scoped_layers"] == 4  # GPT-2 12층의 후반 1/3


def test_single_encode_per_step(trained):
    """I2 — 학습 스텝마다 백본 인코딩은 1회다."""
    assert trained["model"].encode_counter.count == 1
