#!/usr/bin/env python
"""학습 진입점.

    python scripts/train.py exp=phase0_mult train.lr=5e-4
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lsrr.builder import build_slots
from lsrr.config import load_config
from lsrr.config.schema import get_path
from lsrr.data import PromptEncoder, PromptSpec
from lsrr.data.collate import make_loader
from lsrr.model import LSRRModel
from lsrr.runtime import Trainer, parameter_summary, resolve_device, set_seed
from lsrr.telemetry import ExperimentTracker


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = load_config(argv)

    exp_name = next(
        (a.split("=", 1)[1] for a in argv if a.startswith(("exp=", "config="))), "exp"
    )
    seed = int(get_path(cfg, "seed", 42))
    generator = set_seed(seed, bool(get_path(cfg, "deterministic", True)))
    device = resolve_device(str(get_path(cfg, "device", "auto")))

    bundle = build_slots(cfg)
    model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)

    spec = PromptSpec(
        max_question_tokens=int(get_path(cfg, "prompt.max_question_tokens", 256)),
        max_answer_tokens=int(get_path(cfg, "prompt.max_answer_tokens", 32)),
    )
    encoder = PromptEncoder(bundle.encoder.tokenizer, spec)
    loader = make_loader(
        bundle.data.get_split("train"),
        encoder,
        batch_size=int(get_path(cfg, "train.bs", 16)),
        shuffle=True,
        generator=generator,
    )

    with ExperimentTracker(cfg, exp_name=exp_name, seed=seed) as tracker:
        params = parameter_summary(model)
        backbone_n = bundle.encoder.num_parameters()
        tracker.update_meta(
            backbone_params=backbone_n,
            trainable_params=params["total"],
            trainable_ratio=params["total"] / backbone_n,
            trainable_by_module=params["by_module"],
            backbone_weight_hash=bundle.encoder.weight_hash(),
            batch_size=int(get_path(cfg, "train.bs", 16)),
            device=str(device),
        )
        print(
            f"학습 파라미터 {params['total']:,} / 백본 {backbone_n:,} "
            f"= {params['total'] / backbone_n:.1%}"
        )
        print(f"모듈별: {params['by_module']}")

        trainer = Trainer(model, bundle.objective, tracker, cfg, device=device)
        result = trainer.fit(loader)
        print(f"완료: {result['steps']} 스텝, {result['seconds']:.1f}초 → {tracker.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
