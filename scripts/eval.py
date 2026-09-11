#!/usr/bin/env python
"""평가 — Phase 0 게이트의 판정 데이터를 만든다 (M5).

한 번의 평가에서 네 게이트가 쓸 산출을 함께 낸다.

    <run>/metrics.json      accuracy, num_samples, mean_cycles     ← 게이트 ③
    <run>/gate_inputs.json  collapse, anytime, delta_trajectories  ← 게이트 ①②④

**게이트 ②는 사이클별로 생성해서 잰다.** 게이트 ③이 생성 기반 exact match이므로
②를 teacher-forcing 토큰 정확도로 재면 두 게이트가 서로 다른 것을 말하게 된다.
비용이 M배라 기본은 배치 1개만 잰다 (`--anytime-batches`).

사용:
    python scripts/eval.py exp=phase0_mult --run runs/<run_id>
    python scripts/eval.py exp=phase0_mult --run runs/<run_id> --split test
    python scripts/eval.py exp=phase0_mult --run runs/<run_id> --limit 200

평가가 끝나면 판정은 별개다:
    python scripts/check_gates.py --run runs/<run_id> --runs-dir runs \\
        --hydra-glob '...' --mlp-glob '...'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lsrr.builder import build_slots
from lsrr.config import load_config
from lsrr.config.schema import get_path
from lsrr.data import PromptEncoder, prompt_spec_from_cfg
from lsrr.data.collate import make_loader
from lsrr.metrics.evaluate import evaluate, scorer_for
from lsrr.model import LSRRModel
from lsrr.runtime import resolve_device, set_seed


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run", type=Path, help="산출을 쓸 런 디렉터리")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None, help="평가 샘플 수 상한")
    ap.add_argument("--anytime-batches", type=int, default=1)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("-h", "--help", action="store_true")
    args, rest = ap.parse_known_args(argv)
    if args.help:
        print(__doc__)
        return 0

    cfg = load_config(rest)
    seed = int(get_path(cfg, "seed", 42))
    set_seed(seed, bool(get_path(cfg, "deterministic", True)))
    device = resolve_device(str(get_path(cfg, "device", "auto")))

    bundle = build_slots(cfg)
    model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner).to(device)

    if args.checkpoint is not None:
        from lsrr.runtime import load_checkpoint

        load_checkpoint(model, args.checkpoint)
        print(f"[평가] 체크포인트: {args.checkpoint}")
    else:
        print("[평가] 경고: 체크포인트 없이 초기 가중치를 평가한다.")

    spec = prompt_spec_from_cfg(cfg)
    encoder = PromptEncoder(bundle.encoder.tokenizer, spec)
    samples = bundle.data.get_split(args.split)
    if args.limit:
        samples = samples[: args.limit]

    loader = make_loader(
        samples, encoder,
        batch_size=int(get_path(cfg, "eval.bs", get_path(cfg, "train.bs", 16))),
        shuffle=False,
    )
    print(f"[평가] split={args.split} 샘플 {len(samples)}개, 장치 {device}")

    tokenizer = bundle.encoder.tokenizer

    def decode(tokens: torch.Tensor) -> list[str]:
        return [
            tokenizer.decode(row, skip_special_tokens=True).strip() for row in tokens
        ]

    def to_device(batch: dict) -> dict:
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    result = evaluate(
        model,
        (to_device(b) for b in loader),
        decode=decode,
        scorer=scorer_for(str(get_path(cfg, "data.type", "multiplication"))),
        anytime_batches=args.anytime_batches,
    )

    print(f"[평가] 정확도 {result.accuracy:.4f} ({result.num_samples}개), "
          f"평균 사이클 {result.mean_cycles:.2f}")
    if result.anytime:
        curve = ", ".join(f"m{m}={a:.3f}" for m, a in sorted(result.anytime.items()))
        print(f"[평가] anytime: {curve}")
    if result.collapse is not None:
        c = result.collapse
        print(f"[평가] 붕괴 진단: 랭크 {c.effective_rank:.2f}, "
              f"유사도 {c.mean_similarity:.3f}")

    if args.run is not None:
        args.run.mkdir(parents=True, exist_ok=True)
        (args.run / "metrics.json").write_text(
            json.dumps(result.metrics_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        gate_inputs = result.gate_inputs_payload()
        (args.run / "gate_inputs.json").write_text(
            json.dumps(gate_inputs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        missing = [
            k for k in ("collapse", "anytime", "delta_trajectories")
            if k not in gate_inputs
        ]
        print(f"[평가] 저장: {args.run}/metrics.json, gate_inputs.json")
        if missing:
            print(f"[평가] 주의: {', '.join(missing)} 가 비어 게이트가 판정 불가다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
