#!/usr/bin/env python
"""주입 통로 진단 — M3의 첫 관문 (ADR-013, FINDINGS F-006).

    "M3의 첫 관문은 손실 하강 여부가 아니라 주입 통로가 실제로 학습되는지다."

teacher-forcing에서는 주입을 **전혀 쓰지 않고도** 답 토큰의 통계(자릿수·형식)만
학습해 손실이 크게 내려갈 수 있다. 따라서 손실 하강은 증거가 되지 못한다.

이 스크립트는 세 가지를 분리해 보고한다:

1. **개입 비교** — `h_fusion`을 셔플/평균/영벡터로 바꿨을 때 NLL이 얼마나
   나빠지는가. 셔플 대비 이득이 곧 **질문별 정보의 가치**다.
2. **단계별 질문 의존성** — 어느 단계에서 질문 정보가 사라지는가.
3. **융합 항 분해** — 앵커 `h_ctx`와 사고 항 `W_r·h_SSM` 중 무엇이 질문별
   변동을 만드는가.

사용:
    python scripts/diagnose_injection.py runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from lsrr.builder import build_slots
from lsrr.config import load_snapshot
from lsrr.config.schema import get_path
from lsrr.core.invariants import IGNORE_INDEX
from lsrr.data import PromptEncoder, PromptSpec
from lsrr.data.collate import make_loader
from lsrr.model import LSRRModel
from lsrr.runtime import load_checkpoint


def _load(run_dir: Path, checkpoint: str | None):
    cfg = load_snapshot(run_dir / "config.yaml")
    bundle = build_slots(cfg)
    model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)
    ckpts = sorted((run_dir / "checkpoints").glob("*.pt"))
    path = Path(checkpoint) if checkpoint else (ckpts[-1] if ckpts else None)
    if path is not None:
        load_checkpoint(model, path)
    model.to(bundle.device).eval()
    return cfg, bundle, model, path


def _batches(cfg, bundle, split: str, limit: int, batch_size: int):
    spec = PromptSpec(
        max_question_tokens=int(get_path(cfg, "prompt.max_question_tokens", 256)),
        max_answer_tokens=int(get_path(cfg, "prompt.max_answer_tokens", 32)),
    )
    encoder = PromptEncoder(bundle.encoder.tokenizer, spec)
    samples = bundle.data.get_split(split)[:limit]
    loader = make_loader(samples, encoder, batch_size=batch_size)
    return encoder, [
        {k: (v.to(bundle.device) if torch.is_tensor(v) else v) for k, v in b.items()}
        for b in loader
    ]


def _forward(model, batch):
    """인코딩 → 메모리 → 정제 → 융합까지. 주입 직전 상태를 돌려준다."""
    model.encode_counter.reset()  # 진단은 배치마다 새로 인코딩한다 (I2)
    ctx = model.encode(batch["input_ids"], batch.get("attention_mask"))
    R0 = model.build_memory(ctx)
    trace = model.refine(R0, is_eval=False)
    h_fusion, alpha = model.readout.fuse(trace.R_star, ctx.h_ctx)
    return ctx, R0, trace, h_fusion, alpha


@torch.no_grad()
def intervention_nll(model, batches, transform, first_token_only=False) -> float:
    """주입 벡터를 바꿔가며 답 NLL을 잰다."""
    total, count = 0.0, 0
    for b in batches:
        ctx, _, trace, h_fusion, _ = _forward(model, b)
        h = transform(h_fusion, ctx, model)
        logits = model.readout.answer_head.teacher_forced(
            h, ctx.kv_cache, b["target_ids"],
            attention_mask=ctx.attention_mask, q_len=ctx.q_len,
        )
        labels = b["labels"]
        if first_token_only:
            logits, labels = logits[:, :1], labels[:, :1]
        total += float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
        )
        count += int((labels != IGNORE_INDEX).sum())
    return total / max(count, 1)


def dependence_stats(x: torch.Tensor) -> dict[str, float]:
    """샘플 간 유사도와 변동계수. 코사인 1.0 = 질문과 무관 = 붕괴."""
    xc = x.reshape(x.shape[0], -1).float()
    norm = float(xc.norm(dim=-1).mean())
    deviation = float((xc - xc.mean(0, keepdim=True)).norm(dim=-1).mean())
    cos = F.cosine_similarity(xc.unsqueeze(1), xc.unsqueeze(0), dim=-1)
    off = cos[~torch.eye(len(xc), dtype=torch.bool, device=cos.device)]
    return {
        "norm": norm,
        "deviation": deviation,
        "cosine": float(off.mean()),
        "cv": deviation / max(norm, 1e-9),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg, bundle, model, ckpt = _load(args.run_dir, args.checkpoint)
    encoder, batches = _batches(cfg, bundle, args.split, args.limit, args.batch_size)
    print(f"런: {args.run_dir}  체크포인트: {ckpt}")
    print(f"엔진: {get_path(cfg, 'engine.type')}  d_model: {model.d_model}\n")

    gen = torch.Generator(device=bundle.device).manual_seed(0)
    conditions = {
        "학습된 h_fusion": lambda h, c, m: h,
        "배치 셔플": lambda h, c, m: h[
            torch.randperm(h.shape[0], device=h.device, generator=gen)
        ],
        "배치 평균": lambda h, c, m: h.mean(0, keepdim=True).expand_as(h),
        "h_ctx만": lambda h, c, m: m.readout.calibrator(c.h_ctx),
        "영벡터": lambda h, c, m: torch.zeros_like(h),
    }

    print("① 개입 비교 (셔플 대비 이득 = 질문별 정보의 가치)")
    print(f"   {'조건':<18}{'전체 NLL':>11}{'첫토큰 NLL':>13}")
    nlls: dict[str, tuple[float, float]] = {}
    for name, fn in conditions.items():
        nlls[name] = (
            intervention_nll(model, batches, fn),
            intervention_nll(model, batches, fn, first_token_only=True),
        )
        print(f"   {name:<18}{nlls[name][0]:>11.4f}{nlls[name][1]:>13.4f}")

    trained, shuffled = nlls["학습된 h_fusion"], nlls["배치 셔플"]
    gain_all = shuffled[0] - trained[0]
    gain_first = shuffled[1] - trained[1]
    print(f"\n   질문별 정보의 가치: 전체 {gain_all:+.4f} nats | 첫토큰 {gain_first:+.4f} nats")

    print("\n② 단계별 질문 의존성 (코사인 1.0 = 질문과 무관)")
    with torch.no_grad():
        ctx, R0, trace, h_fusion, alpha = _forward(model, batches[0])
    stages = {
        "h_ctx (백본 앵커)": ctx.h_ctx,
        "H_last (레이어 축)": ctx.H_last,
        "H_pool (질문 풀링)": ctx.H_pool,
        "R0 (어댑터 후)": R0,
        "R* (정제 후)": trace.R_star,
        "h_fusion (주입)": h_fusion,
    }
    stage_stats = {}
    print(f"   {'단계':<20}{'노름':>10}{'변동':>10}{'코사인':>9}{'변동계수':>10}")
    for name, tensor in stages.items():
        if tensor is None:
            continue
        s = dependence_stats(tensor)
        stage_stats[name] = s
        print(
            f"   {name:<20}{s['norm']:>10.3f}{s['deviation']:>10.3f}"
            f"{s['cosine']:>9.4f}{s['cv']:>10.4f}"
        )

    print("\n③ 융합 항 분해:  h_fusion = h_ctx + W_r·h_SSM")
    with torch.no_grad():
        a = F.softmax(model.fusion.score(trace.R_star).squeeze(-1), dim=-1)
        h_ssm = torch.einsum("bl,bld->bd", a, trace.R_star)
        thought = model.fusion.w_r(h_ssm)
    anchor_dev = dependence_stats(ctx.h_ctx)["deviation"]
    thought_dev = dependence_stats(thought)["deviation"]
    print(f"   ‖h_ctx‖ = {ctx.h_ctx.norm(dim=-1).mean():.3f}   "
          f"‖W_r·h_SSM‖ = {thought.norm(dim=-1).mean():.3f}")
    print(f"   질문별 변동:  앵커 {anchor_dev:.3f}  vs  사고 항 {thought_dev:.3f} "
          f"({thought_dev / max(anchor_dev, 1e-9):.1f}배)")
    print(f"   α 레이어 분포: {[round(x, 3) for x in a.mean(0).tolist()]}")

    q_last = batches[0]["input_ids"][:, -1]
    print(f"\n   질문 마지막 토큰 고유값: {q_last.unique().numel()} / {len(q_last)}"
          f"  {encoder.tokenizer.convert_ids_to_tokens(q_last.unique().tolist())[:5]}")

    verdict = "통과" if gain_first > 0.5 else ("미달" if gain_first < 0.1 else "경계")
    print(f"\n판정: 주입 통로 질문별 학습 = {verdict} (첫토큰 이득 {gain_first:+.4f} nats)")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "run_dir": str(args.run_dir),
                    "engine": get_path(cfg, "engine.type"),
                    "nll": {k: list(v) for k, v in nlls.items()},
                    "gain_all": gain_all,
                    "gain_first": gain_first,
                    "stages": stage_stats,
                    "anchor_deviation": anchor_dev,
                    "thought_deviation": thought_dev,
                    "alpha": a.mean(0).tolist(),
                    "verdict": verdict,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"JSON 저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
