#!/usr/bin/env python
"""궤적 분화 진단 — ADR-015 의 선행 관문 (F-025).

    "M개 토큰이 서로 달라지는가"는 Ablation A 를 돌리기 **전에** 판정해야 한다.
    궤적이 분화하지 않으면 `동적 M ≡ 단일 벡터` 이고, v2 의 킬 스위치가
    과학적 이유가 아니라 구현상의 이유로 실패한다.

**정확도로 판정하지 않는다.** F-010 이 보인 대로 손실 하강·정확도는 주입 통로가
쓰이고 있다는 증거가 되지 못한다. 대신 **개입 비교**로 잰다.

## 재는 것

1. **토큰 분화** — 방출 토큰들이 서로 다른가.
   `쌍별 코사인`, `유효 랭크`(특이값 엔트로피). 랭크가 1에 가까우면 M개 토큰이
   사실상 하나다.
2. **토큰별 기여** — 각 토큰이 실제로 쓰이는가.
   토큰 m 하나를 영벡터/셔플로 바꿨을 때 답 NLL 이 얼마나 나빠지는가.
   전 토큰의 기여가 0 이면 백본이 궤적을 무시하고 있다는 뜻이다.
3. **궤적 길이 기여** — 앞쪽 k개만 주입했을 때의 NLL 곡선.
   길이를 늘려도 나아지지 않으면 동적 M 의 근거가 없다.

사용:
    python scripts/diagnose_trajectory.py exp=phase0_mult \\
        --anchors ctx,none,ramp,first --train-steps 600
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from lsrr.builder import build_slots
from lsrr.config import load_config
from lsrr.config.schema import get_path
from lsrr.data import PromptEncoder, PromptSpec
from lsrr.data.collate import make_loader
from lsrr.model import LSRRModel
from lsrr.objectives.targets import loss_targets, token_nll
from lsrr.runtime import resolve_device, set_seed


class _Collect:
    """사이클별 정제 상태를 모은다."""

    def __init__(self) -> None:
        self.states: list[torch.Tensor] = []

    def on_cycle(self, m: int, R_m: torch.Tensor, R_next: torch.Tensor, diagnostics: Any) -> None:
        self.states.append(R_next)


def _trajectory(model: LSRRModel, batch: dict, M: int):
    # 배치마다 인코딩 카운터를 리셋한다 (I2). 진단은 같은 배치를 여러 번 훑으므로
    # 리셋을 빠뜨리면 계약이 재인코딩으로 오인한다.
    model.encode_counter.reset()
    ctx = model.encode(batch["input_ids"], batch.get("attention_mask"))
    R0 = model.build_memory(ctx)
    col = _Collect()
    model.refine(R0, hooks=[col], is_eval=False, M=M)
    traj, alphas = model.readout.emit_trajectory(col.states, ctx.h_ctx)
    return ctx, traj, alphas


def _nll(model: LSRRModel, ctx, traj: torch.Tensor, batch: dict) -> float:
    logits = model.readout.answer_head.teacher_forced(
        traj, ctx.kv_cache, batch["target_ids"],
        attention_mask=ctx.attention_mask, q_len=ctx.q_len,
    )
    return float(token_nll(logits, loss_targets(batch)))


def differentiation(traj: torch.Tensor) -> dict[str, float]:
    """토큰 분화 지표. traj [B, M, d]."""
    T = F.normalize(traj.float(), dim=-1)
    G = T @ T.transpose(-1, -2)  # [B, M, M]
    M = G.shape[-1]
    if M < 2:
        return {"mean_cosine": 1.0, "min_cosine": 1.0, "effective_rank": 1.0}
    off = ~torch.eye(M, dtype=torch.bool, device=G.device)
    ranks = []
    for sample in traj.float():
        sv = torch.linalg.svdvals(sample)
        p = (sv / sv.sum().clamp(min=1e-12)).clamp(min=1e-12)
        ranks.append(float(torch.exp(-(p * p.log()).sum())))
    return {
        "mean_cosine": float(G[:, off].mean()),
        "min_cosine": float(G[:, off].min()),
        "effective_rank": sum(ranks) / len(ranks),
    }


@torch.no_grad()
def interventions(model: LSRRModel, batch: dict, M: int) -> dict[str, Any]:
    """개입 비교 — 토큰을 망가뜨렸을 때 NLL 이 얼마나 나빠지는가."""
    ctx, traj, _ = _trajectory(model, batch, M)
    base = _nll(model, ctx, traj, batch)

    per_token = []
    for m in range(M):
        damaged = traj.clone()
        damaged[:, m] = 0.0
        per_token.append(_nll(model, ctx, damaged, batch) - base)

    shuffled = traj[torch.randperm(traj.shape[0], device=traj.device)]
    prefix = [_nll(model, ctx, traj[:, : k + 1], batch) for k in range(M)]

    return {
        "nll": base,
        "per_token_delta": per_token,
        "shuffle_delta": _nll(model, ctx, shuffled, batch) - base,
        "zero_all_delta": _nll(model, ctx, torch.zeros_like(traj), batch) - base,
        "prefix_nll": prefix,
        **differentiation(traj),
    }


def train_briefly(model: LSRRModel, loader, device, steps: int, M: int, lr: float) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    last = float("nan")
    step = 0
    while step < steps:
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            ctx, traj, _ = _trajectory(model, batch, M)
            loss = torch.as_tensor(0.0, device=device) + _nll_t(model, ctx, traj, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last = float(loss)
            step += 1
            if step >= steps:
                break
    return last


def _nll_t(model: LSRRModel, ctx, traj: torch.Tensor, batch: dict) -> torch.Tensor:
    logits = model.readout.answer_head.teacher_forced(
        traj, ctx.kv_cache, batch["target_ids"],
        attention_mask=ctx.attention_mask, q_len=ctx.q_len,
    )
    return token_nll(logits, loss_targets(batch))


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--anchors", default="ctx,none,ramp,first")
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--json", type=Path)
    ap.add_argument("-h", "--help", action="store_true")
    args, rest = ap.parse_known_args(argv)
    if args.help:
        print(__doc__)
        return 0

    M = args.cycles
    results: dict[str, Any] = {}
    for anchor in [a.strip() for a in args.anchors.split(",") if a.strip()]:
        cfg = load_config([*rest, f"readout.fusion.anchor={anchor}"])
        set_seed(int(get_path(cfg, "seed", 42)), True)
        device = resolve_device(str(get_path(cfg, "device", "auto")))

        bundle = build_slots(cfg)
        model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner).to(device)
        spec = PromptSpec(
            max_question_tokens=int(get_path(cfg, "prompt.max_question_tokens", 256)),
            max_answer_tokens=int(get_path(cfg, "prompt.max_answer_tokens", 32)),
        )
        encoder = PromptEncoder(bundle.encoder.tokenizer, spec)
        loader = make_loader(
            bundle.data.get_split("train"), encoder,
            batch_size=args.batch_size, shuffle=True,
        )
        probe = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in next(iter(make_loader(
                     bundle.data.get_split("val")[:32], encoder,
                     batch_size=32, shuffle=False))).items()}

        before = interventions(model, probe, M)
        final_loss = train_briefly(model, loader, device, args.train_steps, M, args.lr)
        after = interventions(model, probe, M)
        results[anchor] = {"before": before, "after": after, "train_loss": final_loss}
        print(f"[{anchor}] 학습 {args.train_steps}스텝 → loss {final_loss:.4f}", flush=True)

    _report(results, M)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[진단] JSON 저장: {args.json}")
    return 0


def _report(results: dict[str, Any], M: int) -> None:
    print(f"\n═══ 궤적 분화 관문 (M={M}, 학습 후) ═══\n")
    print(f"{'anchor':>8} {'NLL':>8} {'유효랭크':>9} {'평균cos':>9} "
          f"{'셔플Δ':>9} {'전체영Δ':>9} {'토큰별Δ 합':>11}")
    for a, r in results.items():
        x = r["after"]
        print(f"{a:>8} {x['nll']:>8.4f} {x['effective_rank']:>9.2f} "
              f"{x['mean_cosine']:>9.4f} {x['shuffle_delta']:>9.4f} "
              f"{x['zero_all_delta']:>9.4f} {sum(x['per_token_delta']):>11.4f}")

    print(f"\n토큰별 기여 Δ NLL (해당 토큰만 영벡터로):")
    print(f"{'anchor':>8} " + " ".join(f"{'m'+str(i+1):>8}" for i in range(M)))
    for a, r in results.items():
        print(f"{a:>8} " + " ".join(f"{d:>8.4f}" for d in r["after"]["per_token_delta"]))

    print(f"\n접두 길이별 NLL (앞 k개만 주입):")
    print(f"{'anchor':>8} " + " ".join(f"{'k='+str(i+1):>8}" for i in range(M)))
    for a, r in results.items():
        print(f"{a:>8} " + " ".join(f"{v:>8.4f}" for v in r["after"]["prefix_nll"]))

    print(f"\n학습 전후 유효 랭크:")
    for a, r in results.items():
        print(f"  {a:>8}: {r['before']['effective_rank']:.2f} → {r['after']['effective_rank']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
