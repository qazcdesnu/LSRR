#!/usr/bin/env python
"""정제 상태의 스케일 진단 — §4.3 의 전제를 확인한다 (FINDINGS F-028).

§4.3 의 종료 조건 `Δ⁽ᵐ⁾ < ε` 은 **고정 ε** 을 쓴다. 그러려면 상태 스케일이
학습 내내 비교 가능해야 하는데, 손실은 `R` 의 스케일에 무관하다 — 판독이
`h_fusion` 을 RMS 로 보정하므로 `R` 을 상수배 해도 주입 벡터가 같다. 즉
‖R‖ 을 묶어 두는 기울기 압력이 없고, 실제로 250스텝 만에 10¹⁴ 배로 간다.

같은 폭주가 §4.4 의 층 풀링도 죽인다. `β = softmax(w_pool^T r_l)` 의 입력이
커지면 β 가 one-hot 으로 굳어 "레이어 축을 따라 스캔한 결과를 가중 합한다" 가
"레이어 하나를 고른다" 가 된다.

손실 하강은 이 둘 중 무엇도 잡아내지 못한다. 그래서 따로 잰다.

사용:
    python scripts/diagnose_state_scale.py --run runs/<run_id>   # 학습 후
    python scripts/diagnose_state_scale.py --exp prosqa_smoke    # 초기화 직후
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from omegaconf import OmegaConf

from lsrr.builder import build_slots
from lsrr.config import load_config, load_snapshot
from lsrr.data import PromptEncoder, prompt_spec_from_cfg
from lsrr.data.collate import make_loader
from lsrr.model import LSRRModel
from lsrr.runtime import load_checkpoint, set_seed


def _scale(tensor: torch.Tensor) -> dict[str, float]:
    t = tensor.detach().float()
    return {"absmax": float(t.abs().max()), "rms": float(t.pow(2).mean().sqrt())}


def _line(name: str, tensor: torch.Tensor) -> None:
    s = _scale(tensor)
    print(f"  {name:26s} shape={tuple(tensor.shape)}  "
          f"absmax={s['absmax']:.4g}  rms={s['rms']:.4g}")


def _lora_cfg(cfg) -> Optional[dict]:
    """설정의 `backbone.lora` 절. Phase B 체크포인트 적재에 필요하다 (ADR-014)."""
    from omegaconf import OmegaConf

    node = get_path(cfg, "backbone.lora", None)
    return OmegaConf.to_container(node, resolve=True) if node is not None else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--run", type=Path, default=None, help="런 디렉터리 (학습 후)")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None, help="JSON 저장 경로")
    ap.add_argument("-h", "--help", action="store_true")
    args, rest = ap.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    if args.help:
        print(__doc__)
        return 0

    if args.run is not None:
        cfg = load_snapshot(args.run / "config.yaml")
        if rest:
            # 스냅샷 위에 덮어쓴다 — Δ 궤적 전체를 보려면 종료를 꺼야 한다
            # (`termination.m_min=8 termination.eps=0.0`).
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(rest))
    else:
        cfg = load_config(rest)

    # 초기화 직후를 잴 때 난수 상태가 스크립트마다 다르면 같은 설정이 다른
    # Δ 를 낸다 — 초기 Δ 는 무작위 초기화에 민감하다.
    set_seed(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", True)))

    bundle = build_slots(cfg)
    model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)

    ckpt = args.checkpoint
    if ckpt is None and args.run is not None:
        candidate = args.run / "checkpoints" / "last.pt"
        ckpt = candidate if candidate.exists() else None
    if ckpt is not None:
        load_checkpoint(model, ckpt, lora_cfg=_lora_cfg(cfg))
        print(f"체크포인트: {ckpt}")
    else:
        print("체크포인트 없음 — 초기화 직후 상태를 잰다")
    model.to(bundle.device).eval()

    encoder = PromptEncoder(bundle.encoder.tokenizer, prompt_spec_from_cfg(cfg))
    samples = bundle.data.get_split(args.split)[: args.bs]
    batch = next(iter(make_loader(samples, encoder, batch_size=args.bs)))
    batch = {k: (v.to(bundle.device) if torch.is_tensor(v) else v)
             for k, v in batch.items()}

    report: dict[str, object] = {"checkpoint": str(ckpt) if ckpt else None}

    with torch.no_grad():
        model.encode_counter.reset()
        ctx = model.encode(batch["input_ids"], batch.get("attention_mask"))
        ctx = model.pool_context(ctx)
        print("\n[단계별 크기]")
        _line("H_last", ctx.H_last)
        _line("h_ctx", ctx.h_ctx)

        R0 = model.build_memory(ctx)
        _line("R0 (어댑터 출력)", R0)
        report["R0"] = _scale(R0)

        trace = model.refine(R0, is_eval=True)
        _line("R* (정제 후)", trace.R_star)
        report["R_star"] = _scale(trace.R_star)
        report["growth"] = report["R_star"]["rms"] / max(report["R0"]["rms"], 1e-12)

        print("\n[Δ 궤적]  §4.3 의 ε 은 이 값과 같은 자릿수여야 한다")
        eps = float(cfg.termination.get("eps", 0.05)) if "termination" in cfg else 0.05
        deltas = []
        for m, d in enumerate(trace.per_cycle):
            value = getattr(d, "delta_state", None)
            value = float(value.mean()) if torch.is_tensor(value) else value
            deltas.append(value)
            mark = "" if value is None else ("  < ε" if value < eps else "")
            print(f"  m={m}: Δ={value:.6g}{mark}")
        report["deltas"] = deltas
        report["eps"] = eps
        report["eps_ever_met"] = any(d is not None and d < eps for d in deltas)

        print("\n[층 풀링 β 의 포화]  §4.4 의 h_ssm 이 '가중 합' 인지 '선택' 인지")
        readout = model.read(trace.R_star, ctx)
        alpha = readout.alpha if readout.alpha is not None else trace.alpha
        if alpha is None:
            print("  (β 를 얻지 못했다)")
        else:
            a = alpha.detach().float().reshape(-1, alpha.shape[-1])
            entropy = float(-(a.clamp_min(1e-12).log() * a).sum(-1).mean())
            ceiling = math.log(a.shape[-1])
            report["beta_entropy"] = entropy
            report["beta_entropy_max"] = ceiling
            report["beta_max_mean"] = float(a.max(-1).values.mean())
            print(f"  엔트로피 {entropy:.4f} / 균등 {ceiling:.4f}  "
                  f"(β_max 평균 {report['beta_max_mean']:.4f}, 균등이면 {1/a.shape[-1]:.4f})")
            if entropy < 0.05 * ceiling:
                print("  → one-hot 으로 굳었다. 레이어 축 가중 합이 레이어 선택이 됐다 (F-028).")

    if not report.get("eps_ever_met", True):
        print(f"\n경고: 어떤 사이클도 ε={eps} 를 만족하지 못했다 — delta_state 종료가 "
              f"사실상 fixed_m(M=m_max) 로 퇴화한다 (F-028).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
