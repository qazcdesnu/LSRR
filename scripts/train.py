#!/usr/bin/env python
"""학습 진입점.

    python scripts/train.py exp=phase0_mult train.lr=5e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from lsrr.builder import build_slots
from lsrr.config import load_config
from lsrr.config.schema import get_path
from lsrr.data import PromptEncoder, prompt_spec_from_cfg
from lsrr.data.collate import make_loader
from lsrr.model import LSRRModel
from lsrr.runtime import Trainer, load_checkpoint, parameter_summary, resolve_device, set_seed
from lsrr.runtime.curriculum import apply_stage, stages_from_cfg
from lsrr.runtime.phases import apply_phase, attach_phase_lora, phases_from_cfg
from lsrr.telemetry import ExperimentTracker


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # 런 루트는 **실험 설정이 아니라 실행 인자**다. 설정 키로 두면 스냅샷에 들어가
    # 같은 실험이 저장 위치에 따라 다른 신원을 갖게 된다.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="런 이름의 앞자리. 스윕이 자식 이름을 물려줄 때 쓴다 (기본: exp=<name>)",
    )
    ap.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="학습 파라미터의 시작점 체크포인트. 2단(Phase B)이 1단의 Phase A "
             "체크포인트에서 이어 갈 때 쓴다 (ADR-016). 저장 위치이지 실험 설정이 "
             "아니므로 플래그다",
    )
    ap.add_argument("-h", "--help", action="store_true")
    args, rest = ap.parse_known_args(argv)
    if args.help:
        print(__doc__)
        print("옵션:\n  --runs-dir DIR   런 디렉터리 루트 (기본 runs)")
        print("  --run-name NAME  런 이름 앞자리 (기본: exp=<name>)")
        print("  --init-from CKPT 시작점 체크포인트 (2단이 1단의 Phase A 를 이어 받을 때)")
        print("  나머지 인자는 설정 오버라이드다: exp=<name> 또는 key=value")
        return 0

    cfg = load_config(rest)

    exp_name = args.run_name or next(
        (a.split("=", 1)[1] for a in rest if a.startswith(("exp=", "config="))), "exp"
    )
    seed = int(get_path(cfg, "seed", 42))
    generator = set_seed(seed, bool(get_path(cfg, "deterministic", True)))
    device = resolve_device(str(get_path(cfg, "device", "auto")))

    bundle = build_slots(cfg)
    model = LSRRModel(bundle=bundle, cfg=cfg, runner=bundle.runner)

    init_meta = None
    if args.init_from:
        # 1단의 Phase A 체크포인트에서 시작한다. "나머지 동일" 의 가장 강한 형태 —
        # 같은 가중치에서 출발하므로 Phase A 의 난수 편차(F-032)가 두 단계 사이에
        # 끼어들지 않는다. 백본 지문 대조는 load_checkpoint 가 한다.
        payload = load_checkpoint(model, args.init_from, verify_backbone=True)
        if payload.get("lora"):
            raise SystemExit(
                f"--init-from 체크포인트에 LoRA 델타가 있다: {args.init_from}. "
                f"시작점은 Phase A(어댑터 없음)여야 한다 — Phase B 산출물 위에 다시 "
                f"Phase B 를 얹으면 정렬 이득이 이중 계상된다."
            )
        init_meta = {
            "path": str(args.init_from),
            "step": payload.get("step"),
            "phase": (payload.get("meta") or {}).get("phase"),
        }
        print(f"시작점: {args.init_from} (step {init_meta['step']}, "
              f"phase {init_meta['phase']})")

    spec = prompt_spec_from_cfg(cfg)
    encoder = PromptEncoder(bundle.encoder.tokenizer, spec)
    loader = make_loader(
        bundle.data.get_split("train"),
        encoder,
        batch_size=int(get_path(cfg, "train.bs", 16)),
        shuffle=True,
        generator=generator,
    )

    with ExperimentTracker(
        cfg, exp_name=exp_name, seed=seed, root=args.runs_dir
    ) as tracker:
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
            init_from=init_meta,
        )
        print(
            f"학습 파라미터 {params['total']:,} / 백본 {backbone_n:,} "
            f"= {params['total'] / backbone_n:.1%}"
        )
        print(f"모듈별: {params['by_module']}")

        # 2원화 학습 (v2.1 §5.0). `train.phases` 가 없으면 단일 페이즈로 되돌린다.
        plan = phases_from_cfg(cfg)
        multi = len(plan) > 1 or plan[0].attach_lora
        if multi:
            print(f"페이즈 {len(plan)}개: "
                  + " → ".join(f"{ph.name}({ph.epochs}ep, lr={ph.lr:g}, "
                               f"{'+'.join(ph.trainable)})" for ph in plan))

        results = []
        for phase in plan:
            if phase.attach_lora:
                n = attach_phase_lora(bundle.encoder, cfg)
                if n:
                    print(f"[{phase.name}] LoRA 장착: {n:,} 파라미터 "
                          f"({n / backbone_n:.2%} of 백본)")
                    tracker.update_meta(lora_params=n)
            groups = apply_phase(model, bundle.encoder, phase)
            total = sum(groups["by_group"].values())
            print(f"[{phase.name}] 학습 대상 {total:,} — {groups['by_group']}")
            tracker.update_meta(**{f"phase_{phase.name}_trainable": groups["by_group"]})

            stages = stages_from_cfg(cfg) if phase.name == "A" else []
            if not stages:
                trainer = Trainer(
                    model, bundle.objective, tracker, cfg, device=device,
                    params=groups["params"], param_groups=groups["param_groups"], lr=phase.lr, epochs=phase.epochs,
                    phase=phase.name if multi else None,
                )
                r = trainer.fit(loader)
                print(f"[{phase.name}] 완료: {r['steps']} 스텝, {r['seconds']:.1f}초")
                results.append(r)
                continue

            # 단계적 잠재화 커리큘럼 (§5.1). 스테이지가 바꾸는 것은 타깃(남은 CoT +
            # 답)·M·옵티마이저(리셋)뿐이고, 학습 대상은 페이즈가 정한 그대로다.
            print(f"[{phase.name}] 커리큘럼 {len(stages)}단계: "
                  + " → ".join(f"{st.name}(k={st.remove_cot}, M={st.M or '동적'}, {st.epochs}ep)"
                               for st in stages))
            total_steps = total_sec = 0.0
            for st in stages:
                stage_spec = st.prompt_spec(spec)
                stage_loader = make_loader(
                    bundle.data.get_split("train"),
                    PromptEncoder(bundle.encoder.tokenizer, stage_spec),
                    batch_size=int(get_path(cfg, "train.bs", 16)),
                    shuffle=True, generator=generator,
                )
                info = apply_stage(bundle.runner, cfg, st)
                trainer = Trainer(  # 새 Trainer = 옵티마이저·스케줄러 리셋
                    model, bundle.objective, tracker, cfg, device=device,
                    params=groups["params"], param_groups=groups["param_groups"], lr=phase.lr, epochs=st.epochs,
                    phase=f"{phase.name}_{st.name}",
                )
                r = trainer.fit(stage_loader)
                total_steps += r["steps"]; total_sec += r["seconds"]
                print(f"[{phase.name}·{st.name}] 완료: {r['steps']} 스텝, {r['seconds']:.1f}초  "
                      f"(M={info['M']}, 감독 최대 {stage_spec.max_answer_tokens}토큰)")
                tracker.update_meta(**{f"stage_{st.name}": {"steps": r["steps"], "M": info["M"],
                                                             "remove_cot": st.remove_cot}})
            # 마지막 스테이지(S3)가 phase_A.pt 가 되도록 페이즈 이름으로 한 번 더 남긴다
            from lsrr.runtime.checkpoint import save_checkpoint
            save_checkpoint(model, tracker.checkpoint_dir / f"phase_{phase.name}.pt",
                            step=int(total_steps), meta={"phase": phase.name, "curriculum": True})
            results.append({"steps": total_steps, "seconds": total_sec})

        result = {
            "steps": sum(r["steps"] for r in results),
            "seconds": sum(r["seconds"] for r in results),
        }
        print(f"완료: {result['steps']} 스텝, {result['seconds']:.1f}초 → {tracker.dir}")
        # 기계 판독용 한 줄. 스윕이 이 줄로 런 디렉터리를 집는다 —
        # 사람이 읽는 줄을 파싱하게 두면 문구를 고칠 때 스윕이 조용히 깨진다.
        print(f"RUN_DIR={tracker.dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
