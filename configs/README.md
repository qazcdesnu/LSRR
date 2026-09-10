# `configs/` — 설정 트리

> `configs/`는 **설정 파일(YAML)**, `lsrr/config/`는 **설정을 다루는 코드**.

## 역할
실험을 정의한다. **실험 = 설정 조합**이며, 새 실험을 위해 새 파이썬 파일을 만들어야 한다면 슬롯 설계가 틀린 것이다 (ADR-009).

## 구조

```
configs/
  base.yaml                 # 공통 기본값 (시드, 결정성, 학습 하이퍼파라미터 기본)
  published_numbers.yaml    # 외부 공표치 인용 대장 (†표기 원본)

  backbone/     gpt2.yaml, llama32_1b.yaml, llama32_3b.yaml, llama31_8b.yaml, mistral_7b.yaml
  data/         gsm8k_aug.yaml, prosqa.yaml, prontoqa.yaml, mult_4x4.yaml,
                math_ood.yaml, commonsenseqa.yaml, closed_book_multihop.yaml
  memory/       default.yaml (composer·scope·adapter·layer_emb)
  engine/       hydra_qs.yaml, mamba_up.yaml, mamba_down.yaml, bidir_add.yaml,
                attn_block.yaml, mlp_onepass.yaml
  recurrence/   default.yaml (schedule·tbptt·hooks)
  termination/  delta_state.yaml, kl_output.yaml, entropy_output.yaml, fixed_m.yaml
  readout/      default.yaml (fusion_type·W_r 초기화·주입 정합)
  stability/    none.yaml, jacobian.yaml, spectral.yaml, monotone.yaml
  objective/    default.yaml, with_deepsup.yaml, with_kd.yaml

  exp/          phase0_mult.yaml, phase0_prosqa.yaml, gsm8k_main.yaml,
                scope_check.yaml, scale_curve.yaml
  ablation/     A_memory_scope.yaml, B_termination.yaml, C_engine.yaml,
                D_recurrence.yaml, E_distillation.yaml
```

## 병합 순서
`base.yaml` → 도메인 조각 → `exp/` 또는 `ablation/` → CLI dotlist(최우선).

```bash
python scripts/train.py exp=phase0_prosqa engine.type=mamba_up train.lr=5e-4
```

## 규약
- 설정 키는 **슬롯 이름과 일치**한다: `memory.scope.type`, `engine.type`, `termination.type`, `objective.deep_supervision.gamma`.
- 레지스트리 키(= `type` 값)는 **논문 표기와 일치**하며 한 번 표에 실리면 바꾸지 않는다.
- 모든 런은 해석 완료 설정을 `runs/<run_id>/config.yaml`에 스냅샷한다. 그 해시가 런 신원이다.
- 검증은 전부 로드 시점에 (`lsrr/config/validate.py`).

## 절제 실험 대응 (§7)

| 실험 | 노브 | 파일 |
|---|---|---|
| **A. 메모리 범위** | `memory.scope.type ∈ {final_only, all_layers, mid_band, late_band}` | `ablation/A_memory_scope.yaml` |
| **B. 종료 규칙** | `termination.type` × ε 스윕 | `ablation/B_termination.yaml` |
| **C. 사고 엔진** | `engine.type` + `engine.budget_match` | `ablation/C_engine.yaml` |
| **D. 재귀 적응·감독** | `engine.damping_alpha`, `engine.reinject_r0`, `engine.cycle_embedding`, `objective.deep_supervision.{enabled,gamma}` | `ablation/D_recurrence.yaml` |
| **KD 부스터** | `objective.distillation.enabled` | `ablation/E_distillation.yaml` |

`mid_band`·`late_band`의 경계는 **레이어 수의 비율**로 지정한다(절대 인덱스 금지 — 백본을 바꾸면 의미가 달라진다).

## 레거시 참조
`Legacy_LSRR/configs/` — **개작**. 계층 defaults 구조와 `sweep:` 절은 그대로. `decoder/` 제거(ADR-001), `memory/`·`readout/`·`recurrence/`·`stability/`·`ablation/` 추가, 슬롯 이름 변경 반영.
`published_numbers.yaml`은 **이식**.

## 상태
계획 — M1부터 점진 추가
