# 모듈 인덱스 (지속 갱신 보드)

> **이 문서는 코드베이스의 현재 상태판이다.** 모듈을 추가·삭제·이동하거나 상태가 바뀌면 해당 폴더 `README.md`와 **이 표를 함께** 갱신한다 (`CONVENTIONS.md` §5).
> 상태: `계획` · `구현중` · `구현`(동작함) · `검증`(테스트 통과)

**최종 갱신:** 2026-09-11 · 전체 상태: **M5 완료 — 감독과 진단, Phase 0 게이트 판정기 (테스트 287종 통과)**

> 이 표의 `상태` 열은 **`Legacy_LSRR` 폐기 게이트의 판정 근거**이기도 하다 (`LEGACY_MAP.md` §4). 이식/개작 항목이 전부 `검증`이 되기 전에는 레거시를 지울 수 없다.

---

## 진행 요약

| 패키지 | 모듈 수 | 계획 | 구현중 | 구현 | 검증 | 담당 마일스톤 |
|---|---:|---:|---:|---:|---:|---|
| `core` | 5 | 0 | 0 | 0 | 5 | M1 ✅ |
| `config` | 5 | 0 | 0 | 0 | 5 | M1 ✅ |
| `backbone` | 6 | 1 | 0 | 0 | 5 | M2 ✅ (interventions는 M7) |
| `memory` | 5 | 0 | 0 | 0 | 5 | M3 ✅ |
| `engine` | 6 | 3 | 0 | 0 | 3 | M3 부분 ✅ / M4 |
| `recurrence` | 5 | 0 | 0 | 0 | 5 | M3 ✅ |
| `termination` | 5 | 2 | 0 | 0 | 3 | M3 부분 ✅ |
| `readout` | 4 | 1 | 0 | 0 | 3 | M3 ✅ |
| `stability` | 5 | 5 | 0 | 0 | 0 | 조건부 (ADR-007) |
| `objectives` | 6 | 3 | 0 | 0 | 3 | M3 부분 ✅ / M5 |
| `data` | 6+7 | 8 | 0 | 0 | 5 | M3 부분 ✅ |
| `runtime` | 7 | 1 | 0 | 0 | 6 | M3 ✅ |
| `metrics` | 5 | 5 | 0 | 0 | 0 | M5 / M6 |
| `telemetry` | 4 | 2 | 0 | 0 | 2 | M3 부분 ✅ |
| `analysis` | 6 | 6 | 0 | 0 | 0 | M5(collapse) / M7 |
| `reporting` | 3 | 3 | 0 | 0 | 0 | M6 |
| `gates` | 3 | 3 | 0 | 0 | 0 | M5 |
| 루트 | 3 | 0 | 0 | 0 | 3 | M3 ✅ |
| **합계** | **96** | **38** | **0** | **0** | **58** | |

---

## 패키지 루트

| 모듈 | 역할 | 상태 |
|---|---|---|
| `lsrr/__init__.py` | 패키지 진입점, 버전 | 검증 |
| `lsrr/model.py` | `LSRRModel` — 조립된 전체 모델의 순전파 계약 | 검증 |
| `lsrr/builder.py` | 설정 → 슬롯 인스턴스화 → 모델 조립 (유일하게 전 슬롯을 아는 곳) | 검증 |

## `lsrr/core/` — 계약

| 모듈 | 역할 | 상태 |
|---|---|---|
| `interfaces.py` | 전 슬롯의 추상 기반 클래스 | 검증 |
| `types.py` | `DataSample`, `ContextBundle`, `CycleDiagnostics`, `ReasoningTrace`, `ReadoutResult`, `CostReport` | 검증 |
| `registry.py` | 범용 레지스트리 + 슬롯별 인스턴스 | 검증 |
| `invariants.py` | I1–I8 실행 시 단언 헬퍼 (인코딩 카운터, 동결 검사, 인스턴스 동일성) | 검증 |
| `errors.py` | 타입 예외 (`FrozenBackboneViolation`, `MultipleEncodeError`, `MissingTargetError`, `LeakageError`, …) | 검증 |

## `lsrr/config/` — 설정 처리

| 모듈 | 역할 | 상태 |
|---|---|---|
| `schema.py` | 설정 트리의 타입 스키마 | 검증 |
| `loader.py` | 계층 defaults 해석 + CLI dotlist 병합 | 검증 |
| `validate.py` | 교차 필드 검증 (로드 시점 실패) | 검증 |
| `sweep.py` | 데카르트 곱 전개, 묶음 축, 인덱스 주소지정(slurm 배열) | 검증 |
| `snapshot.py` | 해석 완료 설정 덤프 + 설정 해시(런 신원) | 검증 |

## `lsrr/backbone/` — 동결 백본

| 모듈 | 역할 | 상태 |
|---|---|---|
| `session.py` | `BackboneSession` — 1회 순전파 → `ContextBundle` (I2 카운터 소유) | 검증 |
| `freeze.py` | 동결 강제 + 가중치 해시 (I1) | 검증 |
| `extractor.py` | 레이어별 수직 1열 추출 (`H_last`) | 검증 |
| `pooling.py` | 레이어별 질문 전체 어텐션 풀링 (`H_pool`, §4.1 / ADR-004) | 검증 |
| `continuation.py` | `h_fusion` 주입 + KV 재사용 자기회귀 생성 (§4.4 / ADR-001) | 검증 |
| `interventions.py` | back-patching 오라클, logit-lens 투영 훅 (분석용) | 계획 |

## `lsrr/memory/` — H → R⁰

| 모듈 | 역할 | 상태 |
|---|---|---|
| `composer.py` | `H_last` + `H_pool` 결합 (add/concat/gate/last_only) | 검증 |
| `scoping.py` | 레이어 범위 선택 — **Ablation A** | 검증 |
| `adapters.py` | per-layer affine / shared affine / identity (+RMSNorm) | 검증 |
| `layer_embedding.py` | 레이어 위치 임베딩 | 검증 |
| `pipeline.py` | 위 넷을 하나의 `nn.Module`로 조립 → `R⁰` | 검증 |

## `lsrr/engine/` — 레이어 축 정제 연산자

| 모듈 | 역할 | 상태 |
|---|---|---|
| `wrapper.py` | §4.2 갱신식 소유: 감쇠·R⁰ 재주입·사이클 임베딩·pre-norm — **Ablation D** | 검증 |
| `scan.py` | 선택적 스캔 프리미티브 + quasiseparable shift (SSM 코어 공용) | 검증 |
| `core_hydra.py` | quasiseparable 양방향 스캔 (기본) | 검증 |
| `core_mamba.py` | 단방향 상향/하향, 휴리스틱 양방향 — **Ablation C** | 검증 |
| `core_attention.py` | 동예산 어텐션 블록 (내부 베이스라인) | 검증 |
| `core_mlp.py` | 1회 통과 MLP (Phase 0 게이트 ③ 비교 대상) | 검증 |
| `budget.py` | 파라미터 예산 정합 (비교 공정성, ADR-009). FLOPs 축은 M8 | 검증 |

## `lsrr/recurrence/` — 사이클 축 제어

| 모듈 | 역할 | 상태 |
|---|---|---|
| `runner.py` | `CycleRunner.run_train` / `run_eval` | 검증 |
| `schedules.py` | M 샘플링 (fixed/uniform/lognormal), γ 가중 | 검증 |
| `tbptt.py` | 절단 BPTT 윈도 계산·detach 정책 (윈도 경계를 공개, ADR-006) | 검증 |
| `state.py` | 샘플별 조기 종료 래칭, 실행 마스크 | 검증 |
| `hooks.py` | 사이클 콜백 프로토콜 (진단·깊은 감독·anytime) | 검증 |

## `lsrr/termination/` — 수렴 종료

| 모듈 | 역할 | 상태 |
|---|---|---|
| `base.py` | 규칙 기반 클래스 — `M_max` 폴백 강제 (I5) | 검증 |
| `signals.py` | 신호 계산: 상태 Δ, 출력 KL, 엔트로피 | 검증 |
| `rules.py` | `fixed_m` / `delta_state` / `kl_output` / `entropy_output` — **Ablation B** | 검증 |
| `calibration.py` | ε 스윕·검증셋 임계값 선택 | 계획 |
| `behavior.py` | 수렴 거동 분류 (수렴/진동/드리프트) — **게이트 ④** | 검증 |

## `lsrr/readout/` — 공유 판독 경로

| 모듈 | 역할 | 상태 |
|---|---|---|
| `fusion.py` | α 어텐션 풀링, `h_fusion = h_ctx + W_r·h_SSM` (ADR-003) | 검증 |
| `injection.py` | 주입 공간 정합·노름 보정 (I8) | 검증 |
| `path.py` | `ReadoutPath` — 전 사이클 공유 단일 객체 (I3) | 검증 |
| `decode.py` | teacher-forcing / greedy 디코딩, EOS 절단 | 계획 |

## `lsrr/stability/` — 안정화 사다리 (조건부)

| 모듈 | 역할 | 상태 |
|---|---|---|
| `ladder.py` | 승급 정책과 활성 단 기록 (ADR-007) | 계획 |
| `jacobian.py` | Jacobian 정규화 (Hutchinson 추정) — 1단 | 계획 |
| `spectral_norm.py` | 스펙트럴 노름 제약 — 2단 | 계획 |
| `monotone.py` | monotone 재파라미터화 — 3단 | 계획 |
| `probes.py` | 수축 계수 추정, Δ 궤적 비감소 구간 측정 | 계획 |

## `lsrr/objectives/` — 손실

| 모듈 | 역할 | 상태 |
|---|---|---|
| `targets.py` | 정답 구간 마스킹, `IGNORE_INDEX` 규약 (I6) | 검증 |
| `answer_nll.py` | `L_NLL` | 검증 |
| `deep_supervision.py` | `L_DeepSup` — 답-앵커형, TBPTT 윈도 내, γ 가중 (ADR-006) | 검증 |
| `variance_reg.py` | `L_VarReg` — 상태 분산 하한 | 계획 |
| `distillation.py` | `L_KD` — **ablation 전용** (JS 유계·stop-grad 교사·정답 조건부 가중) | 계획 |
| `composite.py` | 가중 합성과 항별 지표 분리 보고 | 검증 |

## `lsrr/data/` — 데이터

| 모듈 | 역할 | 상태 |
|---|---|---|
| `schema.py` | `DataSample` 스키마와 split 규약 | 검증 |
| `prompting.py` | 질문 렌더링·토크나이즈 계약 (정답 비노출, I6) | 검증 |
| `collate.py` | 배치·패딩·라벨 마스킹 | 검증 |
| `scoring.py` | 데이터셋별 정답 정규화·exact match | 계획 |
| `cache.py` | 샤딩 safetensors H 캐시 (보조 경로, ADR-002) | 계획 |
| `download.py` | 데이터 획득 + SHA-256 검증 | 계획 |
| `datasets/gsm8k_aug.py` | 주 학습·ID 평가 | 계획 |
| `datasets/math_ood.py` | GSM-Hard / MultiArith / SVAMP (평가 전용) | 계획 |
| `datasets/prosqa.py` | 홉 수 통제, 메커니즘 분석 주 무대. `meta["hops"]` 로 층화 | 검증 |
| `datasets/prontoqa.py` | 홉 수 통제 보조 | 계획 |
| `datasets/multiplication.py` | 다자리 곱셈 합성 (용량 확장 검증) | 검증 |
| `datasets/commonsenseqa.py` | 비수학 일반성 (CODI 공개 CoT) | 계획 |
| `datasets/closed_book_multihop.py` | 범위 검증 — 이득 **부재**를 예측 | 계획 |

## `lsrr/runtime/` — 구동

| 모듈 | 역할 | 상태 |
|---|---|---|
| `trainer.py` | 학습 루프, 옵티마이저·스케줄 | 검증 |
| `evaluator.py` | 평가 구동 (채점은 `metrics`에 위임) | 계획 |
| `checkpoint.py` | 저장·로드 (학습 파라미터만) | 검증 |
| `resume.py` | 중단 재개 | 계획 |
| `seeding.py` | 시드·결정성 | 검증 |
| `device.py` | 디바이스·dtype·AMP 정책 | 검증 |
| `oom.py` | OOM 감지·배치 축소 대응 | 계획 |

## `lsrr/metrics/` — 채점과 비용

| 모듈 | 역할 | 상태 |
|---|---|---|
| `accuracy.py` | 최종답 exact match 프로토콜 | 검증 |
| `anytime.py` | 사이클별 정확도 곡선 (§5 부산물) — **게이트 ②** | 검증 |
| `cost.py` | FLOPs·지연 회계 — 백본 1회 포함 (I7) | 검증 |
| `pareto.py` | 정확도-연산/지연 프론티어 | 계획 |
| `aggregate.py` | 시드 집계·신뢰구간·유의성 검정 — **게이트 ③** | 검증 |
| `evaluate.py` | 평가 실행 → 게이트 판정 데이터 수집 (계획 외 추가) | 검증 |

## `lsrr/telemetry/` — 기록

| 모듈 | 역할 | 상태 |
|---|---|---|
| `tracker.py` | 런 디렉터리·설정 스냅샷·지표 기록 | 검증 |
| `traces.py` | 사이클 진단 트레이스 스키마·영속화 | 검증 |
| `run_index.py` | `runs/runs_index.csv` 대장 | 계획 |
| `console.py` | 진행 표시·요약 출력 | 계획 |

## `lsrr/analysis/` — 메커니즘 분석

| 모듈 | 역할 | 상태 |
|---|---|---|
| `plugin.py` | 플러그인 프로토콜과 트레이스 러너 | 계획 |
| `collapse.py` | 표현 동질화·자명해 진단 (게이트 ①) | 검증 |
| `logit_lens.py` | 사이클별 `R⁽ᵐ⁾` 투영, 정제 궤적 | 계획 |
| `alpha_profile.py` | α_l 분포 (중간층 집중 가설) | 계획 |
| `hop_convergence.py` | 홉 수 ↔ 평균 수렴 사이클 | 계획 |
| `backpatch_recovery.py` | 오라클 상한 대비 회수율 | 계획 |

## `lsrr/reporting/` — 보고

| 모듈 | 역할 | 상태 |
|---|---|---|
| `provenance.py` | †공표 / ‡자체 측정 표기, 공란 원칙 | 계획 |
| `tables.py` | 런 인덱스 + 공표치 → 논문 표 | 계획 |
| `figures.py` | anytime·파레토·Δ 궤적·α 히트맵 | 계획 |

## `lsrr/gates/` — 단계 판정

| 모듈 | 역할 | 상태 |
|---|---|---|
| `criteria.py` | 게이트 술어 (유의성 검정 포함) | 검증 |
| `definitions.py` | Phase별 게이트 집합 정의와 임계값 | 검증 |
| `report.py` | PASS/FAIL 판정문 + 킬 스위치 신호 | 검증 |

## `scripts/` — CLI 진입점

| 스크립트 | 역할 | 상태 |
|---|---|---|
| `download_data.py` | 데이터 획득·검증 | 계획 |
| `extract_h.py` | H 캐시 사전 추출 (보조 경로) | 계획 |
| `train.py` | 학습. `--runs-dir` 플래그, `RUN_DIR=` 기계 판독 출력 | 검증 |
| `eval.py` | 평가 → `metrics.json`·`gate_inputs.json`. 종료 규칙 스윕은 M6 | 검증 |
| `sweep.py` | **YAML 주도** 스윕 → train+eval 연쇄. `--index` 로 slurm 배열 1:1. 판정은 하지 않는다 | 검증 |
| `eval_backbone_direct.py` | No-CoT 백본 직접 평가 (하한) | 계획 |
| `profile_cost.py` | FLOPs·지연 프로파일 | 계획 |
| `analyze.py` | 분석 플러그인 실행 | 계획 |
| `diagnose_injection.py` | **주입 통로 진단** — 개입 비교로 질문별 학습을 판정 (F-010) | 검증 |
| `diagnose_trajectory.py` | **궤적 분화 진단** — ADR-015 선행 관문. 토큰 분화 + 토큰별 기여 (F-026) | 검증 |
| `diagnose_state_scale.py` | **상태 스케일 진단** — Δ 자릿수와 β 포화. §4.3 의 고정 ε 전제 확인 (F-028) | 검증 |
| `slurm/sweep_array.sh`·`slurm/submit.sh` | 배열 작업 제출. 배열 크기는 설정에서 읽는다 | 검증 |
| `check_gates.py` | 단계 게이트 판정 | 검증 |
| `make_report.py` | 표·그림 생성 | 계획 |
