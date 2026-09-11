# v1.0 참조 지도

> 이 문서는 v1.0(레거시) 구현의 각 자산을 현재 코드베이스의 어디로 보냈는지, 그리고 **왜**인지를 기록한다.
> 판정: **이식**(거의 그대로) · **개작**(구조/의미 변경) · **폐기**(버림) · **신규**(레거시에 없음)

**최종 갱신:** 2026-09-11
**v1.0 규모:** 파이썬 6,500줄 / 테스트 16종 / 소스 1.6MB

> **레거시는 이제 `v1.0` 브랜치에만 있다.** 작업 트리의 `Legacy_LSRR/` 사본은 삭제됐고, 산출물(`runs/`·`caches/` 29GB)도 회수했다.
>
> **경로 대응 (중요):** 이 문서가 인용하는 **레거시 경로 `<경로>`는 `v1.0` 브랜치의 `<경로>`**다 — 그 브랜치의 저장소 루트가 곧 레거시 구현이다(커밋 `8a3fe34`).
>
> ```bash
> git show v1.0:lsrr/engines/hydra_qs.py     # 파일 하나 꺼내 보기
> git diff v1.0 main -- lsrr/               # 이식·개작의 실제 기록
> ```
>
> 예전 문서가 쓰던 `Legacy_LSRR/lsrr/engines/hydra_qs.py` 표기는 `v1.0:lsrr/engines/hydra_qs.py` 와 같다.
> **`main` 은 더 이상 레거시가 아니다** — 2026-09-11 에 `main` 을 v1.1 로 fast-forward 했다 (CONVENTIONS §8).

---

## 1. 판정 요약

| 레거시 경로 | 신규 위치 | 판정 | 사유 |
|---|---|---|---|
| `lsrr/interfaces.py` | `lsrr/core/interfaces.py` + `core/types.py` | **개작** | 슬롯 재편(디코더 폐지, ReadoutPath·Stability·Gate 추가). 데이터클래스는 `types.py`로 분리 |
| `lsrr/registry.py` | `lsrr/core/registry.py` | **이식** | 설계가 건전하다. 시그니처 필터링·별칭 등록 그대로. 레지스트리 목록만 갱신 |
| `lsrr/config.py` | `lsrr/config/{loader,sweep,snapshot}.py` | **개작** | 기능은 유지하되 파일 분할 + `schema.py`/`validate.py` 신설. 문자열 섹션 자동 해석(`config.py:105-116`)의 암묵 규칙은 명시적 스키마로 대체 |
| `lsrr/model.py` | `lsrr/model.py` + `lsrr/builder.py` | **개작** | 조립 지식을 `builder.py`로 분리. `model_outputs`에 모듈 객체를 싣던 패턴 제거 (ADR-005). `_resolve_d_model`의 `d_model=d_in` 강제는 폐지 (ADR-003) |
| `lsrr/backbones/extractor.py` | `lsrr/backbone/{session,extractor}.py` | **개작** | 동결·해시 로직은 이식. `position_rule` 택일 구조를 `H_last`+`H_pool` 동시 산출로 확장 (ADR-004). KV 캐시 반환 추가 (ADR-002) |
| `lsrr/adapters/layer_adapter.py` | `lsrr/memory/adapters.py` | **이식** | per-layer affine einsum·RMSNorm·layer pos emb 모두 제안서 §4.1에 정확히 대응. 레이어 위치 임베딩만 `layer_embedding.py`로 분리 |
| `lsrr/engines/wrapper.py` | `lsrr/engine/wrapper.py` | **이식** | 감쇠·사이클 임베딩·R⁰ 재주입·pre-norm — §4.2 "재귀 적응 장치" 3종을 정확히 구현. 갱신식 소유권을 문서로 못 박아 재사용 |
| `lsrr/engines/hydra_qs.py` | `lsrr/engine/core_hydra.py` | **이식** | quasiseparable 구현. 이식 시 shift 누락 회귀를 확인·고정 (F-013) |
| `lsrr/engines/mamba_up_down.py` | `lsrr/engine/core_mamba.py` | **이식** | Ablation C의 방향 3종(상향/하향/휴리스틱 양방향) |
| `lsrr/engines/attn_block.py` | `lsrr/engine/core_attention.py` | **이식** | 동예산 어텐션 베이스라인. 하드코딩된 `d_ffn=4710`을 d_model 역산으로 개작 (F-015) |
| `lsrr/engines/mlp_onepass.py` | `lsrr/engine/core_mlp.py` | **이식** | Phase 0 게이트 ③의 비교 대상 |
| `lsrr/engines/ssm_core.py` | `lsrr/engine/scan.py` | **이식** | 선택적 스캔 + shift. 코어 간 의존을 막기 위해 공용 모듈로 분리 |
| `lsrr/iteration/controller.py` | `lsrr/recurrence/{runner,schedules,tbptt,state}.py` | **개작** | 3분할 (ADR-005). 조기 종료 래칭 로직(`controller.py:129-141`)은 의미 그대로 `state.py`로 |
| `lsrr/termination/rules.py` | `lsrr/termination/{rules,signals}.py` | **개작** | 규칙은 이식, 신호 계산을 `signals.py`로 공통화. 엔트로피 규칙 신설, `M_max` 폴백을 공통 기반 클래스로 승격 (I5) |
| `lsrr/fusion/attention_pooling.py` | `lsrr/readout/fusion.py` | **개작** | α 풀링·잔차/게이트/concat 이식. 앵커를 `h_ctx`(백본 원본)로 교체하고 `W_r` 출력 폭을 `d_in`으로 (ADR-003) |
| `lsrr/decoders/light_decoder.py` | — | **폐기** | 학습 디코더 슬롯 폐지 (ADR-001) |
| — | `lsrr/backbone/continuation.py` | **신규** | 백본 연속 디코딩 (§4.4) |
| — | `lsrr/readout/path.py` | **신규** | 전 사이클 공유 판독 경로 (I3) |
| `lsrr/losses/composite.py` | `lsrr/objectives/*.py` | **개작** | 손실별 파일 분리. `AnswerNLL`·`StateVarianceReg` 이식, `DeepSupervision`은 TBPTT 윈도·γ 가중으로 개작 (ADR-006), `distillation.py` 신설 |
| `lsrr/data/{schema,collate,gsm8k,prosqa,multiplication,answer_scoring}.py` | `lsrr/data/{schema,collate,scoring}.py` + `data/datasets/` | **이식** | 데이터셋별 파일을 `datasets/`로 이동. 스키마·콜레이트·IGNORE_INDEX 규약 그대로 |
| `lsrr/data/cache.py` | `lsrr/data/cache.py` | **이식(강등)** | 샤딩 safetensors + manifest + 캐시 키 그대로. 단 기본 경로가 아닌 보조 경로 (ADR-002) |
| `lsrr/training/trainer.py` | `lsrr/runtime/trainer.py` | **개작** | 코사인 스케줄·AMP·체크포인트 이식. 판독이 백본 경유로 바뀌어 스텝 루프 재작성 |
| `lsrr/training/evaluator.py` | `lsrr/runtime/evaluator.py` + `lsrr/metrics/*` | **개작** | 채점·비용 회계를 `metrics/`로 분리. `MissingTargetError`·`flops_includes_backbone` 플래그 규약 계승 |
| `lsrr/utils/flops.py` | `lsrr/metrics/cost.py` | **개작** | 디코더 FLOPs 항을 연속 디코딩 항으로 교체 |
| `lsrr/utils/logging.py` | `lsrr/telemetry/{tracker,traces,run_index}.py` | **개작** | ExperimentTracker 이식 + 사이클 트레이스 스키마 분리 |
| `lsrr/utils/{seed,oom}.py` | `lsrr/runtime/{seeding,oom}.py` | **이식** | — |
| `scripts/download_data.py` | `lsrr/data/download.py` + `scripts/download_data.py` | **개작** | SHA-256 검증 로직은 그대로. 로직을 패키지로 옮기고 스크립트는 얇게 |
| `scripts/{train,eval,sweep,extract_h}.py` | `scripts/` 동명 | **개작** | 얇은 래퍼로 축소 |
| `scripts/make_tables.py` | `lsrr/reporting/tables.py` + `scripts/make_report.py` | **개작** | †/‡ 이원 표기·공란 원칙을 `provenance.py`로 명시화 |
| `scripts/eval_backbone_direct.py` | `scripts/eval_backbone_direct.py` | **이식** | No-CoT 백본 직접 평가 — 내부 하한 베이스라인 |
| `configs/**` | `configs/**` | **개작** | 슬롯 이름 변경 반영(`memory.*`, `readout.*`, `recurrence.*`), `decoder/` 제거, `stability/`·`ablation/` 추가 |
| `configs/published_numbers.yaml` | `configs/published_numbers.yaml` | **이식** | 외부 공표치 인용 대장 |
| `runs/`, `caches/`, `lsrr.egg-info/` | — | **폐기** | 산출물. 디코딩 경로가 달라 신규 결과와 비교 불가 |

---

## 2. 테스트 자산 판정

레거시 테스트 16종은 이 프로젝트의 가장 값진 자산이다 — 무엇이 깨지기 쉬운지에 대한 축적된 지식이기 때문이다.

| 레거시 테스트 | 신규 위치 | 판정 | 대응 불변식 |
|---|---|---|---|
| `test_freeze.py` | `tests/contracts/test_frozen_backbone.py` | 이식 | I1 |
| `test_decoder_anti_leakage.py` | `tests/contracts/test_no_leakage.py` | 개작 | I6 (디코더 자기복사 → 문맥 인코딩 정답 비노출) |
| `test_termination.py` | `tests/contracts/test_termination_fallback.py` | 이식 | I5 |
| `test_eval_protocol.py` | `tests/contracts/test_cost_accounting.py` + `tests/unit/` | 개작 | I7 |
| `test_fusion_residual.py` | `tests/contracts/test_injection_space.py` | 개작 | I8 (앵커 교체 반영, ADR-003) |
| `test_engine_equiv.py` | `tests/unit/test_hydra_port_fidelity.py` | 이식 | — (HydraQS 단방향 ≡ Mamba-Up) |
| `test_hydra_quasiseparable.py` | `tests/unit/test_hydra_port_fidelity.py` | 이식 | — |
| `test_param_matching.py` | `tests/unit/test_hydra_port_fidelity.py` | 이식 | — (Ablation C 공정성, ADR-009) |

> 위 3종은 **한 파일로 통합**했다. 셋 다 "Hydra 이식이 충실한가"라는 하나의 판정에 기여하고, `ROADMAP.md`가 레거시 폐기 게이트를 `test_hydra_port_fidelity.py` 통과로 명시하므로 그 이름을 정본으로 삼았다.
| `test_cache.py` | `tests/regression/test_cache_roundtrip.py` | 이식 | — |
| `test_reproducibility.py` | `tests/regression/test_reproducibility.py` | 이식 | — |
| `test_logging_and_resume.py` | `tests/integration/test_logging_and_resume.py` | 이식 | — |
| `test_target_padding.py` | `tests/unit/test_target_padding.py` | 이식 | I6 보조 |
| `test_gsm8k.py`, `test_download_data.py` | `tests/unit/` | 이식 | — |
| `test_oom.py` | `tests/integration/test_oom.py` | 이식 | — |
| — | `tests/contracts/test_single_encode.py` | **신규** | I2 |
| — | `tests/contracts/test_shared_readout.py` | **신규** | I3 |
| — | `tests/contracts/test_gradient_path.py` | **신규** | I4 |

---

## 3. 레거시에서 배운 것 (코드가 아니라 판단)

구현 시 잊지 말아야 할, 레거시 주석에 남아 있는 함정들:

1. **`labels` vs `target_ids`** — 패딩을 실제 토큰 id로 채운 텐서를 손실 타깃으로 쓰면 패딩을 학습한다. 레거시 `losses/composite.py:_loss_targets`가 이 구분을 명시한다. 신규에서도 `objectives/targets.py`가 같은 책임을 진다.
2. **정답 없는 배치를 건너뛰면** 자유형 데이터셋에서 100%, 수치형에서 0%가 조용히 보고된다. 레거시 `MissingTargetError`의 교훈 — 치명적 실패로 던진다.
3. **백본 FLOPs를 빼고 보고하면** 효율 비교가 무의미해진다. 레거시 evaluator의 `flops_includes_backbone` 플래그 규약을 계승한다.
4. **`d_model`을 슬롯마다 따로 해석하면** 조립 시점에 조용히 어긋난다. 레거시는 단일 해석 지점 + 불일치 시 즉시 실패로 처리했다. 신규는 `config/validate.py`가 같은 일을 로드 시점에 한다.
5. **어댑터 공간과 백본 공간의 혼동** — 레거시는 `d_model=d_in`을 강제해 이 문제를 회피했다. 신규는 ADR-003으로 공간을 명시 분리하므로, 혼동이 발생하면 조용히 학습되는 대신 I8 테스트가 잡는다.

---

## 4. 폐기 — 완료

작업 트리에서 레거시를 들어냈다. **지운 것이 아니라 옮긴 것이다** — 내용은 `v1.0` 브랜치에 그대로 있고, 커밋 이력에도 남아 있다.

| 시점 | 조치 | 상태 | 실행일 |
|---|---|---|---|
| M4 완료 후 | 레거시 `runs/`·`caches/` 삭제 (29 GB 회수) | **완료** | 2026-09-11 |
| — | 작업 트리의 `Legacy_LSRR/` 사본 삭제 (1.6 MB) | **완료** | 2026-09-11 |
| — | 레거시를 `v1.0` 브랜치로 보존 | **완료** | 2026-09-11 |

아카이브·체크섬 절차는 두지 않는다 — 브랜치가 그 역할을 한다. **`v1.0` 브랜치는 삭제하지 않는다** (CONVENTIONS §8): 이력은 `main` 안에도 있지만, 이름이 붙어 있어야 "그때 그 구조" 를 지목할 수 있고 이 문서가 참조하는 대상이기도 하다.

이 문서가 계속 유효한 이유: §1·§2 의 판정(이식/개작/폐기)은 **왜 그렇게 했는가**의 기록이며, 파일이 사라져도 그 근거는 남는다. 소스 파일 상단의 `이식: v1.0:<경로>` 주석이 이 표의 각 행을 가리킨다.
