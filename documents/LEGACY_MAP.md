# Legacy_LSRR 참조 지도

> `Legacy_LSRR/`은 **읽기 전용 참조 구현**이다. 수정하지 않는다.
> 이 문서는 레거시의 각 자산을 신규 코드베이스의 어디로 보낼지, 그리고 **왜**인지를 기록한다.
> 판정: **이식**(거의 그대로) · **개작**(구조/의미 변경) · **폐기**(버림) · **신규**(레거시에 없음)
>
> **언제 레거시를 지울 수 있는가는 §4와 [`ROADMAP.md`의 «Legacy_LSRR 생애주기»](ROADMAP.md#legacy_lsrr-생애주기)가 정한다.**

**최종 갱신:** 2026-09-10
**레거시 규모:** 파이썬 6,500줄 / 테스트 16종 / 디스크 8.3GB (소스 1.6MB + `runs/` 7.7GB + `caches/` 644MB)
**현재 생애주기 단계:** **L0 — 참조 활성**

> **경로 대응 (중요):** 이 문서가 인용하는 `Legacy_LSRR/<경로>`는 **원격 `main` 브랜치의 `<경로>`**와 같다.
> `main`의 저장소 루트가 곧 레거시 구현이고(커밋 `8a3fe34`), 로컬 `Legacy_LSRR/`은 그 사본이다.
> 예: `Legacy_LSRR/lsrr/engines/hydra_qs.py` = `main:lsrr/engines/hydra_qs.py`.
> 따라서 `v1.1` 브랜치는 `Legacy_LSRR/` 사본을 추적하지 않는다 — 부모 커밋이 이미 그 내용이며, `git diff main v1.1`이 곧 이식/개작/폐기 판정의 실제 기록이 된다.

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
| `lsrr/engines/hydra_qs.py` | `lsrr/engine/core_hydra.py` | **이식** | quasiseparable 구현. `test_hydra_quasiseparable.py`도 함께 이식 |
| `lsrr/engines/mamba_up_down.py` | `lsrr/engine/core_mamba.py` | **이식** | Ablation C의 방향 3종(상향/하향/휴리스틱 양방향) |
| `lsrr/engines/attn_block.py` | `lsrr/engine/core_attention.py` | **이식** | 동FLOPs 어텐션 베이스라인 |
| `lsrr/engines/mlp_onepass.py` | `lsrr/engine/core_mlp.py` | **이식** | Phase 0 게이트 ③의 비교 대상 |
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
| `test_engine_equiv.py` | `tests/unit/test_engine_equiv.py` | 이식 | — (HydraQS 순방향 분기 ≡ Mamba-Up) |
| `test_hydra_quasiseparable.py` | `tests/unit/test_hydra_quasiseparable.py` | 이식 | — |
| `test_param_matching.py` | `tests/unit/test_budget_match.py` | 이식 | — (Ablation C 공정성, ADR-009) |
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

## 4. 폐기 계획

`Legacy_LSRR/`은 **별도 경로에서 관리되고 있다.** 이 저장소의 사본은 언제든 지워도 되며, 다시 필요해지면 사용자에게 요청한다. 따라서 아카이브·체크섬 절차는 두지 않는다.

| 시점 | 조치 | 선행 조건 |
|---|---|---|
| **M4 완료 후** | `runs/`·`caches/` 삭제 (8.3 GB) | `tests/unit/test_hydra_port_fidelity.py` 통과 |
| **M7 완료 후** | `Legacy_LSRR/` 전체 삭제 (1.6 MB) | 아래 §1·§2의 이식/개작 항목이 `MODULE_INDEX.md`에서 전부 `검증` 상태 |

상세는 [`ROADMAP.md`의 «Legacy_LSRR 폐기 계획»](ROADMAP.md#legacy_lsrr-폐기-계획).

### 실행 기록

| 시점 | 상태 | 실행일 |
|---|---|---|
| `runs/`·`caches/` 삭제 | 미실행 | — |
| `Legacy_LSRR/` 전체 삭제 | 미실행 | — |
