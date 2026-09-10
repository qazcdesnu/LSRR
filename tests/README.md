# `tests/` — 테스트

## 역할
아키텍처가 무너지지 않았음을 기계적으로 확인한다. 특히 **불변식(I1–I8)은 테스트 없이는 존재하지 않는 것으로 간주한다.**

## 구조

| 디렉터리 | 대상 | 규칙 |
|---|---|---|
| `contracts/` | 불변식 I1–I8 | 불변식 1개당 최소 1개. `ARCHITECTURE.md` §4 표와 1:1 대응 |
| `unit/` | 개별 모듈 | 외부 의존 없이, 작은 텐서로 |
| `integration/` | 슬롯 조합 | 조립 루트를 통과하는 최소 학습 스텝 |
| `regression/` | 수치 재현성 | 시드 고정 비트 단위 동일성, 캐시 왕복 오차 |

전부 **GPU 없이 CPU에서** 통과해야 한다 (작은 백본 스텁 허용).

## 계약 테스트 ↔ 불변식 대응

| 테스트 | 불변식 | 검사 내용 | 상태 |
|---|---|---|---|
| `contracts/test_frozen_backbone.py` | **I1** | 백본 `requires_grad=False`, 학습 전후 가중치 해시 동일 | 검증 |
| `contracts/test_single_encode.py` | **I2** | 샘플 배치당 백본 인코딩 정확히 1회, 사이클 루프 내 재진입 시 예외 | 검증 |
| `contracts/test_shared_readout.py` | **I3** | 모든 사이클 판독이 동일 `ReadoutPath` 인스턴스 | 검증 |
| `contracts/test_gradient_path.py` | **I4** | 백본 파라미터 grad는 `None`, 엔진 grad는 `h_fusion` 경유로만 존재 | 검증 |
| `contracts/test_termination_fallback.py` | **I5** | 수렴 수열은 조기 종료, 진동 수열은 `M_max`에서 정지 | 검증 |
| `contracts/test_no_leakage.py` | **I6** | 인코딩 입력에 정답 토큰 부재, 패딩은 `IGNORE_INDEX` | 검증 |
| `contracts/test_cost_accounting.py` | **I7** | 비용 보고에 백본 항 포함, 미측정 시 플래그 | 계획 |
| `contracts/test_injection_space.py` | **I8** | `h_fusion` 차원 = `d_in`, 어댑터 공간 벡터 주입 시 예외 | 검증 |

## 단위·회귀 테스트 (레거시 이식)

| 테스트 | 검사 내용 | 출처 | 상태 |
|---|---|---|---|
| `unit/test_hydra_quasiseparable.py` | quasiseparable 행렬 구조의 정확성 | 이식 | 계획 |
| `unit/test_engine_equiv.py` | HydraQS 순방향 분기 ≡ Mamba-Up (수치 동일) | 이식 | 계획 |
| `unit/test_hydra_port_fidelity.py` | 레거시 체크포인트의 엔진 가중치를 적재해 순전파 일치 확인 — **레거시 `runs/` 폐기(L1)의 선행 조건** | 신규 | 계획 |
| `unit/test_budget_match.py` | 어텐션 블록 ↔ HydraQS 파라미터 차이 5% 이내 | 이식 | 계획 |
| `unit/test_target_padding.py` | `labels`/`target_ids` 구분, 패딩 미학습 | 이식 | 계획 |
| `unit/test_gsm8k.py`, `unit/test_download_data.py` | 데이터 스키마·다운로드 검증 | 이식 | 계획 |
| `regression/test_reproducibility.py` | 시드 고정 다중 실행 비트 단위 동일 | 이식 | 계획 |
| `regression/test_cache_roundtrip.py` | 캐시 키 유일성, 텐서 복원 오차 < 1e-5 | 이식 | 계획 |
| `integration/test_logging_and_resume.py` | 런 기록·중단 재개 | 이식 | 계획 |
| `integration/test_oom.py` | OOM 감지·배치 축소 | 이식 | 계획 |
| `integration/test_assembly_smoke.py` | 조립 루트 통과 (더미 슬롯) | 신규 | 검증 |
| `integration/test_training_loop.py` | **실제 GPT-2로 학습 루프 1 epoch** — 런 산출물·예산·체크포인트 왕복 | 신규 | 검증 |

## 규약
- **새 불변식을 추가하면 같은 커밋에 계약 테스트를 넣는다** (`CONVENTIONS.md` §4).
- 테스트가 없는 불변식은 문서상의 희망일 뿐이다.
- 레거시 테스트 16종은 "무엇이 깨지기 쉬운가"에 대한 축적된 지식이다. 이식 판정은 `documents/LEGACY_MAP.md` §2 참조.

## 상태
**진행 중** — 204종 통과 (계약 45 / 단위 137 / 통합 22). I1·I2·I3·I4·I5·I6·I8 계약 테스트 완료; I7(비용 회계)은 `metrics`가 생기는 M5.
