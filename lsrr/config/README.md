# `lsrr/config/` — 설정 처리

> `lsrr/config/`는 **설정을 다루는 코드**, `configs/`는 **설정 파일(YAML)**. 헷갈리지 않도록 주의.

## 역할
YAML 설정 트리를 읽고, 병합하고, 검증하고, 스윕으로 전개하고, 런의 신원으로 고정한다.

## 경계
- **한다:** 계층 defaults 해석, CLI dotlist 병합, 타입·교차 제약 검증, 데카르트 곱 전개, 해석 완료 설정 스냅샷과 해시.
- **하지 않는다:** 컴포넌트를 만들지 않는다(그것은 `builder.py`). 설정의 *의미*를 해석하지 않는다 — `engine.type`이 무엇을 뜻하는지는 레지스트리가 안다.

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `schema.py` | 설정 트리의 타입 스키마 (슬롯별 필수·선택 키) | 계획 |
| `loader.py` | `defaults` 재귀 해석 → 병합 → CLI dotlist 최우선 적용 | 계획 |
| `validate.py` | 교차 필드 검증, 로드 시점 즉시 실패 | 계획 |
| `sweep.py` | `sweep:` 절 데카르트 곱 전개, 자식 런 명명 | 계획 |
| `snapshot.py` | 해석 완료 설정 YAML 덤프 + 설정 해시 | 계획 |

## 핵심 계약

### 병합 순서
`configs/base.yaml` → 도메인 조각(`backbone/`, `data/`, `memory/`, `engine/`, `recurrence/`, `termination/`, `readout/`, `stability/`, `objective/`) → `exp/` 또는 `ablation/` → CLI dotlist.

### 검증 규칙 (`validate.py`) — 최소 목록
- `readout.fusion_type == "residual"`이면 `W_r` 출력 폭 = `backbone.d_in` (I8)
- `termination.m_max >= recurrence.schedule.max`
- `recurrence.tbptt_k <= recurrence.schedule.max`
- `objective.deep_supervision.enabled`이면 `recurrence.hooks.readout_per_cycle == true`
- `memory.scope.type == "final_only"`이면 `L' == 1` — 이때 엔진의 스캔 축 길이가 1임을 경고로 알림 (Ablation A의 의도된 축퇴)
- `engine.budget_match`가 설정되면 대상 엔진과 파라미터 수 차이 5% 이내인지 확인 (ADR-009)
- `experimental.reencoding_loop`가 켜지면 I2 예외 플래그를 런 메타에 기록 (ADR-010)

**학습 3시간 뒤에 터지는 설정 오류는 없어야 한다.** 검증은 전부 로드 시점에.

### 런 신원
`snapshot.py`의 설정 해시가 런 신원이며, H 캐시 키와 런 인덱스 조회에 모두 쓰인다.

## 의존
`core` (예외 타입).

## 레거시 참조
`v1.0:lsrr/config.py` — **개작**. 계층 해석·dotlist·스윕 전개 로직은 건전하므로 파일 분할해 이식. 단 문자열 섹션을 자동으로 파일로 해석하던 암묵 규칙(`config.py:105-116`)은 `schema.py`의 명시적 선언으로 대체한다 — 암묵 규칙은 설정 오타를 조용히 삼킨다.

## 상태
**검증** — M1 완료. 검증 규칙 7종 + 경고 3종이 로드 시점에 동작.
