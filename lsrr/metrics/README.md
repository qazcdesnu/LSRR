# `lsrr/metrics/` — 채점과 비용 회계

## 역할
**정확도와 비용을 같은 무게로 다룬다.** 제안서 §6.2: *"모든 표에 정확도와 함께 추가 FLOPs·평균 사이클·지연을 병기하고, 동일 지연 예산에서의 정확도 곡선을 핵심 그림으로 제시한다."*

비용은 부가 정보가 아니라 **주장의 절반**이다. 본 연구의 효율 주장(트랜스포머 1회 + 경량 스캔 M회)은 여기서 측정되지 않으면 존재하지 않는다.

## 경계
- **한다:** 채점 프로토콜, 사이클별 정확도 곡선, FLOPs·지연 회계, 파레토 프론티어, 시드 집계.
- **하지 않는다:** 데이터셋별 정답 정규화(→ `data/scoring.py`). 그림 그리기(→ `reporting/figures.py`).

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `accuracy.py` | 최종답 exact match 프로토콜 (greedy, 샘플별 EOS 절단, 최종답만) | 계획 |
| `anytime.py` | 사이클별 정확도 곡선 (§5의 부산물, 게이트 ② 판정 데이터) | 계획 |
| `cost.py` | FLOPs·지연 회계 (I7) | 계획 |
| `pareto.py` | 정확도 ↔ 연산/지연 프론티어 | 계획 |
| `aggregate.py` | 시드 집계·신뢰구간 (게이트 ③의 유의성 검정 입력) | 계획 |

## 핵심 계약

### 비용 모델 (`cost.py`) — 연속 디코딩 반영
```
CostReport {
    flops_backbone       # 질문 1회 순전파 — 반드시 포함 (I7)
    flops_memory         # 어댑터
    flops_engine         # 사이클당 × 실제 사이클 수
    flops_continuation   # [1 위치 + 답 길이] incremental forward
    latency_ms
    avg_cycles
    flops_includes_backbone: bool
}
```

레거시 대비 바뀐 항: 학습 디코더 FLOPs 항이 사라지고 **연속 디코딩 항**이 들어온다 (ADR-001). 이 항이 작다는 것 — 질문 KV가 재사용되므로 `[1 + 답 길이]`뿐이라는 것 — 이 §4.4의 효율 주장이므로, 측정값이 모델과 어긋나면 구현을 의심한다.

**백본 순전파를 뺀 수치는 보고하지 않는다.** 측정 불가 시 0으로 두지 말고 `flops_includes_backbone: false` 플래그를 세운다(레거시 규약 계승).

### 비교 축
| 비교 대상 | 본 모델 |
|---|---|
| Coconut: 사고 k스텝 = 트랜스포머 **k+1회** 순전파 | 트랜스포머 **1회** + 경량 스캔 M회 |
| Huginn: 재사전학습된 무거운 core 반복 | 동결 백본 + 사후 장착 경량 엔진 |

동일 지연 예산 하의 정확도 곡선(test-time compute scaling)이 **핵심 그림**이며, `pareto.py`가 그 데이터를 만든다.

### anytime 곡선 (`anytime.py`)
판독 경로가 전 사이클 공유이므로(I3) 중간 사이클에서도 답이 판독된다. 사이클별 정확도 곡선은 자동 부산물이며 두 곳에서 쓰인다: **게이트 ②**(M 증가 → 정확도 증가)와 **종료 규칙 분석**의 기반.

## 의존
`core`, `data`(채점 위임). (L5.)

## 레거시 참조
`Legacy_LSRR/lsrr/utils/flops.py`, `Legacy_LSRR/lsrr/training/evaluator.py` — **개작**. 해석적 FLOPs 추정 구조와 `flops_includes_backbone` 플래그 규약, `MissingTargetError` 정책은 계승. 디코더 항을 연속 디코딩 항으로 교체.
`Legacy_LSRR/tests/test_eval_protocol.py`(361줄) — 평가 프로토콜의 함정이 축적된 테스트다. 개작해 이식.

## 상태
계획 — M5 (`pareto.py`는 M6)
