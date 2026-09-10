# `lsrr/backbone/` — 동결 백본

## 역할
**동결된 트랜스포머와 닿는 모든 것.** 질문을 단 1회 인코딩해 계층적 사고 메모리의 재료와 연속 디코딩의 재료를 동시에 산출하고, 학습된 사고 표현을 다시 백본에 주입해 답을 생성한다.

이 패키지는 제안서의 두 축을 모두 떠받친다 — §4.1(재료 추출)과 §4.4(답변 생성).

## 경계
- **한다:** 백본 로드·동결·해시, 1회 순전파, 레이어별 상태 추출, 질문 어텐션 풀링, KV 캐시 보유, `h_fusion` 주입과 자기회귀 생성, 분석용 개입 훅.
- **하지 않는다:** 레이어 축 정렬(→ `memory`), 융합(→ `readout`), 사이클 반복(→ `recurrence`). **백본은 절대 사이클 루프 안에서 호출되지 않는다** (I2).

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `session.py` | `BackboneSession` — 로드·1회 순전파·`ContextBundle` 산출. I2 인코딩 카운터 소유 | 계획 |
| `freeze.py` | `requires_grad=False` 강제, `eval()` 고정, 가중치 해시 (I1) | 계획 |
| `extractor.py` | 레이어별 마지막 토큰 수직 1열 `H_last` | 계획 |
| `pooling.py` | 레이어별 질문 전체 어텐션 풀링 `H_pool` (§4.1, ADR-004) | 계획 |
| `continuation.py` | `h_fusion` 주입 + KV 재사용 → teacher-forcing 로짓 / greedy 생성 (§4.4, ADR-001) | 계획 |
| `interventions.py` | back-patching 오라클, logit-lens 투영 훅 (분석 전용, M7) | 계획 |

## 핵심 계약

### 1회 순전파 (`session.py`)

```
encode(input_ids, attention_mask) -> ContextBundle {
    H_last  [B, L, d_in]   # 질문 마지막 토큰 위치의 레이어별 상태
    H_pool  [B, L, d_in]   # 질문 전체 어텐션 풀링, 레이어별
    h_ctx   [B, d_in]      # 백본 원본 h^(L) — 융합 잔차 앵커 (ADR-003)
    kv_cache               # 질문 구간 past_key_values
    attention_mask, meta
}
```

`h_ctx`는 **어댑터를 통과하지 않은** 원본이다. 이것이 ADR-003의 핵심이며, 주입 공간 정합(I8)의 근거다.

### 연속 디코딩 (`continuation.py`)

```
teacher_forced(h_fusion, kv_cache, answer_ids) -> logits [B, T_a, V]
generate(h_fusion, kv_cache, max_new_tokens, eos_id) -> token_ids
```

- `h_fusion`을 질문 다음 위치의 **입력 임베딩**으로 1회 주입하고 질문 KV를 재사용한다.
- 추가 비용은 `[1 위치 + 답 길이]`의 incremental forward뿐이다 — 이 사실이 `metrics/cost.py`의 비용 모델과 일치해야 한다.
- 역전파는 `h_fusion` 위치를 통해서만 흐른다 (I4). 백본 파라미터에는 그래디언트가 축적되지 않는다.

### 동결 (`freeze.py`)
로드 직후 동결하고 가중치 해시를 런 메타에 기록한다. 학습 종료 후 해시를 재확인한다 (I1).

## 의존
`core`. (L1이므로 형제 패키지 import 금지.)

## 레거시 참조
- `Legacy_LSRR/lsrr/backbones/extractor.py` — **개작**. 동결·토크나이저 해시·마지막 활성 토큰 인덱싱은 이식. `position_rule` 택일 구조를 `H_last`+`H_pool` 동시 산출로 확장하고 KV 캐시 반환을 추가한다.
- `Legacy_LSRR/tests/test_freeze.py` — 이식 (I1).

## 주의 사항
1. **`include_embedding`의 의미** — 임베딩 층 출력을 레이어 0으로 포함할지 여부는 `L`을 바꾸므로 캐시 키와 레이어 위치 임베딩 크기에 모두 영향을 준다. 설정에서 명시적으로 고정하고 런 메타에 기록한다.
2. **패딩 방향 (ADR-011)** — 질문은 **좌측 패딩**이 필수다. 연속 디코딩이 질문 KV 뒤에 주입 위치를 이어 붙이므로, 우측 패딩이면 주입이 패딩 뒤에 놓인다. `position_ids`도 명시적으로 넘겨야 한다(`(mask.cumsum(-1)-1).clamp(min=0)`) — 넘기지 않으면 HF가 패딩 포함 절대 인덱스를 쓴다. 실측: 좌측 패딩 + 명시 `position_ids`에서 KV 경로가 전체 순전파와 6.1e-05 이내로 일치.
3. **KV 캐시 메모리** — 배치·시퀀스 길이에 비례해 크다. 학습 배치 크기 상한이 여기서 결정된다 (ADR-002).

## 상태
**검증** — M2 완료. 5개 모듈 구현, GPT-2로 1회 인코딩·연속 디코딩 검증. `interventions.py`는 M7.
