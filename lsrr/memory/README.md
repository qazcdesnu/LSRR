# `lsrr/memory/` — 계층적 사고 메모리 (H → R⁰)

## 역할
백본이 내놓은 원재료를 **엔진이 읽고 쓸 수 있는 사고 메모리** `R⁰`로 만든다. 제안서 §4.1에 대응한다.

핵심 문제의식: 층간 표현 공간은 정렬되어 있지 않다(tuned lens가 레이어별 변환기를 필요로 한 이유, 층별 노름 증가). 그대로 스캔하면 레이어 축은 의미 있는 시퀀스가 아니다. 이 패키지가 그 정렬을 담당한다.

## 경계
- **한다:** 문맥 구성(마지막 토큰 열 + 질문 풀링 열), 레이어 범위 선택, 레이어별 아핀 변환·정규화, 레이어 위치 임베딩.
- **하지 않는다:** 백본 호출(→ `backbone`), 반복(→ `recurrence`), 융합(→ `readout`). 토큰 축 연산은 일절 하지 않는다 — 여기 도착한 시점에 토큰 축은 이미 사라져 있다.

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `composer.py` | `H_last` + `H_pool` 결합 → `H`. 방식: `last_only` / `add` / `concat` / `gate`(기본) | 계획 |
| `scoping.py` | 레이어 범위 선택 — **Ablation A**: `all_layers` / `final_only` / `mid_band` / `late_band` | 계획 |
| `adapters.py` | `per_layer_affine(+rmsnorm)`(기본) / `shared_affine` / `identity` | 계획 |
| `layer_embedding.py` | 레이어 위치 임베딩 (학습형 / 고정 사인형 / 없음) | 계획 |
| `pipeline.py` | 위 넷을 하나의 `nn.Module`로 조립 → `R⁰` | 계획 |

## 핵심 계약

```
LayerMemoryPipeline(ContextBundle) -> R0 [B, L', d_model]
```

순서는 고정이다: **compose → scope → adapt → layer_pos_emb**.
- scope가 adapt보다 **먼저**인 이유: 선택되지 않은 레이어에 어댑터 파라미터를 할당하지 않기 위해서다. Ablation A의 각 조건이 서로 다른 파라미터 수를 갖는 것을 막으려면 `budget.py`의 정합 대상에 어댑터도 포함해야 한다 — 비교 시 이 점을 보고에 명시한다.
- `d_model`은 `d_in`과 독립이다 (ADR-003). 엔진 폭을 줄여 §5의 "백본 대비 3% 이내" 예산을 지키는 주된 노브가 여기다.

### Ablation A의 구간 정의
`mid_band`·`late_band`의 경계는 백본 레이어 수의 비율로 지정한다(예: `mid_band = [0.33L, 0.66L)`). 절대 인덱스로 지정하면 백본을 바꿀 때 의미가 달라진다. 구간 정의는 `configs/ablation/A_memory_scope.yaml`에 고정하고 백본별로 재계산한다.

## 의존
`core`. (L1.)

## 레거시 참조
- `Legacy_LSRR/lsrr/adapters/layer_adapter.py` — **이식**. `torch.einsum("bli,lio->blo")` 기반 per-layer affine, RMSNorm, 레이어 위치 임베딩이 §4.1에 정확히 대응한다. 레이어 위치 임베딩만 별도 모듈로 분리.
- 레거시에 `composer`·`scoping`은 **없다** — 각각 ADR-004와 Ablation A의 신규 요구다.

## 상태
**검증** — M3 완료. 5개 모듈, Ablation A 범위 4종 포함.
