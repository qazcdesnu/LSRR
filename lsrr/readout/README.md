# `lsrr/readout/` — 판독 경로 (전 사이클 공유)

## 역할
정제된 사고 메모리에서 답을 **읽어내는 단일 경로**. 제안서 §4.4에 대응한다.

$$\alpha_l = \mathrm{softmax}(w^\top r_l^*),\quad h_{SSM} = \sum_l \alpha_l r_l^*,\quad h_{fusion} = h^{(L)} + W_r\,h_{SSM}$$

이후 `h_fusion`을 질문 다음 위치에 주입하고 동결 백본이 답을 생성한다.

## 경계
- **한다:** 레이어 축 어텐션 풀링, 잔차 융합, 주입 공간 정합, 백본 연속 디코딩 호출, 디코딩 유틸.
- **하지 않는다:** 학습 가능한 디코더를 두지 않는다 (ADR-001 / P5). 백본 순전파 메커니즘 자체는 `backbone/continuation.py`가 소유하고 여기서는 호출만 한다.

## 왜 별도 패키지인가
제안서 §5: *"풀링·융합·주입의 판독 경로는 전 사이클 공유(사이클별 헤드 금지) — 임의 시점 종료에도 답이 판독되는 anytime 성질의 원천이다."*

이 성질을 지키려면 판독이 **하나의 객체**여야 하고, 모든 사이클이 그 **동일 인스턴스**를 호출해야 한다. 융합 헤드와 디코더가 따로 떠다니며 딕셔너리로 전달되면(레거시 방식) 이는 검증이 아니라 신뢰의 문제가 된다. 단일 객체로 만들면 인스턴스 동일성으로 검사할 수 있다 (I3, ADR-005).

## 모듈

| 모듈 | 역할 | 상태 |
|---|---|---|
| `fusion.py` | α 어텐션 풀링 + `h_fusion` 구성. 융합 방식: `residual`(기본) / `gate` / `concat` | 계획 |
| `injection.py` | 주입 공간 정합 — `h_fusion`이 백본 입력 임베딩 공간에 있음을 보장, 필요 시 노름 보정 (I8) | 계획 |
| `path.py` | `ReadoutPath` — 융합 → 주입 → 연속 디코딩을 잇는 **공유 단일 객체** | 계획 |
| `decode.py` | teacher-forcing 정렬, greedy 디코딩, EOS 절단 | 계획 |

## 핵심 계약

```
ReadoutPath.readout(R, h_ctx, context) -> ReadoutResult { logits, h_fusion, alpha }
```

- `R`은 최종 상태 `R*`일 수도, 중간 사이클 상태 `R^(m)`일 수도 있다 — **경로는 동일하다.**
- `h_ctx`는 백본 원본 `h^(L)`이며 어댑터를 통과하지 않았다 (ADR-003).
- `W_r: R^{d_model} → R^{d_in}` — 사고 표현을 백본 공간으로 되돌린다. `fusion_type="residual"`이면 출력 폭이 반드시 `d_in`이어야 하고, `config/validate.py`가 로드 시점에 확인한다.

### 주입 공간 정합 (`injection.py`)
`h_fusion`은 백본이 **입력 임베딩 자리에서** 받는 벡터다. 두 가지를 확인한다.
1. **차원** — `d_in`과 일치 (I8, 테스트로 강제).
2. **스케일** — 백본 입력 임베딩의 전형적 노름과 크게 어긋나면 학습이 불안정해진다. 노름 보정은 기본 off이며, 불안정이 관측될 때 켠다. 켜졌는지는 런 메타에 기록한다.

`h_ctx` 자체가 백본의 마지막 레이어 출력이므로 (Coconut의 연속 사고 주입과 같은 통로) 잔차 형태 `h_ctx + W_r·h_SSM`는 `W_r` 출력이 작을 때 자연히 유효한 주입 벡터가 된다. 따라서 **`W_r`을 작게 초기화하는 것이 기본**이다 — 학습 초기에 판독이 백본의 원래 동작(No-CoT)에서 출발하고, 사고 기여분이 점진적으로 더해진다.

### α 가중의 부수 산출
`alpha`는 손실에 쓰이지 않지만 반드시 트레이스에 기록한다 — `analysis/alpha_profile.py`가 "중간층 집중" 예측(§7 메커니즘 분석 ii)을 검증하는 데이터다.

## 의존
`core`, `backbone`. (L2.)

## 레거시 참조
- `v1.0:lsrr/fusion/attention_pooling.py` — **개작**. α 풀링과 residual/gate/concat 분기는 이식. 앵커를 어댑터 출력(`R0[:, -1, :]`)에서 백본 원본 `h_ctx`로 교체하고 `W_r` 출력 폭을 `d_in`으로 고정한다 (ADR-003).
- `v1.0:lsrr/decoders/` — **폐기** (ADR-001).
- `v1.0:tests/test_fusion_residual.py` — 앵커 교체를 반영해 `tests/contracts/test_injection_space.py`로 개작.

## 상태
**검증** — `injection`(M2)·`fusion`·`path`(M3) 완료. `decode.py`는 평가 경로(M5)와 함께.
